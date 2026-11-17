package gemma4gpu

import (
	"context"
	"errors"
	"fmt"
	"os"
	"runtime"
	"slices"
	"testing"

	"github.com/cardy0828/pascallm/internal/generate"
	"github.com/cardy0828/pascallm/internal/llamakld"
	"github.com/cardy0828/pascallm/internal/sampling"
)

// selectionConstraint は語彙の狭い領域を交互に許可し、候補拒否と文法状態の進行を検査します。
type selectionConstraint struct{ position int }

// Allows は状態を変えず、偶数位置では100/101、奇数位置では102だけを許可します。
func (c *selectionConstraint) Allows(id int) bool {
	if c.position%2 == 0 {
		return id == 100 || id == 101
	}
	return id == 102
}

// Accept は合法な実トークンだけで状態を1進めます。
func (c *selectionConstraint) Accept(id int) error {
	if !c.Allows(id) {
		return fmt.Errorf("invalid grammar token %d", id)
	}
	c.position++
	return nil
}

// TestMTPSelectionGPU は本物のモデルで同じ seed の列を、MTP 有無・候補/全 logits・制約付きで比較します。
// PASCA_MTP_GPU_TEST=1 のときだけ GTX 1080 Ti を使います。各条件後の同一要求でキャッシュ復元も確認します。
func TestMTPSelectionGPU(t *testing.T) {
	if os.Getenv("PASCA_MTP_GPU_TEST") != "1" {
		t.Skip("GPU validation is opt-in")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	base, err := llamakld.Open(publicTestAsset(t, "data", "m7/gemma12b-c4096.kld"))
	if err != nil {
		t.Fatal(err)
	}
	defer base.Close()
	prompt := base.ChunkTokens(0)[:64]
	prompt[0] = 2
	opts := Options{DeviceUUID: "", MaxContext: 8192, MTPPath: publicTestAsset(t, "models", "gemma-4-12b-it-qat/mtp-gemma-4-12B-it.gguf")}
	if os.Getenv("PASCA_MTP_VARIANT") == "k32-auto7" {
		opts.KVCache = "k-fp32"
		opts.MTPDraftTokens = 7
		opts.MTPAdaptive = true
	}
	model, err := Open(publicTestAsset(t, "data", "gemma4-12b-qat.pq4m"), opts)
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	for _, c := range []struct {
		name             string
		gpu, constrained bool
		temperature      float64
		topK             int
		penalty          float64
	}{
		{"gpu_temperature", true, false, .7, 40, 1.1},
		{"full_temperature", false, false, .9, 0, 1.1},
		{"gpu_full_fallback", true, false, 2, 256, 1},
		{"grammar_greedy", true, true, 0, 0, 1},
		{"grammar_temperature", true, true, .8, 8, 1.1},
	} {
		var want []int
		for _, enabled := range []bool{false, true} {
			if err = model.Reset(); err != nil {
				t.Fatal(err)
			}
			if err = model.SetMTPEnabled(enabled); err != nil {
				t.Fatal(err)
			}
			engine, err := generate.New(model, model.VocabSize(), nil)
			if err != nil {
				t.Fatal(err)
			}
			engine.SetGPUCandidates(c.gpu)
			for repeat := 0; repeat < 2; repeat++ {
				params := sampling.Params{Temperature: c.temperature, TopK: c.topK, TopP: 1, MinP: 0, RepetitionPenalty: c.penalty, PresencePenalty: .2, FrequencyPenalty: .1, Seed: 42}
				if c.name == "gpu_temperature" {
					params.TopP = .95
					params.MinP = .05
				}
				request := generate.Request{Prompt: prompt, MaxTokens: 48, Sampling: params}
				var constraint *selectionConstraint
				if c.constrained {
					constraint = &selectionConstraint{}
					request.Constraint = constraint
				}
				var got []int
				result, err := engine.Generate(context.Background(), request, func(id int) (bool, error) { got = append(got, id); return false, nil })
				if err != nil {
					t.Fatal(err)
				}
				if !enabled && repeat == 0 {
					want = append([]int(nil), got...)
				} else if !slices.Equal(want, got) {
					t.Fatalf("%s enabled=%v repeat=%d output differs", c.name, enabled, repeat)
				}
				if constraint != nil && constraint.position != len(got) {
					t.Fatal("grammar advanced speculatively")
				}
				if repeat == 1 && result.CachedTokens == 0 {
					t.Fatal("snapshot was not reused")
				}
				if model.Position() != len(prompt)+len(got)-1 {
					t.Fatal("logical position differs")
				}
				if enabled {
					stats := model.MTPStats()
					if stats.Batches == 0 || !stats.SelectionDriven {
						t.Fatal("MTP selection path not used")
					}
					if c.constrained && stats.SelectionRejected == 0 {
						t.Fatal("forced rejection path not used")
					}
					t.Logf("%s repeat=%d batches=%d reused=%d rejected=%d full_logits=%d", c.name, repeat, stats.Batches, stats.Accepted, stats.SelectionRejected, result.FullLogitsSteps)
				}
			}
		}
	}
	checkSelectionInterruption(t, model, prompt)
	checkSelectionLogits(t, model, prompt)
}

// checkSelectionInterruption は stop/cancel/handler error の後、同じ Engine が同一 seed の応答を再現するか確かめます。
func checkSelectionInterruption(t *testing.T, model *Model, prompt []int) {
	if err := model.Reset(); err != nil {
		t.Fatal(err)
	}
	if err := model.SetMTPEnabled(true); err != nil {
		t.Fatal(err)
	}
	engine, err := generate.New(model, model.VocabSize(), nil)
	if err != nil {
		t.Fatal(err)
	}
	request := generate.Request{Prompt: prompt, MaxTokens: 32, Sampling: sampling.Params{Temperature: .7, TopK: 40, TopP: .95, RepetitionPenalty: 1, Seed: 7}}
	var want []int
	if _, err = engine.Generate(context.Background(), request, func(id int) (bool, error) { want = append(want, id); return false, nil }); err != nil {
		t.Fatal(err)
	}
	handlerError := errors.New("test handler interruption")
	for _, kind := range []string{"stop", "cancel", "handler-error"} {
		ctx, cancel := context.WithCancel(context.Background())
		seen := 0
		_, err := engine.Generate(ctx, request, func(id int) (bool, error) {
			seen++
			if seen != 5 {
				return false, nil
			}
			switch kind {
			case "stop":
				return true, nil
			case "cancel":
				cancel()
				return false, nil
			default:
				return false, handlerError
			}
		})
		cancel()
		if kind == "stop" && err != nil {
			t.Fatal(err)
		}
		if kind == "cancel" && !errors.Is(err, context.Canceled) {
			t.Fatalf("cancel: %v", err)
		}
		if kind == "handler-error" && !errors.Is(err, handlerError) {
			t.Fatalf("handler: %v", err)
		}
		var got []int
		if _, err = engine.Generate(context.Background(), request, func(id int) (bool, error) { got = append(got, id); return false, nil }); err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(want, got) {
			t.Fatal("interruption changed next request", kind)
		}
		t.Logf("selection interruption %s passed", kind)
	}
}

// checkSelectionLogits は候補を強制採用/拒否した実トークン列で全語彙 logits を通常生成と突き合わせます。
func checkSelectionLogits(t *testing.T, model *Model, prompt []int) {
	if err := model.Reset(); err != nil {
		t.Fatal(err)
	}
	if err := model.SetMTPEnabled(true); err != nil {
		t.Fatal(err)
	}
	if err := model.SetMTPSelectionDriven(true); err != nil {
		t.Fatal(err)
	}
	if err := model.Prefill(prompt[:len(prompt)-1]); err != nil {
		t.Fatal(err)
	}
	var inputs []int
	var outputs [][]float32
	token := prompt[len(prompt)-1]
	for step := 0; step < 24; step++ {
		inputs = append(inputs, token)
		logits, err := model.Forward(token)
		if err != nil {
			t.Fatal(err)
		}
		outputs = append(outputs, append([]float32(nil), logits...))
		token = model.mtp.expected
		if len(model.mtp.pendingInputs) > 0 {
			token = model.mtp.pendingInputs[0]
		}
		if step%5 == 4 {
			token = 100
		}
	}
	if err := model.SetMTPEnabled(false); err != nil {
		t.Fatal(err)
	}
	if err := model.Reset(); err != nil {
		t.Fatal(err)
	}
	if err := model.Prefill(prompt[:len(prompt)-1]); err != nil {
		t.Fatal(err)
	}
	for step, token := range inputs {
		got, err := model.Forward(token)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(outputs[step], got) {
			t.Fatalf("selection full logits differ at %d", step)
		}
	}
	t.Log("selection full logits: 24 positions exactly equal")
}
