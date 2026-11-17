package gemma4gpu

import (
	"context"
	"os"
	"github.com/cardy0828/pascallm/internal/generate"
	"github.com/cardy0828/pascallm/internal/llamakld"
	"github.com/cardy0828/pascallm/internal/sampling"
	"runtime"
	"slices"
	"testing"
	"time"
)

// TestPrefixSnapshotGPU は A→B→A、分岐prefix、輪状KV後の復元を実機で検証します。
func TestPrefixSnapshotGPU(t *testing.T) {
	if os.Getenv("PASCA_PREFIX_GPU_TEST") != "1" {
		t.Skip("GPU validation is opt-in")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	base, err := llamakld.Open(publicTestAsset(t, "data", "m7/gemma12b-c4096.kld"))
	if err != nil {
		t.Fatal(err)
	}
	defer base.Close()
	tokens := slices.Clone(base.ChunkTokens(0))
	tokens[0] = 2
	opts := Options{DeviceUUID: "", MaxContext: 8192, MTPPath: publicTestAsset(t, "models", "gemma-4-12b-it-qat/mtp-gemma-4-12B-it.gguf"), KVCache: os.Getenv("PASCA_PREFIX_FORMAT")}
	m, err := Open(publicTestAsset(t, "data", "gemma4-12b-qat.pq4m"), opts)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	a := slices.Clone(tokens[:512])
	b := slices.Clone(a)
	b[8] = 100
	branch := slices.Clone(a)
	branch[300] = 101
	params := sampling.Params{Temperature: .7, TopK: 40, TopP: .95, MinP: .05, RepetitionPenalty: 1.1, Seed: 42}
	run := func(e *generate.Engine, prompt []int) ([]int, generate.Result) {
		var out []int
		result, err := e.Generate(context.Background(), generate.Request{Prompt: prompt, MaxTokens: 24, Sampling: params}, func(id int) (bool, error) { out = append(out, id); return false, nil })
		if err != nil {
			t.Fatal(err)
		}
		return out, result
	}
	cold := func(prompt []int) []int {
		if err := m.Reset(); err != nil {
			t.Fatal(err)
		}
		e, err := generate.New(m, m.VocabSize(), nil)
		if err != nil {
			t.Fatal(err)
		}
		out, _ := run(e, prompt)
		return out
	}
	wantA := cold(a)
	wantB := cold(b)
	wantBranch := cold(branch)
	if err := m.Reset(); err != nil {
		t.Fatal(err)
	}
	e, err := generate.New(m, m.VocabSize(), nil)
	if err != nil {
		t.Fatal(err)
	}
	budget := int64(512 * 1024 * 1024)
	if opts.KVCache == "k-fp32" {
		budget = 1024 * 1024 * 1024
	}
	if err := e.SetPrefixCache(budget, 8); err != nil {
		t.Fatal(err)
	}
	for index, prompt := range [][]int{a, b, a, branch} {
		out, r := run(e, prompt)
		want := wantA
		if index == 1 {
			want = wantB
		}
		if index == 3 {
			want = wantBranch
		}
		if !slices.Equal(out, want) {
			t.Fatalf("case %d output differs", index)
		}
		if index == 2 && (r.CachedTokens != 511 || r.PrefixMetrics.Hits != 1) {
			t.Fatalf("A restore: %+v", r)
		}
		if index == 3 && (r.CachedTokens != 256 || r.PrefixMetrics.Hits != 1) {
			t.Fatalf("shared restore: %+v", r)
		}
		t.Logf("case=%d cached=%d prompt_ms=%.3f cache=%+v", index, r.CachedTokens, float64(r.PromptDuration)/float64(time.Millisecond), r.PrefixMetrics)
	}
	// handler中断後にも別会話を挟んで同じprefixを復元する。
	_, err = e.Generate(context.Background(), generate.Request{Prompt: a, MaxTokens: 24, Sampling: params}, func(int) (bool, error) { return true, nil })
	if err != nil {
		t.Fatal(err)
	}
	run(e, b)
	got, _ := run(e, a)
	if !slices.Equal(got, wantA) {
		t.Fatal("interruption leaked state")
	}
	// speculative候補のないprompt境界を保存し、輪状KVの全スロットを上書きしてから復元する。
	if err := m.Reset(); err != nil {
		t.Fatal(err)
	}
	if err := m.SetMTPEnabled(false); err != nil {
		t.Fatal(err)
	}
	const position = 3500
	for start := 0; start < position; start += 512 {
		if err := m.Prefill(tokens[start:min(start+512, position)]); err != nil {
			t.Fatal(err)
		}
	}
	snapshot, err := m.SnapshotState()
	if err != nil {
		t.Fatal(err)
	}
	bytes := m.SnapshotBytes()
	logits, err := m.Forward(tokens[position])
	if err != nil {
		t.Fatal(err)
	}
	want := slices.Clone(logits)
	if err := (&Model{}).RestoreSnapshot(snapshot); err == nil {
		t.Fatal("foreign snapshot accepted")
	}
	if err := m.Reset(); err != nil {
		t.Fatal(err)
	}
	other := make([]int, position)
	for i := range other {
		other[i] = 100
	}
	for start := 0; start < position; start += 512 {
		if err := m.Prefill(other[start:min(start+512, position)]); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := m.Forward(100); err != nil {
		t.Fatal(err)
	}
	if err := m.RestoreSnapshot(snapshot); err != nil {
		t.Fatal(err)
	}
	logits, err = m.Forward(tokens[position])
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(logits, want) {
		t.Fatal("ring snapshot logits differ")
	}
	t.Logf("ring position=%d bytes=%d all_logits_equal=true", position, bytes)
}
