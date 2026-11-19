package generate

import (
	"context"
	"github.com/cardy0828/pascallm/internal/gpucore"
	"github.com/cardy0828/pascallm/internal/grammar"
	"github.com/cardy0828/pascallm/internal/sampling"
	"slices"
	"testing"
)

// fakeModel はキャッシュの状態遷移だけを記録します。GPU の数値計算は実 API で別途確認します。
type fakeModel struct {
	canRestore                        bool
	resets, restores, position, saved int
}

// VocabSize は試験用語彙数を返します。
func (m *fakeModel) VocabSize() int { return 256 }

// PrefillChunkSize はテスト用モデルの区切り（512 個）を返します。
func (m *fakeModel) PrefillChunkSize() int { return 512 }

// MaxContext は試験の文脈上限です。
func (m *fakeModel) MaxContext() int { return 8192 }

// AllocatedBytes は GPU を使わないため 0 です。
func (m *fakeModel) AllocatedBytes() int64 { return 0 }

// Close は確保資産がないため何もしません。
func (m *fakeModel) Close() error { return nil }

// Reset は再入力の開始を記録します。
func (m *fakeModel) Reset() error { m.resets++; m.position = 0; return nil }

// Prefill は入力数だけ位置を進めます。
func (m *fakeModel) Prefill(ids []int) error { m.position += len(ids); return nil }

// Feed は 1 個処理して停止トークンを返します。
func (m *fakeModel) Feed(int) (int, error) { m.position++; return 0, nil }

// Forward は試験では使用しません。
func (m *fakeModel) Forward(int) ([]float32, error) { panic("unexpected Forward") }

// CopyLogits は試験では使用しません。
func (m *fakeModel) CopyLogits() ([]float32, error) { panic("unexpected CopyLogits") }

// SaveState は現在位置を記録します。
func (m *fakeModel) SaveState() error { m.saved = m.position; return nil }

// CanRestore は試験ごとに指定した復元可否を返します。
func (m *fakeModel) CanRestore() bool { return m.canRestore }

// RestoreState は呼び出し回数を記録します。
func (m *fakeModel) RestoreState() error { m.restores++; m.position = m.saved; return nil }

// ResetPenalties は試験ではペナルティを使用しないため何もしません。
func (m *fakeModel) ResetPenalties([]int) error { return nil }

// FeedCandidates は試験では使用しません。
func (m *fakeModel) FeedCandidates(int, bool, gpucore.CandidateParams) (gpucore.Candidates, error) {
	panic("unexpected candidates")
}

// TestExpiredSnapshotFallsBack は長い生成後に Gemma の輪状 KV が復元不能でも、エラーにせず再入力することを確認します。
func TestExpiredSnapshotFallsBack(t *testing.T) {
	for _, canRestore := range []bool{true, false} {
		m := &fakeModel{canRestore: canRestore, position: 2000, saved: 2}
		e, err := New(m, 256, []int{0})
		if err != nil {
			t.Fatal(err)
		}
		e.cached = []int{1, 2, 3, 4}
		e.saved = []int{1, 2}
		result, err := e.Generate(context.Background(), Request{Prompt: []int{1, 2, 7}, MaxTokens: 10, Sampling: sampling.Params{RepetitionPenalty: 1, TopP: 1}}, func(int) (bool, error) { return false, nil })
		if err != nil {
			t.Fatal(err)
		}
		if canRestore {
			if result.CachedTokens != 2 || m.restores != 1 || m.resets != 0 {
				t.Fatal(result, m)
			}
		} else {
			if result.CachedTokens != 0 || m.restores != 0 || m.resets != 1 {
				t.Fatal(result, m)
			}
		}
	}
}

// bannedGreedyModel は最大 logits が禁止トークンになる場合の選び直しを検査します。
type bannedGreedyModel struct {
	fakeModel
	copies int
}

// Feed は禁止候補 1 を返し、通常どおり位置を 1 進めます。
func (m *bannedGreedyModel) Feed(int) (int, error) { m.position++; return 1, nil }

// CopyLogits は禁止候補 1 と次点の合法候補 2 を含む分布を返します。
func (m *bannedGreedyModel) CopyLogits() ([]float32, error) {
	m.copies++
	values := make([]float32, 256)
	values[1] = 10
	values[2] = 9
	return values, nil
}

// TestGreedyBanFallback は先読みの argmax が禁止されていても、合法な次点へ選び直すことを確認します。
func TestGreedyBanFallback(t *testing.T) {
	m := &bannedGreedyModel{}
	e, err := New(m, 256, []int{0})
	if err != nil {
		t.Fatal(err)
	}
	chosen := -1
	result, err := e.Generate(context.Background(), Request{Prompt: []int{3}, MaxTokens: 1, Sampling: sampling.Params{RepetitionPenalty: 1, TopP: 1}, Constraint: grammar.NewTokenBan(256, 1)}, func(token int) (bool, error) { chosen = token; return false, nil })
	if err != nil {
		t.Fatal(err)
	}
	if chosen != 2 || m.copies != 1 || result.FullLogitsSteps != 1 || m.position != 1 {
		t.Fatalf("chosen=%d copies=%d position=%d result=%+v", chosen, m.copies, m.position, result)
	}
}

// chunkModel は API と同じ生成経路で各 Prefill 呼び出しの行数を記録する。
type chunkModel struct {
	fakeModel
	chunks []int
}

// PrefillChunkSize は境界の端数も検査できるように 3 を返す。
func (m *chunkModel) PrefillChunkSize() int { return 3 }

// Prefill は処理した行数を残し、試験用モデルの位置も進める。
func (m *chunkModel) Prefill(ids []int) error {
	m.chunks = append(m.chunks, len(ids))
	return m.fakeModel.Prefill(ids)
}

// TestFeedPromptUsesModelChunk は 7 トークンを 3・3・1 行に分割し、最後の入力を別に扱うことを確かめる。
func TestFeedPromptUsesModelChunk(t *testing.T) {
	model := &chunkModel{}
	engine, err := New(model, 256, nil)
	if err != nil {
		t.Fatal(err)
	}
	_, err = engine.Generate(context.Background(), Request{Prompt: []int{1, 2, 3, 4, 5, 6, 7, 8}, MaxTokens: 1, Sampling: sampling.Params{RepetitionPenalty: 1, TopP: 1}}, func(int) (bool, error) { return false, nil })
	if err != nil || !slices.Equal(model.chunks, []int{3, 3, 1}) {
		t.Fatalf("chunks %v, error %v", model.chunks, err)
	}
}
