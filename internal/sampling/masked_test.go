package sampling

import (
	"math/rand/v2"
	"slices"
	"testing"
)

// maskedTestVocabulary はテストの語彙の大きさです。
const maskedTestVocabulary = 4096

// randomLogits は乱数の logits を作ります。例: randomLogits(1) → 4096 個
func randomLogits(seed uint64) []float32 {
	random := rand.New(rand.NewPCG(seed, seed))
	logits := make([]float32, maskedTestVocabulary)
	for index := range logits {
		logits[index] = float32(random.NormFloat64() * 3)
	}
	return logits
}

// everyThird は、番号が 3 の倍数のトークンだけを許します。
func everyThird(id int) bool { return id%3 == 0 }

// topCandidates は、logits の上位 count 個を GPU の候補と同じ形にします。
func topCandidates(logits []float32, count int) Candidates {
	order := make([]int, len(logits))
	for index := range order {
		order[index] = index
	}
	slices.SortFunc(order, func(a, b int) int {
		switch {
		case logits[a] > logits[b]:
			return -1
		case logits[a] < logits[b]:
			return 1
		default:
			return a - b
		}
	})
	candidates := Candidates{MaxLogit: logits[order[0]]}
	for _, id := range order[:count] {
		candidates.IDs = append(candidates.IDs, int32(id))
		candidates.Logits = append(candidates.Logits, logits[id])
	}
	return candidates
}

// TestMaskedGreedy は、temperature 0 で、許されるトークンのうち最大のものを選ぶことを確かめます。
func TestMaskedGreedy(t *testing.T) {
	logits := randomLogits(1)
	want := -1
	for id, value := range logits {
		if everyThird(id) && (want < 0 || value > logits[want]) {
			want = id
		}
	}
	params := Params{Temperature: 0, TopP: 1, RepetitionPenalty: 1}
	got, ok := New(params, nil, maskedTestVocabulary).ChooseMasked(slices.Clone(logits), everyThird)
	if !ok || got != want {
		t.Fatalf("ChooseMasked = %d, %v, want %d", got, ok, want)
	}
	got, ok = New(params, nil, maskedTestVocabulary).ChooseFromCandidatesMasked(topCandidates(logits, 128), everyThird)
	if !ok || got != want {
		t.Fatalf("ChooseFromCandidatesMasked = %d, %v, want %d", got, ok, want)
	}
}

// TestMaskedMatchesFullMask は、許されないトークンを -∞ にして Choose した結果と、同じ seed で同じトークンを
// 選ぶことを確かめます（logits 全体の場合と、候補で決まる場合の両方）。
func TestMaskedMatchesFullMask(t *testing.T) {
	settings := []Params{
		{Temperature: 0.7, TopK: 50, TopP: 1, RepetitionPenalty: 1},
		{Temperature: 1.0, TopK: 0, TopP: 0.9, RepetitionPenalty: 1},
		{Temperature: 0.1, TopK: 50, TopP: 1, MinP: 0.05, RepetitionPenalty: 1},
	}
	for seed := uint64(1); seed <= 30; seed++ {
		logits := randomLogits(seed)
		masked := slices.Clone(logits)
		for id := range masked {
			if !everyThird(id) {
				masked[id] = float32(-1e30)
			}
		}
		for _, params := range settings {
			params.Seed = seed
			want := New(params, nil, maskedTestVocabulary).Choose(slices.Clone(masked))
			got, ok := New(params, nil, maskedTestVocabulary).ChooseMasked(slices.Clone(logits), everyThird)
			if !ok || got != want {
				t.Errorf("seed %d %+v: ChooseMasked = %d, want %d", seed, params, got, want)
			}
			// 候補を多めにして、候補だけで決まる場合も確かめます（決まらなければ ok = false で比べない）。
			const manyCandidates = 512
			if fromCandidates, ok := New(params, nil, maskedTestVocabulary).ChooseFromCandidatesMasked(topCandidates(logits, manyCandidates), everyThird); ok && fromCandidates != want {
				t.Errorf("seed %d %+v: ChooseFromCandidatesMasked = %d, want %d", seed, params, fromCandidates, want)
			}
		}
	}
}

// TestMaskedNothingAllowed は、許されるトークンがなければ ok = false を返すことを確かめます。
func TestMaskedNothingAllowed(t *testing.T) {
	params := Params{Temperature: 0.5, TopP: 1, RepetitionPenalty: 1}
	if _, ok := New(params, nil, maskedTestVocabulary).ChooseMasked(randomLogits(2), func(int) bool { return false }); ok {
		t.Fatal("ChooseMasked chose a token although nothing is allowed")
	}
}
