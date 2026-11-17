package sampling

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

// テストで作る候補の数です（lfm3gpu.CandidateCount と同じ）。
const testCandidateCount = 128

// candidateProbabilityTolerance は、候補から求めた確率と全体から求めた確率の相対差の上限です
// （ExpSum を float32 で受け取るため、相対 1e-7 程度ずれる）。
const candidateProbabilityTolerance = 1e-6

// makeCandidates は、GPU の candidates_final と同じ内容（上位 count 個・最大値・exp の合計）を CPU で作ります。
// 例: makeCandidates([]float32{1, 3, 2}, 2, 1) → IDs [1 2]、MaxLogit 3
func makeCandidates(logits []float32, count int, temperature float64) Candidates {
	order := make([]int, len(logits))
	for id := range order {
		order[id] = id
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
	maxLogit := logits[order[0]]
	expSum := 0.0
	for _, logit := range logits {
		expSum += math.Exp((float64(logit) - float64(maxLogit)) / temperature)
	}
	candidates := Candidates{MaxLogit: maxLogit, ExpSum: float32(expSum)}
	for _, id := range order[:count] {
		candidates.IDs = append(candidates.IDs, int32(id))
		candidates.Logits = append(candidates.Logits, logits[id])
	}
	return candidates
}

// TestFilterCandidatesMatchesReference は、候補だけで決まると判断したときに、全体から求めた結果と同じ候補・確率を
// 返すかを確かめます。top_k が候補の数以下のとき・どの制限もないときは、決まるかどうかも確かめます。
func TestFilterCandidatesMatchesReference(t *testing.T) {
	const vocabulary = 5000
	cases := []struct {
		params   Params
		decision string // "always"（必ず決まる）・"never"（決まらない）・"maybe"（logits による）
	}{
		{Params{Temperature: 0.1, TopK: 50, TopP: 1, RepetitionPenalty: 1}, "always"},
		{Params{Temperature: 0.7, TopK: 40, TopP: 0.8, MinP: 0.05, RepetitionPenalty: 1}, "always"},
		{Params{Temperature: 1, TopK: testCandidateCount, TopP: 0.95, RepetitionPenalty: 1}, "always"},
		{Params{Temperature: 0.3, TopK: 0, TopP: 0.9, RepetitionPenalty: 1}, "maybe"},
		{Params{Temperature: 1, TopK: 0, TopP: 0.9, RepetitionPenalty: 1}, "maybe"},
		{Params{Temperature: 0.5, TopK: 0, TopP: 1, MinP: 0.1, RepetitionPenalty: 1}, "maybe"},
		{Params{Temperature: 2, TopK: 3000, TopP: 0.99, RepetitionPenalty: 1}, "never"},
		{Params{Temperature: 1, TopK: 0, TopP: 1, RepetitionPenalty: 1}, "never"},
	}
	random := rand.New(rand.NewPCG(3, 4))
	decided := 0
	for caseIndex, testCase := range cases {
		params := testCase.params
		for trial := 0; trial < 20; trial++ {
			logits := make([]float32, vocabulary)
			for id := range logits {
				logits[id] = float32(random.NormFloat64() * 3)
			}
			sampler := New(params, nil, vocabulary)
			gotOrder, gotProbabilities, ok := sampler.filterCandidates(makeCandidates(logits, testCandidateCount, params.Temperature))
			switch {
			case testCase.decision == "always" && !ok:
				t.Fatalf("case %d trial %d: candidates should decide the result", caseIndex, trial)
			case testCase.decision == "never" && ok:
				t.Fatalf("case %d trial %d: candidates must not decide the result", caseIndex, trial)
			case !ok:
				continue
			}
			decided++
			wantOrder, wantProbabilities := trimInsignificant(referenceFilter(logits, params))
			gotOrder, gotProbabilities = trimInsignificant(gotOrder, gotProbabilities)
			if !slices.Equal(gotOrder, wantOrder) {
				t.Fatalf("case %d trial %d: order differs (got %d candidates, want %d)", caseIndex, trial, len(gotOrder), len(wantOrder))
			}
			for index := range wantProbabilities {
				relative := math.Abs(gotProbabilities[index]-wantProbabilities[index]) / wantProbabilities[index]
				if relative > candidateProbabilityTolerance {
					t.Fatalf("case %d trial %d: probability %d differs: %g vs %g", caseIndex, trial, index,
						gotProbabilities[index], wantProbabilities[index])
				}
			}
		}
	}
	t.Logf("decided by candidates: %d of %d", decided, len(cases)*20)
}

// TestChooseFromCandidatesGreedy は、temperature 0 なら最初の候補を選ぶことを確かめます。
func TestChooseFromCandidatesGreedy(t *testing.T) {
	sampler := New(Params{Temperature: 0, TopP: 1, RepetitionPenalty: 1}, nil, 10)
	got, ok := sampler.ChooseFromCandidates(Candidates{IDs: []int32{7, 3}, Logits: []float32{5, 4}, MaxLogit: 5, ExpSum: 1})
	if !ok || got != 7 {
		t.Fatalf("ChooseFromCandidates = %d, %v, want 7, true", got, ok)
	}
}
