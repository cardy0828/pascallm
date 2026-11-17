package sampling

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

// referenceFilter は、全部の候補を確率の高い順に並べてから top_k・top_p・min_p を適用する素直な実装です
// （Sampler.filter が候補を絞っても同じ結果になるかを比べるため）。
func referenceFilter(logits []float32, params Params) ([]int, []float64) {
	maxLogit := float64(slices.Max(logits))
	inverseTemperature := 1 / params.Temperature // filter と同じ丸めにするため、割り算ではなく逆数を掛ける
	weights := make([]float64, len(logits))
	total := 0.0
	for id, logit := range logits {
		weights[id] = math.Exp((float64(logit) - maxLogit) * inverseTemperature)
		total += weights[id]
	}
	order := make([]int, len(logits))
	for id := range order {
		order[id] = id
	}
	slices.SortStableFunc(order, func(a, b int) int {
		switch {
		case weights[a] > weights[b]:
			return -1
		case weights[a] < weights[b]:
			return 1
		default:
			return 0
		}
	})
	probabilities := make([]float64, len(order))
	for index, id := range order {
		probabilities[index] = weights[id] / total
	}
	if params.TopK > 0 && len(order) > params.TopK {
		order, probabilities = order[:params.TopK], probabilities[:params.TopK]
	}
	if params.TopP < 1 {
		// ここでは候補を絞っていないため、残った候補の合計が、そのまま top_p の分母になります。
		order, probabilities = cutTopP(order, probabilities, params.TopP, sum(probabilities))
	}
	if params.MinP > 0 {
		order, probabilities = cutMinP(order, probabilities, params.MinP)
	}
	return order, probabilities
}

// significantProbability は、比べる対象にする最小の確率です（filter は確率 1e-13 程度より下の候補を外すため）。
const significantProbability = 1e-9

// trimInsignificant は、確率が significantProbability 未満の候補を末尾から除きます。
func trimInsignificant(order []int, probabilities []float64) ([]int, []float64) {
	end := len(order)
	for end > 0 && probabilities[end-1] < significantProbability {
		end--
	}
	return order[:end], probabilities[:end]
}

// TestFilterMatchesReference は、候補を絞る filter が、全部を並べる実装と同じ候補・確率を返すかを確かめます。
func TestFilterMatchesReference(t *testing.T) {
	const vocabulary = 5000
	cases := []Params{
		{Temperature: 0.1, TopK: 50, TopP: 1, RepetitionPenalty: 1},
		{Temperature: 1, TopK: 0, TopP: 0.9, RepetitionPenalty: 1},
		{Temperature: 0.7, TopK: 40, TopP: 0.8, MinP: 0.05, RepetitionPenalty: 1},
		{Temperature: 1.5, TopK: 0, TopP: 1, MinP: 0.1, RepetitionPenalty: 1},
		{Temperature: 1, TopK: 0, TopP: 1, RepetitionPenalty: 1},
		{Temperature: 2, TopK: 3000, TopP: 0.99, RepetitionPenalty: 1},
	}
	random := rand.New(rand.NewPCG(1, 2))
	for caseIndex, params := range cases {
		for trial := 0; trial < 20; trial++ {
			logits := make([]float32, vocabulary)
			for id := range logits {
				logits[id] = float32(random.NormFloat64() * 3)
			}
			wantOrder, wantProbabilities := trimInsignificant(referenceFilter(logits, params))
			sampler := New(params, nil, vocabulary)
			gotOrder, gotProbabilities := trimInsignificant(sampler.filter(slices.Clone(logits)))
			if !slices.Equal(gotOrder, wantOrder) {
				t.Fatalf("case %d trial %d: order differs (got %d candidates, want %d)", caseIndex, trial, len(gotOrder), len(wantOrder))
			}
			for index := range wantProbabilities {
				if math.Abs(gotProbabilities[index]-wantProbabilities[index]) > 1e-12 {
					t.Fatalf("case %d trial %d: probability %d differs: %g vs %g", caseIndex, trial, index,
						gotProbabilities[index], wantProbabilities[index])
				}
			}
		}
	}
}

// TestPenalties は、repetition_penalty・presence_penalty・frequency_penalty の効き方を確かめます。
func TestPenalties(t *testing.T) {
	params := Params{Temperature: 0, TopP: 1, RepetitionPenalty: 2, PresencePenalty: 0.5, FrequencyPenalty: 0.25}
	sampler := New(params, []int{0, 1}, 4)
	sampler.Accept(2)
	sampler.Accept(2)
	logits := []float32{4, -1, 3, 1}
	sampler.applyPenalties(logits)
	// 0: プロンプトにある正の値 → 4 / 2。1: 負の値 → -1 × 2。2: 割った後に 0.5 + 0.25 × 2 を引く。3: そのまま。
	want := []float32{2, -2, 1.5 - 1, 1}
	if !slices.Equal(logits, want) {
		t.Fatalf("penalties = %v, want %v", logits, want)
	}
}

// TestChooseIgnoresUnknownTokens は、トークナイザーにない番号（validTokens 以上）を選ばないことを確かめます。
func TestChooseIgnoresUnknownTokens(t *testing.T) {
	sampler := New(Params{Temperature: 0, TopP: 1, RepetitionPenalty: 1}, nil, 3)
	if got := sampler.Choose([]float32{1, 2, 3, 100}); got != 2 {
		t.Fatalf("Choose = %d, want 2", got)
	}
}

// TestSeedIsDeterministic は、同じ seed なら同じ列を選ぶことを確かめます。
func TestSeedIsDeterministic(t *testing.T) {
	params := Params{Temperature: 1, TopP: 1, RepetitionPenalty: 1, Seed: 42}
	logits := []float32{1, 1, 1, 1, 1, 1, 1, 1}
	draw := func() []int {
		sampler := New(params, nil, len(logits))
		var chosen []int
		for range 20 {
			chosen = append(chosen, sampler.Choose(slices.Clone(logits)))
		}
		return chosen
	}
	if first, second := draw(), draw(); !slices.Equal(first, second) {
		t.Fatalf("same seed gave %v and %v", first, second)
	}
}
