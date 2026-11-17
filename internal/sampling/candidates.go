package sampling

import "math"

// この file は、GPU で絞った候補（lfm3gpu.Candidates）から選ぶ処理です。
// 候補はペナルティを適用した logits の上位 n 個（値の大きい順、同じ値なら番号の小さい順）と、
// 全体の最大値・exp の合計です。候補だけで top_k・top_p・min_p の結果が決まるときだけ選び、
// 決まらないときは ok = false を返します（呼び出し側は logits 全体で Choose を使う）。

// Candidates は、GPU で絞った候補です（lfm3gpu.Candidates と同じ内容）。
type Candidates struct {
	IDs      []int32   // 値の大きい順の番号
	Logits   []float32 // ペナルティを適用した値（temperature は掛けていない）
	MaxLogit float32   // 全体の最大値
	ExpSum   float32   // 全体の Σ exp((値 - MaxLogit) / temperature)
}

// ChooseFromCandidates は候補から次のトークンを選びます。候補だけでは決まらない場合は ok = false で、
// 乱数を使いません（logits 全体で Choose を呼んでも、同じ seed の列が保たれる）。
// ペナルティは GPU で適用済みなので、ここでは適用しません（Accept での記録は従来どおり必要）。
// 例: next, ok := s.ChooseFromCandidates(candidates)
func (s *Sampler) ChooseFromCandidates(candidates Candidates) (int, bool) {
	if len(candidates.IDs) == 0 {
		return 0, false
	}
	if s.params.Temperature == 0 {
		return int(candidates.IDs[0]), true
	}
	order, probabilities, ok := s.filterCandidates(candidates)
	if !ok {
		return 0, false
	}
	return s.draw(order, probabilities), true
}

// filterCandidates は、候補に temperature・top_k・top_p・min_p を適用し、残った候補を確率の高い順に返します。
// 確率は全体での値（ExpSum で割る）です。候補の外に、残るべきトークンがありうる場合は ok = false です。
//   - top_k を使う: top_k が候補の数以下なら決まる（top_p の分母は top_k 個の合計）
//   - top_k を使わず top_p を使う: 候補の確率の合計が top_p に達すれば決まる
//   - min_p だけ: 最後の候補の確率が「最大の確率 × min_p」未満なら決まる（それより下は全部除かれる）
//   - どれも使わない: 全体が必要なので決まらない
func (s *Sampler) filterCandidates(candidates Candidates) ([]int, []float64, bool) {
	count := len(candidates.IDs)
	inverseTemperature := 1 / s.params.Temperature
	order := make([]int, count)
	probabilities := make([]float64, count)
	for index := range count {
		order[index] = int(candidates.IDs[index])
		scaled := (float64(candidates.Logits[index]) - float64(candidates.MaxLogit)) * inverseTemperature
		probabilities[index] = math.Exp(scaled) / float64(candidates.ExpSum)
	}

	mass := 1.0
	switch {
	case s.params.TopK > 0:
		if s.params.TopK > count {
			return nil, nil, false
		}
		order, probabilities = order[:s.params.TopK], probabilities[:s.params.TopK]
		mass = sum(probabilities)
	case s.params.TopP < 1:
		if sum(probabilities) < s.params.TopP {
			return nil, nil, false
		}
	case s.params.MinP > 0:
		if probabilities[count-1] >= probabilities[0]*s.params.MinP {
			return nil, nil, false
		}
	default:
		return nil, nil, false
	}
	if s.params.TopP < 1 {
		order, probabilities = cutTopP(order, probabilities, s.params.TopP, mass)
	}
	if s.params.MinP > 0 {
		order, probabilities = cutMinP(order, probabilities, s.params.MinP)
	}
	return order, probabilities, true
}
