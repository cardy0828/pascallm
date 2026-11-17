package sampling

import (
	"math"
	"slices"
)

// この file は、選んでよいトークンを制限して選ぶ処理です（制約付き生成、M6）。
//
// 制限（マスク）は、ペナルティの後、temperature・top_k・top_p・min_p の前に適用します（vLLM・llama.cpp の
// 文法と同じ）。つまり、選べないトークンの logits を -∞ にしてから、ふだんと同じ選び方をします。
// 語彙 12.5 万個すべてを毎回確かめると遅いため、値の大きい順に確かめ、結果が変わらないところで止めます。

// Allows は、トークンを選んでよいかを返す関数です。例: constraint.Allows
type Allows func(id int) bool

// ChooseFromCandidatesMasked は、GPU で絞った候補のうち allows が許すものだけから選びます。
// 候補の外に、結果を変えうるトークンが残っている場合は ok = false で、乱数を使いません
// （呼び出し側は logits 全体で ChooseMasked を使う）。
//   - temperature 0: 許される最初の候補（候補の外のトークンは候補より値が小さいため、これが最大）
//   - top_k を使い、許される候補が top_k 個以上: その先頭 top_k 個で決まる
//   - それ以外: 候補の外のトークン（値は最後の候補以下）が、許される最大の候補に比べて無視できる確率なら決まる
//
// 例: next, ok := s.ChooseFromCandidatesMasked(candidates, constraint.Allows)
func (s *Sampler) ChooseFromCandidatesMasked(candidates Candidates, allows Allows) (int, bool) {
	var ids []int
	var logits []float32
	for index, id := range candidates.IDs {
		if int(id) < s.validTokens && allows(int(id)) {
			ids = append(ids, int(id))
			logits = append(logits, candidates.Logits[index])
		}
	}
	if len(ids) == 0 {
		return 0, false
	}
	if s.params.Temperature == 0 {
		return ids[0], true
	}
	enoughForTopK := s.params.TopK > 0 && len(ids) >= s.params.TopK
	lastCandidate := candidates.Logits[len(candidates.Logits)-1]
	if !enoughForTopK && !negligibleBelow(lastCandidate, logits[0], s.params.Temperature) {
		return 0, false
	}
	return s.chooseAmong(ids, logits), true
}

// negligibleBelow は、値が limit 以下のトークンの確率が、値 best のトークンに比べて無視できるかを返します
// （collect が候補から外す基準と同じ）。例: (limit 10, best 15, temperature 0.1) → true（-50 < -30）
func negligibleBelow(limit, best float32, temperature float64) bool {
	return (float64(limit)-float64(best))/temperature < negligibleScaledLogit
}

// ChooseMasked は、logits 全体（長さはモデルの語彙）から、allows が許すトークンだけを候補にして選びます。
// ペナルティを適用してから、値の大きい順に allows を確かめます。logits は書き換えます。
// 許されるトークンが 1 個もなければ ok = false です。
// 例: next, ok := s.ChooseMasked(logits, constraint.Allows)
func (s *Sampler) ChooseMasked(logits []float32, allows Allows) (int, bool) {
	s.applyPenalties(logits)
	ids, values := s.allowedTop(logits[:s.validTokens], allows)
	if len(ids) == 0 {
		return 0, false
	}
	if s.params.Temperature == 0 {
		best := 0
		for index := range ids {
			if values[index] > values[best] || (values[index] == values[best] && ids[index] < ids[best]) {
				best = index
			}
		}
		return ids[best], true
	}
	// chooseAmong は値の大きい順の並びを受け取るため、並べ替えます（同じ値なら番号の小さい順）。
	order := make([]int, len(ids))
	for index := range order {
		order[index] = index
	}
	slices.SortFunc(order, func(a, b int) int {
		switch {
		case values[a] > values[b]:
			return -1
		case values[a] < values[b]:
			return 1
		default:
			return ids[a] - ids[b]
		}
	})
	sortedIDs := make([]int, len(order))
	sortedValues := make([]float32, len(order))
	for position, index := range order {
		sortedIDs[position], sortedValues[position] = ids[index], values[index]
	}
	return s.chooseAmong(sortedIDs, sortedValues), true
}

// maskScanFirstWidth は、allowedTop が最初に調べる値の幅（最大値からの差）です。足りなければ倍にします。
const maskScanFirstWidth = 1.0

// allowedTop は、選び方の結果を変えうる、許されるトークンを集めます。値の大きい順に幅を広げながら調べ、
// 調べた範囲（値が cut 以上）の許されるトークンについて次のどれかが成り立ったら止めます。
//   - temperature 0 で、1 個以上見つかった（最大のものはその中にある）
//   - top_k を使い、top_k 個以上見つかった
//   - 見つかった最大の値から cut までの差が、無視できる確率の幅を超えた（cut より下は結果に影響しない）
//   - 全部調べた
//
// 返す並びは番号の順です。
func (s *Sampler) allowedTop(logits []float32, allows Allows) ([]int, []float32) {
	maxLogit := logits[argmax(logits)]
	minLogit := slices.Min(logits)
	var ids []int
	var values []float32
	upper := float32(math.Inf(1)) // 調べ終えた範囲の下限（この値以上は調べた）
	width := float32(maskScanFirstWidth)
	for {
		cut := maxLogit - width
		for id, value := range logits {
			if value >= cut && value < upper && allows(id) {
				ids = append(ids, id)
				values = append(values, value)
			}
		}
		upper = cut
		if s.enoughAllowed(values, cut) || cut <= minLogit {
			return ids, values
		}
		width *= 2
	}
}

// enoughAllowed は、値が cut 以上の許されるトークン（values）だけで、選び方の結果が決まるかを返します。
func (s *Sampler) enoughAllowed(values []float32, cut float32) bool {
	if len(values) == 0 {
		return false
	}
	if s.params.Temperature == 0 {
		return true
	}
	if s.params.TopK > 0 && len(values) >= s.params.TopK {
		return true
	}
	return negligibleBelow(cut, slices.Max(values), s.params.Temperature)
}

// chooseAmong は、許された候補（値の大きい順）だけを全体として、temperature・top_k・top_p・min_p を適用して
// 1 個選びます。確率は候補の中で正規化します（選べないトークンの logits を -∞ にしたのと同じ）。
func (s *Sampler) chooseAmong(ids []int, logits []float32) int {
	inverseTemperature := 1 / s.params.Temperature
	probabilities := make([]float64, len(ids))
	total := 0.0
	for index := range ids {
		probabilities[index] = math.Exp((float64(logits[index]) - float64(logits[0])) * inverseTemperature)
		total += probabilities[index]
	}
	for index := range probabilities {
		probabilities[index] /= total
	}
	order := ids
	mass := 1.0
	if s.params.TopK > 0 && len(order) > s.params.TopK {
		order, probabilities = order[:s.params.TopK], probabilities[:s.params.TopK]
		mass = sum(probabilities)
	}
	if s.params.TopP < 1 {
		order, probabilities = cutTopP(order, probabilities, s.params.TopP, mass)
	}
	if s.params.MinP > 0 {
		order, probabilities = cutMinP(order, probabilities, s.params.MinP)
	}
	return s.draw(order, probabilities)
}
