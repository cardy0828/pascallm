// Package sampling は、logits から次のトークンを選びます（M3）。
//
// 処理の順は Hugging Face の generate と同じく、ペナルティ → temperature → top_k → top_p → min_p です。
//   - repetition_penalty: プロンプトと生成済みのトークン（重複なし）の logits を、正なら割り、負なら掛ける
//   - presence_penalty・frequency_penalty: 生成済みのトークンについて、OpenAI の定義で引く
//   - temperature が 0 なら、ペナルティの後の最大値を選ぶ（乱数を使わない）
//
// 128,000 個の logits をすべて並べ替えると 1 トークンに数 ms かかるため、候補を絞ってから並べます
// （絞り方は candidates の説明を参照）。結果は全体を並べた場合と同じです。
package sampling

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
)

// Params は選び方の設定です。
type Params struct {
	Temperature       float64 // 0 なら最大値を選ぶ。例: 0.1
	TopK              int     // 0 なら制限なし。例: 50
	TopP              float64 // 1 なら制限なし。例: 0.9
	MinP              float64 // 0 なら制限なし。例: 0.05
	RepetitionPenalty float64 // 1 ならなし。例: 1.1
	PresencePenalty   float64 // 0 ならなし。例: 0.5
	FrequencyPenalty  float64 // 0 ならなし。例: 0.5
	Seed              uint64  // 乱数の種
}

// 設定の範囲です（OpenAI・vLLM の受け付ける範囲に合わせる）。
const (
	maxTemperature = 2.0
	minPenalty     = -2.0
	maxPenalty     = 2.0
)

// Validate は設定が範囲内かを確かめます。例: Params{Temperature: -1}.Validate() → エラー
func (p Params) Validate() error {
	switch {
	case p.Temperature < 0 || p.Temperature > maxTemperature:
		return fmt.Errorf("temperature must be between 0 and %g", maxTemperature)
	case p.TopK < 0:
		return errors.New("top_k must not be negative")
	case p.TopP <= 0 || p.TopP > 1:
		return errors.New("top_p must be in (0, 1]")
	case p.MinP < 0 || p.MinP > 1:
		return errors.New("min_p must be in [0, 1]")
	case p.RepetitionPenalty <= 0:
		return errors.New("repetition_penalty must be positive")
	case p.PresencePenalty < minPenalty || p.PresencePenalty > maxPenalty:
		return fmt.Errorf("presence_penalty must be between %g and %g", minPenalty, maxPenalty)
	case p.FrequencyPenalty < minPenalty || p.FrequencyPenalty > maxPenalty:
		return fmt.Errorf("frequency_penalty must be between %g and %g", minPenalty, maxPenalty)
	}
	return nil
}

// NeedsLogits は、logits を CPU に移す必要があるか（GPU の argmax だけでは選べないか）を返します。
// 例: temperature 0 でペナルティなし → false
func (p Params) NeedsLogits() bool {
	return p.Temperature > 0 || p.RepetitionPenalty != 1 || p.PresencePenalty != 0 || p.FrequencyPenalty != 0
}

// Sampler は 1 回の生成で使う選び手です。生成済みのトークンの回数と乱数の状態を持ちます。
type Sampler struct {
	params        Params
	validTokens   int          // 選んでよい番号の上限（含まない）。トークナイザーの番号の数
	seen          map[int]bool // repetition_penalty の対象（プロンプトと生成済み）
	generated     map[int]int  // 生成済みのトークン → 回数
	random        *rand.Rand
	probabilities []float64 // 候補の確率（作業領域）
	order         []int     // 候補の番号（作業領域）
}

// New は Sampler を作ります。prompt は repetition_penalty の対象に入れるプロンプトのトークン列、
// validTokens は選んでよい番号の上限です。例: New(params, promptIDs, 125033)
func New(params Params, prompt []int, validTokens int) *Sampler {
	s := &Sampler{
		params:      params,
		validTokens: validTokens,
		seen:        make(map[int]bool),
		generated:   make(map[int]int),
		random:      rand.New(rand.NewPCG(params.Seed, params.Seed^seedMixer)),
	}
	for _, id := range prompt {
		s.seen[id] = true
	}
	return s
}

// seedMixer は、PCG の 2 つ目の種を 1 つ目から作るための定数です（任意の奇数）。
const seedMixer = 0x9E3779B97F4A7C15

// Accept は、選んだトークンを記録します（次のペナルティに使う）。例: s.Accept(id)
func (s *Sampler) Accept(id int) {
	s.seen[id] = true
	s.generated[id]++
}

// Choose は logits（長さはモデルの語彙）から次のトークンを選びます。logits は書き換えます。
// 例: Choose(model.Forward(...) の戻り値) → 279
func (s *Sampler) Choose(logits []float32) int {
	s.applyPenalties(logits)
	if s.params.Temperature == 0 {
		return argmax(logits[:s.validTokens])
	}
	return s.sample(logits[:s.validTokens])
}

// applyPenalties は repetition_penalty・presence_penalty・frequency_penalty を logits に適用します。
func (s *Sampler) applyPenalties(logits []float32) {
	if s.params.RepetitionPenalty != 1 {
		penalty := float32(s.params.RepetitionPenalty)
		for id := range s.seen {
			if id >= len(logits) {
				continue
			}
			if logits[id] > 0 {
				logits[id] /= penalty
			} else {
				logits[id] *= penalty
			}
		}
	}
	if s.params.PresencePenalty != 0 || s.params.FrequencyPenalty != 0 {
		for id, count := range s.generated {
			if id >= len(logits) {
				continue
			}
			logits[id] -= float32(s.params.PresencePenalty + s.params.FrequencyPenalty*float64(count))
		}
	}
}

// argmax は最大値の番号を返します（同じ値なら小さい番号）。例: argmax([1, 3, 3]) → 1
func argmax(values []float32) int {
	best := 0
	for index, value := range values {
		if value > values[best] {
			best = index
		}
	}
	return best
}

// negligibleScaledLogit は、(logit - 最大値) / temperature がこれより小さいトークンを候補から外す値です。
// 確率は最大のトークンの exp(-30)（約 1e-13）倍以下で、float32 の logits の精度より小さいため、結果に影響しません。
const negligibleScaledLogit = -30.0

// candidateThresholdStart は候補を絞る確率のしきい値の最初の値、candidateThresholdDivisor は
// 候補が足りないときにしきい値を割る値、candidateThresholdFloor はしきい値の下限（これより下は全部を候補にする）です。
const (
	candidateThresholdStart   = 1e-3
	candidateThresholdDivisor = 16
	candidateThresholdFloor   = 1e-12
)

// sample は temperature・top_k・top_p・min_p を適用して、確率に従って 1 個選びます。
func (s *Sampler) sample(logits []float32) int {
	return s.draw(s.filter(logits))
}

// draw は、候補の確率に比例する重みで 1 個選びます（確率の合計は 1 でなくてよい）。
// 例: 確率 [0.6, 0.2] なら、3 : 1 の割合で order[0]・order[1] を選ぶ
func (s *Sampler) draw(order []int, probabilities []float64) int {
	target := s.random.Float64() * sum(probabilities)
	for index, probability := range probabilities {
		target -= probability
		if target < 0 {
			return order[index]
		}
	}
	return order[len(order)-1]
}

// filter は temperature・top_k・top_p・min_p を適用し、残った候補を確率の高い順に返します
// （確率は temperature を適用した全体での値で、合計は 1 とは限らない）。
func (s *Sampler) filter(logits []float32) ([]int, []float64) {
	s.collect(logits)
	s.narrow()
	order, probabilities := s.order, s.probabilities
	// mass は top_p の割合の分母です。top_k で切ったなら残った k 個の確率の合計（Hugging Face と同じく、
	// top_k の後に残った候補の中で比べる）、切っていなければ全体の 1 です。narrow で減らした候補の合計で
	// 割ってはいけません（しきい値より下の候補の分だけ割合が大きくなり、早く切りすぎる）。
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
	return order, probabilities
}

// collect は、無視できないトークンの番号と確率（temperature を適用し、合計 1 に正規化）を
// s.order・s.probabilities に入れます（並びは番号順のまま）。
func (s *Sampler) collect(logits []float32) {
	maxLogit := float64(logits[argmax(logits)])
	inverseTemperature := 1 / s.params.Temperature
	s.order, s.probabilities = s.order[:0], s.probabilities[:0]
	total := 0.0
	for id, logit := range logits {
		scaled := (float64(logit) - maxLogit) * inverseTemperature
		if scaled < negligibleScaledLogit {
			continue
		}
		weight := math.Exp(scaled)
		s.order = append(s.order, id)
		s.probabilities = append(s.probabilities, weight)
		total += weight
	}
	for index := range s.probabilities {
		s.probabilities[index] /= total
	}
}

// narrow は、候補を確率の高い順に並べます。全部を並べると遅いため、確率が「しきい値」以上の候補だけを
// 並べ、top_k・top_p・min_p の結果が変わらないだけの候補が残るまで、しきい値を下げます。
// しきい値より下の候補は、残した候補のどれよりも確率が低いため、上位からの切り取りには影響しません。
func (s *Sampler) narrow() {
	threshold := candidateThresholdStart
	if s.needsAllCandidates() {
		threshold = 0
	}
	maxProbability := slices.Max(s.probabilities)
	for {
		kept, covered := s.countAbove(threshold)
		if kept == len(s.order) || threshold == 0 || s.enoughCandidates(kept, covered, threshold, maxProbability) {
			s.keepAbove(threshold)
			break
		}
		threshold /= candidateThresholdDivisor
		if threshold < candidateThresholdFloor {
			threshold = 0
		}
	}
	s.sortByProbability()
}

// needsAllCandidates は、top_k・top_p・min_p のどれも使わず、全部の候補が必要かを返します。
func (s *Sampler) needsAllCandidates() bool {
	return s.params.TopK == 0 && s.params.TopP >= 1 && s.params.MinP == 0
}

// enoughCandidates は、確率がしきい値以上の候補（kept 個、確率の合計 covered）だけで結果が変わらないかを返します。
//   - top_k を使う: kept が top_k 以上（top_p・min_p はその中で切るため、ほかの条件は要らない）
//   - top_p を使う: covered が top_p 以上
//   - min_p だけ: しきい値が「最大の確率 × min_p」以下
func (s *Sampler) enoughCandidates(kept int, covered, threshold, maxProbability float64) bool {
	switch {
	case s.params.TopK > 0:
		return kept >= s.params.TopK
	case s.params.TopP < 1:
		return covered >= s.params.TopP
	default:
		return threshold <= maxProbability*s.params.MinP
	}
}

// countAbove は、確率がしきい値以上の候補の数と確率の合計を返します。
func (s *Sampler) countAbove(threshold float64) (int, float64) {
	kept, covered := 0, 0.0
	for _, probability := range s.probabilities {
		if probability >= threshold {
			kept++
			covered += probability
		}
	}
	return kept, covered
}

// keepAbove は、確率がしきい値以上の候補だけを残します。
func (s *Sampler) keepAbove(threshold float64) {
	kept := 0
	for index, probability := range s.probabilities {
		if probability >= threshold {
			s.order[kept], s.probabilities[kept] = s.order[index], probability
			kept++
		}
	}
	s.order, s.probabilities = s.order[:kept], s.probabilities[:kept]
}

// sortByProbability は、候補を確率の高い順（同じなら番号の小さい順）に並べます。
func (s *Sampler) sortByProbability() {
	indexes := make([]int, len(s.order))
	for index := range indexes {
		indexes[index] = index
	}
	slices.SortFunc(indexes, func(a, b int) int {
		switch {
		case s.probabilities[a] > s.probabilities[b]:
			return -1
		case s.probabilities[a] < s.probabilities[b]:
			return 1
		default:
			return s.order[a] - s.order[b]
		}
	})
	order := make([]int, len(indexes))
	probabilities := make([]float64, len(indexes))
	for position, index := range indexes {
		order[position], probabilities[position] = s.order[index], s.probabilities[index]
	}
	s.order, s.probabilities = order, probabilities
}

// cutTopP は、確率の高い順の候補のうち、確率を mass で割った割合の合計が topP に達するまでを残します
// （達した 1 個を含む）。mass は割合の分母です（filter の説明を参照）。
// 例: 確率 [0.5, 0.3, 0.2]、topP 0.7、mass 1 → 先頭 2 個
func cutTopP(order []int, probabilities []float64, topP, mass float64) ([]int, []float64) {
	cumulative := 0.0
	for index, probability := range probabilities {
		cumulative += probability / mass
		if cumulative >= topP {
			return order[:index+1], probabilities[:index+1]
		}
	}
	return order, probabilities
}

// sum は値の合計を返します。例: sum([0.5, 0.25]) → 0.75
func sum(values []float64) float64 {
	total := 0.0
	for _, value := range values {
		total += value
	}
	return total
}

// cutMinP は、確率が「最大の確率 × minP」未満の候補を除きます（先頭は必ず残る）。
// 例: 確率 [0.5, 0.3, 0.01]、minP 0.1 → 先頭 2 個
func cutMinP(order []int, probabilities []float64, minP float64) ([]int, []float64) {
	limit := probabilities[0] * minP
	for index, probability := range probabilities {
		if probability < limit {
			return order[:index], probabilities[:index]
		}
	}
	return order, probabilities
}
