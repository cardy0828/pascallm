// Package compare は、2 つの実装の次トークン分布の差を集計します（M7）。
//
// 基準の分布（llama.cpp の対数確率など）と、評価する実装の logits を位置ごとに受け取り、
// llama.cpp の llama-perplexity --kl-divergence と同じ考え方の指標をまとめます。
//   - KL(基準 || 評価) の平均・中央値・99 パーセンタイル・最大
//   - 上位 1 位の一致率
//   - 正解のトークン（実際の次のトークン）の対数確率から求めた、両者の perplexity
package compare

import (
	"math"
	"sort"
)

// Summary は集計の結果です。
type Summary struct {
	Positions      int     `json:"positions"`       // 比べた位置の数
	KLMean         float64 `json:"kl_mean"`         // KL(基準 || 評価) の平均
	KLMedian       float64 `json:"kl_median"`       // 中央値
	KLP99          float64 `json:"kl_p99"`          // 99 パーセンタイル
	KLMax          float64 `json:"kl_max"`          // 最大
	Top1Agreement  float64 `json:"top1_agreement"`  // 上位 1 位が同じ位置の割合（0〜1）
	BasePerplexity float64 `json:"base_perplexity"` // 基準の perplexity
	TestPerplexity float64 `json:"test_perplexity"` // 評価する実装の perplexity
	// MaxTopLogProbDiff は、基準の上位 1 位のトークンについて、対数確率の差の絶対値の最大です。
	MaxTopLogProbDiff float64 `json:"max_top_log_prob_diff"`
}

// Accumulator は、位置ごとの結果をためて Summary を作ります。
type Accumulator struct {
	divergences []float64
	sameTop     int
	baseNLL     float64
	testNLL     float64
	maxTopDiff  float64
	testLogProb []float64 // 作業領域（語彙の数）
}

// LogSoftmax は logits の log softmax を dst に書きます（float64 で計算する）。
// 例: [0, 0] -> [-0.6931, -0.6931]
func LogSoftmax(logits []float32, dst []float64) {
	maximum := math.Inf(-1)
	for _, value := range logits {
		maximum = math.Max(maximum, float64(value))
	}
	var sum float64
	for _, value := range logits {
		sum += math.Exp(float64(value) - maximum)
	}
	logSum := maximum + math.Log(sum)
	for index, value := range logits {
		dst[index] = float64(value) - logSum
	}
}

// argmax は最大の要素の位置を返します（同じ値なら小さい位置）。
func argmax[T float32 | float64](values []T) int {
	best := 0
	for index, value := range values {
		if value > values[best] {
			best = index
		}
	}
	return best
}

// Add は 1 位置の結果を加えます。baseLogProbs は基準の対数確率、testLogits は評価する実装の logits、
// target はその位置の正解（実際の次のトークン）です。
// baseFloor 以下の基準の値は「それ以下」としかわからないため、KL の合計から除きます
// （llama.cpp の基準ファイルは、最大値より 16 以上小さい logits を下限に切り上げて保存する）。
// 切り上げがなければ、baseFloor に負の無限大を渡します。
// 例: Add(base, floor, logits, tokens[position+1])
func (a *Accumulator) Add(baseLogProbs []float32, baseFloor float32, testLogits []float32, target int) {
	if len(a.testLogProb) != len(testLogits) {
		a.testLogProb = make([]float64, len(testLogits))
	}
	LogSoftmax(testLogits, a.testLogProb)
	var divergence float64
	for token, base := range baseLogProbs {
		if base <= baseFloor {
			continue
		}
		probability := math.Exp(float64(base))
		divergence += probability * (float64(base) - a.testLogProb[token])
	}
	a.divergences = append(a.divergences, divergence)
	baseTop := argmax(baseLogProbs)
	if baseTop == argmax(a.testLogProb) {
		a.sameTop++
	}
	a.maxTopDiff = math.Max(a.maxTopDiff, math.Abs(float64(baseLogProbs[baseTop])-a.testLogProb[baseTop]))
	a.baseNLL -= float64(baseLogProbs[target])
	a.testNLL -= a.testLogProb[target]
}

// percentile は、昇順に並べた値の p（0〜1）の位置の値を返します。例: percentile([1, 2, 3, 4], 0.5) = 3
func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	index := int(p * float64(len(sorted)))
	return sorted[min(index, len(sorted)-1)]
}

// Summary は、これまでに加えた位置の集計を返します。
func (a *Accumulator) Summary() Summary {
	count := len(a.divergences)
	if count == 0 {
		return Summary{}
	}
	sorted := append([]float64(nil), a.divergences...)
	sort.Float64s(sorted)
	var total float64
	for _, value := range sorted {
		total += value
	}
	const medianPoint, p99Point = 0.5, 0.99
	return Summary{
		Positions:         count,
		KLMean:            total / float64(count),
		KLMedian:          percentile(sorted, medianPoint),
		KLP99:             percentile(sorted, p99Point),
		KLMax:             sorted[count-1],
		Top1Agreement:     float64(a.sameTop) / float64(count),
		BasePerplexity:    math.Exp(a.baseNLL / float64(count)),
		TestPerplexity:    math.Exp(a.testNLL / float64(count)),
		MaxTopLogProbDiff: a.maxTopDiff,
	}
}
