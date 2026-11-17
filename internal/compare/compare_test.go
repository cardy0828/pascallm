package compare

import (
	"math"
	"testing"
)

// TestAccumulator は、同じ分布なら KL が 0 で上位 1 位が一致し、違う分布なら KL が正になることと、
// 下限以下の基準の値を除くことを確かめます。
func TestAccumulator(t *testing.T) {
	logits := []float32{1, 2, 3}
	logProbs := make([]float64, len(logits))
	LogSoftmax(logits, logProbs)
	base := []float32{float32(logProbs[0]), float32(logProbs[1]), float32(logProbs[2])}
	noFloor := float32(math.Inf(-1))
	const target = 2

	var same Accumulator
	same.Add(base, noFloor, logits, target)
	if summary := same.Summary(); math.Abs(summary.KLMean) > 1e-6 || summary.Top1Agreement != 1 || summary.Positions != 1 {
		t.Errorf("same distribution: %+v", summary)
	}

	reversed := []float32{3, 2, 1}
	var different Accumulator
	different.Add(base, noFloor, reversed, target)
	summary := different.Summary()
	if summary.KLMean <= 0 || summary.Top1Agreement != 0 || summary.TestPerplexity <= summary.BasePerplexity {
		t.Errorf("different distribution: %+v", summary)
	}

	// 下限を最小の値にすると、そのトークンの項（負の寄与）が除かれて、KL の合計が変わる。
	var floored Accumulator
	floored.Add(base, base[0], reversed, target)
	if floored.Summary().KLMean == summary.KLMean {
		t.Errorf("the floor did not remove a term: %g", summary.KLMean)
	}
}
