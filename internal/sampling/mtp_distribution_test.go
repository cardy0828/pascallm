package sampling

import (
	"math"
	"testing"
)

// TestDeterministicDraftCouplingDistribution は点提案 d=0 に対する受理率 p(d) と拒否時補正分布を検査します。
// 通常サンプラーを1回だけ呼ぶ方式で、禁止候補の確率0と強制候補の確率1も確認します。
func TestDeterministicDraftCouplingDistribution(t *testing.T) {
	params := Params{Temperature: 1, TopP: 1, RepetitionPenalty: 1, Seed: 7}
	sampler := New(params, nil, 3)
	const draws = 100000
	counts := [3]int{}
	for i := 0; i < draws; i++ {
		logits := []float32{float32(math.Log(.2)), float32(math.Log(.3)), float32(math.Log(.5))}
		counts[sampler.Choose(logits)]++
	}
	for id, p := range []float64{.2, .3, .5} {
		if math.Abs(float64(counts[id])/draws-p) > .006 {
			t.Fatal(counts)
		}
	}
	rejected := counts[1] + counts[2]
	if math.Abs(float64(counts[1])/float64(rejected)-.375) > .008 {
		t.Fatal("residual distribution", counts)
	}
	for _, allowed := range []int{0, 1} {
		for i := 0; i < 100; i++ {
			id, ok := sampler.ChooseMasked([]float32{1, 2, 3}, func(id int) bool { return id == allowed })
			if !ok || id != allowed {
				t.Fatal("point mass boundary", id)
			}
		}
	}
}
