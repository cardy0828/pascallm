package lfm3

import (
	"math"
	"testing"
)

// TestRopeInverseFrequenciesPartial は、部分 RoPE で先頭だけが 0 以外になり、
// 周波数の並びが全次元のもの（1/theta^(2i/head_dim)）のままであることを確かめます。
func TestRopeInverseFrequenciesPartial(t *testing.T) {
	inverse := ropeInverseFrequencies(RopeParameters{Theta: 1e6, PartialRotaryFactor: 0.5, Type: "default"}, 128)
	if len(inverse) != 64 {
		t.Fatalf("len = %d, want 64", len(inverse))
	}
	if inverse[0] != 1 {
		t.Errorf("inverse[0] = %g, want 1", inverse[0])
	}
	want := 1 / math.Pow(1e6, 2.0*31/128)
	if math.Abs(float64(inverse[31])-want) > 1e-6*want {
		t.Errorf("inverse[31] = %g, want %g", inverse[31], want)
	}
	for index := 32; index < 64; index++ {
		if inverse[index] != 0 {
			t.Fatalf("inverse[%d] = %g, want 0", index, inverse[index])
		}
	}
}

// TestApplyRopeRotatesPairs は、次元 i と i+half が組になって回転し、周波数 0 の次元は変わらないことを確かめます。
func TestApplyRopeRotatesPairs(t *testing.T) {
	vector := []float32{1, 5, 0, 7} // half = 2: 組は (0, 2) と (1, 3)
	cos := []float32{0, 1}          // 組 (0, 2) は 90 度回転、組 (1, 3) は回転なし
	sin := []float32{1, 0}
	applyRope(vector, cos, sin)
	want := []float32{0, 5, 1, 7}
	for index := range want {
		if vector[index] != want[index] {
			t.Fatalf("vector = %v, want %v", vector, want)
		}
	}
}

// TestRMSNorm は、二乗平均平方根で割り、重みを掛けることを確かめます。
func TestRMSNorm(t *testing.T) {
	dst := make([]float32, 2)
	rmsNorm(dst, []float32{3, 4}, []float32{1, 2}, 0)
	scale := 1 / math.Sqrt(12.5)
	if math.Abs(float64(dst[0])-3*scale) > 1e-6 || math.Abs(float64(dst[1])-8*scale) > 1e-6 {
		t.Fatalf("rmsNorm = %v", dst)
	}
}
