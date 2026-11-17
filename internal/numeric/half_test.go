package numeric

import (
	"math"
	"testing"
)

// TestFloat16RoundTripAllBits は、NaN 以外の全 65,536 通りの FP16 ビット列が
// FP32 を経由して同じビット列に戻ることを確かめます。
func TestFloat16RoundTripAllBits(t *testing.T) {
	for value := 0; value <= 0xFFFF; value++ {
		bits := uint16(value)
		decoded := Float16ToFloat32(bits)
		if math.IsNaN(float64(decoded)) {
			continue
		}
		if back := Float32ToFloat16(decoded); back != bits {
			t.Fatalf("bits 0x%04X -> %g -> 0x%04X", bits, decoded, back)
		}
	}
}

// TestFloat32ToFloat16Examples は、代表的な値の変換結果（丸め・あふれ・非正規化数）を確かめます。
func TestFloat32ToFloat16Examples(t *testing.T) {
	cases := []struct {
		value float32
		want  uint16
	}{
		{1, 0x3C00},
		{-2, 0xC000},
		{65504, 0x7BFF},                         // FP16 の最大値
		{65520, 0x7C00},                         // 最大値を超えて無限大へ丸まる
		{float32(math.Pow(2, -24)), 0x0001},     // 最小の非正規化数
		{float32(math.Pow(2, -25)), 0x0000},     // ちょうど中間は偶数（0）へ
		{1 + float32(math.Pow(2, -11)), 0x3C00}, // 1 と次の値のちょうど中間は偶数へ
		{1 + 3*float32(math.Pow(2, -11)), 0x3C02},
	}
	for _, c := range cases {
		if got := Float32ToFloat16(c.value); got != c.want {
			t.Errorf("Float32ToFloat16(%g) = 0x%04X, want 0x%04X", c.value, got, c.want)
		}
	}
}

// TestBF16ToFloat32 は、bfloat16 の変換が上位 16 ビットの単純な拡張であることを確かめます。
func TestBF16ToFloat32(t *testing.T) {
	if got := BF16ToFloat32(0x3F80); got != 1 {
		t.Errorf("BF16ToFloat32(0x3F80) = %g, want 1", got)
	}
	if got := BF16ToFloat32(0xC000); got != -2 {
		t.Errorf("BF16ToFloat32(0xC000) = %g, want -2", got)
	}
}
