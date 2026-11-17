// Package pq4 は、Pascal 向けの 4bit 量子化形式 PQ4 の量子化・逆量子化と、CPU での行列 × ベクトルを実装します。
//
// PQ4 の仕様（docs/ARCHITECTURE.md「4.1 PQ4 の基本形」）:
//   - 入力方向に連続する 32 要素を 1 ブロックとし、FP16 の係数 d を 1 個持つ
//   - 各要素は 4bit の符号なし整数 q（0〜15）で、復元値は (q - 8) × d
//   - 4bit 本体は行ごとに [ブロック][16 バイト] と並べ、FP16 の係数は別の配列 [行][ブロック] に置く
//   - 1 ブロック 16 バイトのうち、バイト j の下位 4bit が要素 j、上位 4bit が要素 j+16
//
// CPU の MulVec は、GPU の DP4A カーネルと同じ手順（活性値を 32 要素ごとに int8 化し、整数で内積）で計算します。
// そのため、CPU の参照実装で測った誤差は、GPU 実装の誤差の見積もりとして使えます。
package pq4

import (
	"fmt"
	"math"

	"github.com/cardy0828/pascallm/internal/numeric"
)

// ブロックに関する定数です。
const (
	BlockSize       = 32 // 1 ブロックの要素数
	BlockBytes      = 16 // 1 ブロックの 4bit 本体のバイト数（32 × 4bit）
	halfBlock       = BlockSize / 2
	quantCenter     = 8  // 4bit 値の中心。復元値は (q - quantCenter) × d
	quantMax        = 15 // 4bit 値の最大
	lowNibbleMask   = 0x0F
	highNibbleShift = 4
)

// Method は、ブロックの係数の決め方です。
type Method string

const (
	// MethodAbsMax は、絶対値が最大の要素がちょうど -8 になるように係数を決めます（llama.cpp の Q4_0 と同じ）。
	MethodAbsMax Method = "absmax"
	// MethodMSE は、係数の候補をいくつか試し、ブロック内の二乗誤差が最小のものを選びます。
	// 実行時の計算は MethodAbsMax と同じで、変換時間だけが増えます。
	MethodMSE Method = "mse"
)

// ParseMethod は、コマンドライン引数などの文字列を Method に変換します。例: "mse" -> MethodMSE
func ParseMethod(text string) (Method, error) {
	switch Method(text) {
	case MethodAbsMax, MethodMSE, MethodSearch:
		return Method(text), nil
	default:
		return "", fmt.Errorf("pq4: unknown quantization method %q (want absmax, mse or search)", text)
	}
}

// mseDivisors は MethodMSE で試す「最大絶対値 ÷ 係数」の候補です。
// 正の値は最大絶対値の要素を -側（q = 0 側）に、負の値は +側（q = 15 側）に合わせることを意味します。
// 例: 8 は Q4_0 と同じ割り当て、7 は最大絶対値の要素を +7 / -7 に合わせる割り当て。
var mseDivisors = []float32{
	8.5, 8.25, 8, 7.75, 7.5, 7.25, 7, 6.75, 6.5, 6.25, 6,
	-7.5, -7.25, -7, -6.75, -6.5, -6.25, -6,
}

// QuantizeBlock は 32 要素を 4bit に量子化し、16 バイトの本体と FP16 の係数を返します。
// values は長さ 32、packed は長さ 16 の出力先です。戻り値は係数の FP16 のビット列です。
// MethodSearch は重みをすべて 1 として searchBlock（search.go）で探します。
func QuantizeBlock(values []float32, packed []byte, method Method) uint16 {
	if method == MethodSearch {
		return searchBlock(values, uniformWeights(len(values)), packed)
	}
	signedMax := signedAbsMax(values)
	if signedMax == 0 {
		clear(packed)
		return numeric.Float32ToFloat16(0)
	}
	bestScale := numeric.Float32ToFloat16(signedMax / -quantCenter)
	if method == MethodMSE {
		bestError := float32(math.Inf(1))
		for _, divisor := range mseDivisors {
			candidate := numeric.Float32ToFloat16(signedMax / -divisor)
			if err := blockSquaredError(values, numeric.Float16ToFloat32(candidate)); err < bestError {
				bestError = err
				bestScale = candidate
			}
		}
	}
	encodeBlock(values, numeric.Float16ToFloat32(bestScale), packed)
	return bestScale
}

// signedAbsMax は、絶対値が最大の要素を符号付きで返します。例: [1, -3, 2] -> -3
func signedAbsMax(values []float32) float32 {
	var best float32
	for _, value := range values {
		if math.Abs(float64(value)) > math.Abs(float64(best)) {
			best = value
		}
	}
	return best
}

// quantizeValue は 1 要素を 4bit 値（0〜15）にします。scale が 0 のときは中心の 8 を返します。
// 例: value = -0.8, scale = 0.1 -> round(-8) + 8 = 0
func quantizeValue(value, scale float32) int {
	if scale == 0 {
		return quantCenter
	}
	q := int(math.RoundToEven(float64(value/scale))) + quantCenter
	return min(max(q, 0), quantMax)
}

// blockSquaredError は、係数 scale で量子化・復元したときのブロック内の二乗誤差の合計を返します。
func blockSquaredError(values []float32, scale float32) float32 {
	var sum float32
	for _, value := range values {
		restored := float32(quantizeValue(value, scale)-quantCenter) * scale
		difference := value - restored
		sum += difference * difference
	}
	return sum
}

// encodeBlock は、係数 scale で 32 要素を量子化し、PQ4 のニブル配置で packed（16 バイト）に書きます。
func encodeBlock(values []float32, scale float32, packed []byte) {
	for index := 0; index < halfBlock; index++ {
		low := quantizeValue(values[index], scale)
		high := quantizeValue(values[index+halfBlock], scale)
		packed[index] = byte(low | high<<highNibbleShift)
	}
}

// DecodeBlock は、16 バイトの本体と係数から 32 要素を復元し、dst（長さ 32）に書きます。
func DecodeBlock(packed []byte, scaleBits uint16, dst []float32) {
	scale := numeric.Float16ToFloat32(scaleBits)
	for index := 0; index < halfBlock; index++ {
		dst[index] = float32(int(packed[index]&lowNibbleMask)-quantCenter) * scale
		dst[index+halfBlock] = float32(int(packed[index]>>highNibbleShift)-quantCenter) * scale
	}
}
