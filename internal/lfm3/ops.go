package lfm3

import (
	"math"

	"github.com/cardy0828/pascallm/internal/linear"
)

// rmsNorm は dst = src / sqrt(mean(src²) + eps) * weight を計算します。dst と src は同じでもかまいません。
// 二乗和は float64 で足し、桁落ちを抑えます。
// 例: src = [3, 4], weight = [1, 1], eps = 0 -> [0.8485, 1.1314]（rms = 3.5355）
func rmsNorm(dst, src, weight []float32, eps float32) {
	var sumSquares float64
	for _, value := range src {
		sumSquares += float64(value) * float64(value)
	}
	mean := float32(sumSquares / float64(len(src)))
	inverseRMS := float32(1 / math.Sqrt(float64(mean+eps)))
	for index, value := range src {
		dst[index] = weight[index] * (value * inverseRMS)
	}
}

// addInPlace は dst[i] += src[i] を計算します（残差接続）。
func addInPlace(dst, src []float32) {
	for index, value := range src {
		dst[index] += value
	}
}

// dotFloat32 は 2 つのベクトルの内積です（attention のスコア計算に使う）。
func dotFloat32(a, b []float32) float32 {
	return linear.Dot(a, b)
}

// softmaxInPlace は values を softmax に置き換えます。最大値を引いてから指数を取り、桁あふれを防ぎます。
// 例: [0, 0] -> [0.5, 0.5]
func softmaxInPlace(values []float32) {
	maximum := float32(math.Inf(-1))
	for _, value := range values {
		maximum = max(maximum, value)
	}
	var sum float64
	for index, value := range values {
		exponent := float32(math.Exp(float64(value - maximum)))
		values[index] = exponent
		sum += float64(exponent)
	}
	inverse := float32(1 / sum)
	for index := range values {
		values[index] *= inverse
	}
}

// sigmoid は 1 / (1 + exp(-x)) です。例: sigmoid(0) = 0.5
func sigmoid(x float32) float32 {
	return float32(1 / (1 + math.Exp(-float64(x))))
}

// silu は x * sigmoid(x) です（SwiGLU の活性化）。例: silu(0) = 0、silu(1) ≒ 0.7311
func silu(x float32) float32 {
	return x * sigmoid(x)
}

// ropeInverseFrequencies は、RoPE の逆周波数（長さ head_dim/2）を返します。
//
// LFM3 の部分 RoPE は、回転する次元を縮めるのではなく、全次元の周波数の並びのまま
// 先頭 partial_rotary_factor × head_dim / 2 個だけを使い、残りを 0（回転しない）にします。
// 例: head_dim = 128、factor = 0.5、theta = 1e6 -> 先頭 32 個が 1/theta^(2i/128)、後ろ 32 個が 0
func ropeInverseFrequencies(rope RopeParameters, headDim int) []float32 {
	half := headDim / 2
	rotated := int(rope.partialRotaryFactor()*float64(headDim)) / 2
	inverse := make([]float32, half)
	for index := 0; index < rotated; index++ {
		// PyTorch と同じく、指数（2i / head_dim）とべき乗を float32 で表した値にそろえる。
		exponent := float32(2*index) / float32(headDim)
		inverse[index] = 1 / float32(math.Pow(rope.Theta, float64(exponent)))
	}
	return inverse
}

// applyRope は、1 ヘッド分のベクトル（長さ head_dim）に RoPE を掛けます（rotate_half 方式）。
// 次元 i と i + head_dim/2 を組にして回転します。cos・sin は長さ head_dim/2 です。
// 逆周波数が 0 の次元は cos = 1、sin = 0 になり、値は変わりません。
func applyRope(vector, cos, sin []float32) {
	half := len(cos)
	for index := 0; index < half; index++ {
		first := vector[index]
		second := vector[index+half]
		vector[index] = first*cos[index] - second*sin[index]
		vector[index+half] = second*cos[index] + first*sin[index]
	}
}
