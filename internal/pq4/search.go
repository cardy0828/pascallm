package pq4

import (
	"fmt"
	"math"

	"github.com/cardy0828/pascallm/internal/linear"
	"github.com/cardy0828/pascallm/internal/numeric"
)

// この file は M5 で加えた量子化の方法です。どれも保存形式（PQ4）と実行時の計算は変えず、変換だけが変わります。
//
//   - MethodSearch: 係数の候補を細かく試し、各候補で「その割り当てに対して誤差が最小の係数」を
//     閉じた式で求め直して比べます（llama.cpp の make_qx_quants と同じ考え方）。
//   - 重要度（imatrix）: 誤差を入力の列ごとの重み w_j を掛けて測ります。w_j は較正文での入力の 2 乗の平均
//     E[x_j²] で、出力の誤差の期待値 E[(Σ_j Δw_j x_j)²] ≒ Σ_j Δw_j² E[x_j²] を小さくする係数を選ぶことになります。
//   - PQ4S の模擬: 8 ブロック（256 要素）ごとに共通の係数 D を持ち、各ブロックの係数を D × u（u は符号付き 6bit の
//     整数 −31〜31）に制限します。本当の PQ4S（約 4.25 bpw）は D（FP16）と u（6bit）だけを保存しますが、
//     ここでは D × u を FP16 に丸めて PQ4 の係数として保存し、品質だけを先に測ります（FP16 の丸めの差は 0.05% 以下）。

// MethodSearch は、細かい候補と閉じた式による係数の求め直しで、（重み付きの）二乗誤差が最小の係数を選びます。
const MethodSearch Method = "search"

// searchDivisors は MethodSearch で試す「最大絶対値 ÷ 係数」の候補です（正は最大絶対値の要素を −側に、負は +側に合わせる）。
// MethodMSE の候補（0.25 刻み）を 0.125 刻みにし、範囲を少し広げたものです。
var searchDivisors = makeDivisors()

// 候補の範囲と刻みです。
const (
	searchNegativeSideMinimum = 5.5   // −側に合わせる候補の最小（係数が最も大きい）
	searchNegativeSideMaximum = 9.0   // −側に合わせる候補の最大
	searchPositiveSideMinimum = 5.5   // +側に合わせる候補の最小
	searchPositiveSideMaximum = 7.75  // +側に合わせる候補の最大（+7 が 4bit の上端）
	searchDivisorStep         = 0.125 // 候補の刻み
)

// makeDivisors は searchDivisors を作ります。例: [5.5, 5.625, ..., 9, -5.5, ..., -7.75]
func makeDivisors() []float32 {
	var divisors []float32
	for value := searchNegativeSideMinimum; value <= searchNegativeSideMaximum; value += searchDivisorStep {
		divisors = append(divisors, float32(value))
	}
	for value := searchPositiveSideMinimum; value <= searchPositiveSideMaximum; value += searchDivisorStep {
		divisors = append(divisors, float32(-value))
	}
	return divisors
}

// PQ4S の模擬の定数です。
const (
	SuperBlockBlocks = 8  // 共通の係数を持つブロックの数（8 × 32 = 256 要素）
	subScaleLimit    = 31 // 各ブロックの係数の整数 u の絶対値の上限（符号付き 6bit）
)

// QuantizeOptions は、行列の量子化の設定です。
type QuantizeOptions struct {
	Method Method
	// Importance は入力の列ごとの重要度（長さ cols、imatrix の値）です。nil ならすべて 1 として扱います。
	// MethodSearch のときだけ使います。
	Importance []float32
	// EmulatePQ4S が true なら、係数を PQ4S と同じ形（8 ブロックごとの共通の係数 × 6bit の整数）に制限します。
	EmulatePQ4S bool
}

// QuantizeMatrixWith は、FP32 の行優先の重み（rows × cols）を、options に従って PQ4 に量子化します。
// 行ごとに並列で処理します。例: QuantizeMatrixWith(7680, 2048, w1, QuantizeOptions{Method: MethodSearch, Importance: imatrix})
func QuantizeMatrixWith(rows, cols int, values []float32, options QuantizeOptions) (*Matrix, error) {
	if err := checkShape(rows, cols); err != nil {
		return nil, err
	}
	if len(values) != rows*cols {
		return nil, fmt.Errorf("pq4: %dx%d matrix needs %d values, got %d", rows, cols, rows*cols, len(values))
	}
	if options.Importance != nil && len(options.Importance) != cols {
		return nil, fmt.Errorf("pq4: importance has %d values for %d columns", len(options.Importance), cols)
	}
	blocksPerRow := cols / BlockSize
	if options.EmulatePQ4S && blocksPerRow%SuperBlockBlocks != 0 {
		return nil, fmt.Errorf("pq4: PQ4S needs a multiple of %d blocks per row, got %d", SuperBlockBlocks, blocksPerRow)
	}
	packed := make([]byte, rows*blocksPerRow*BlockBytes)
	scales := make([]uint16, rows*blocksPerRow)
	weights := options.Importance
	if weights == nil {
		weights = uniformWeights(cols)
	}
	linear.ParallelRows(rows, func(begin, end int) {
		for row := begin; row < end; row++ {
			quantizeRow(values[row*cols:(row+1)*cols], weights, options,
				packed[row*blocksPerRow*BlockBytes:(row+1)*blocksPerRow*BlockBytes],
				scales[row*blocksPerRow:(row+1)*blocksPerRow])
		}
	})
	return newMatrix(rows, cols, packed, scales), nil
}

// uniformWeights は、長さ n のすべて 1 の重みを返します。
func uniformWeights(n int) []float32 {
	weights := make([]float32, n)
	for index := range weights {
		weights[index] = 1
	}
	return weights
}

// quantizeRow は 1 行分（values・weights は長さ cols）を量子化し、packed と scales に書きます。
func quantizeRow(values, weights []float32, options QuantizeOptions, packed []byte, scales []uint16) {
	for block := range scales {
		source := values[block*BlockSize : (block+1)*BlockSize]
		target := packed[block*BlockBytes : (block+1)*BlockBytes]
		if options.Method == MethodSearch {
			scales[block] = searchBlock(source, weights[block*BlockSize:(block+1)*BlockSize], target)
		} else {
			scales[block] = QuantizeBlock(source, target, options.Method)
		}
	}
	if options.EmulatePQ4S {
		for first := 0; first < len(scales); first += SuperBlockBlocks {
			restrictToSuperBlock(values[first*BlockSize:(first+SuperBlockBlocks)*BlockSize],
				packed[first*BlockBytes:(first+SuperBlockBlocks)*BlockBytes], scales[first:first+SuperBlockBlocks])
		}
	}
}

// weightedError は、係数 scale で量子化・復元したときの、重み付きの二乗誤差 Σ w (x − x̂)² を返します。
func weightedError(values, weights []float32, scale float32) float64 {
	var sum float64
	for index, value := range values {
		restored := float32(quantizeValue(value, scale)-quantCenter) * scale
		difference := float64(value - restored)
		sum += float64(weights[index]) * difference * difference
	}
	return sum
}

// refinedScale は、係数 scale で決まる整数の割り当て q（中心を引いた −8〜7）を固定したとき、
// 重み付きの二乗誤差が最小になる係数 Σ w x q / Σ w q² を返します。分母が 0 なら scale を返します。
// 例: values = [1, 2]、weights = [1, 1]、scale = 1 → q = [1, 2] → (1 + 4) / (1 + 4) = 1
func refinedScale(values, weights []float32, scale float32) float32 {
	var numerator, denominator float64
	for index, value := range values {
		q := float64(quantizeValue(value, scale) - quantCenter)
		numerator += float64(weights[index]) * float64(value) * q
		denominator += float64(weights[index]) * q * q
	}
	if denominator == 0 {
		return scale
	}
	return float32(numerator / denominator)
}

// searchBlock は MethodSearch で 1 ブロックを量子化し、packed（16 バイト）に書いて係数の FP16 を返します。
// 各候補で、候補の係数と、その割り当てから求め直した係数の両方を FP16 に丸めて試し、誤差が最小のものを選びます。
// 重みがすべて 0 のブロック（較正文で一度も使われない列）は、重み 1 として扱います。
// 例: 正規分布の 32 要素では、MethodMSE より二乗誤差が小さいか同じ係数になります。
func searchBlock(values, weights []float32, packed []byte) uint16 {
	signedMax := signedAbsMax(values)
	if signedMax == 0 {
		clear(packed)
		return numeric.Float32ToFloat16(0)
	}
	if allZero(weights) {
		weights = uniformWeights(len(values))
	}
	bestScale := numeric.Float32ToFloat16(signedMax / -quantCenter)
	bestError := weightedError(values, weights, numeric.Float16ToFloat32(bestScale))
	for _, divisor := range searchDivisors {
		candidate := signedMax / -divisor
		for _, scale := range []float32{candidate, refinedScale(values, weights, candidate)} {
			bits := numeric.Float32ToFloat16(scale)
			if err := weightedError(values, weights, numeric.Float16ToFloat32(bits)); err < bestError {
				bestError = err
				bestScale = bits
			}
		}
	}
	encodeBlock(values, numeric.Float16ToFloat32(bestScale), packed)
	return bestScale
}

// allZero は、すべての要素が 0 かを返します。
func allZero(values []float32) bool {
	for _, value := range values {
		if value != 0 {
			return false
		}
	}
	return true
}

// restrictToSuperBlock は、8 ブロック分の係数を PQ4S の形（共通の係数 D × 整数 u、|u| ≤ 31）に丸め、
// 丸めた係数で 8 ブロックを量子化し直します。values は 256 要素、packed は 8 × 16 バイト、scales は 8 個です。
// D は FP16 で、最大の |係数| がちょうど 31 × D になるように決めます。
// 例: 係数が [0.31, -0.1, ...] なら D = 0.01、u = [31, -10, ...]
func restrictToSuperBlock(values []float32, packed []byte, scales []uint16) {
	var largest float32
	for _, bits := range scales {
		largest = max(largest, float32(math.Abs(float64(numeric.Float16ToFloat32(bits)))))
	}
	if largest == 0 {
		return
	}
	superScale := numeric.Float16ToFloat32(numeric.Float32ToFloat16(largest / subScaleLimit))
	for block, bits := range scales {
		u := math.Round(float64(numeric.Float16ToFloat32(bits) / superScale))
		u = min(max(u, -subScaleLimit), subScaleLimit)
		scale := numeric.Float32ToFloat16(float32(u) * superScale)
		scales[block] = scale
		encodeBlock(values[block*BlockSize:(block+1)*BlockSize], numeric.Float16ToFloat32(scale),
			packed[block*BlockBytes:(block+1)*BlockBytes])
	}
}
