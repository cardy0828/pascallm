package pq4

import (
	"fmt"
	"math"

	"github.com/cardy0828/pascallm/internal/linear"
	"github.com/cardy0828/pascallm/internal/numeric"
)

// この file は PQ8（混合精度で、品質に効く行列に使う約 8bit の形式）です（M5、docs/ARCHITECTURE.md「4.3 混合精度」）。
//
// PQ8 は、入力方向の 32 要素ごとに整数 m（−136〜119）と FP16 の係数 d を持ち、復元値は m × d です。
// m を m + 136 = 16h + l（h・l は 0〜15）と分け、PQ4 のブロック 2 個として保存します。
//   - 上位のブロック: 4bit 値 h、係数 16d（復元値 16d × (h − 8)）
//   - 下位のブロック: 4bit 値 l、係数 d（復元値 d × (l − 8)）
//   - 2 個の和は d × (16h + l − 136) = d × m
// 1 行の中では、入力のブロック b の上位を重みのブロック 2b、下位を 2b + 1 に置きます。
// このため GPU では PQ4 と同じカーネルで計算でき、活性値のブロック番号だけを「重みのブロック番号 ÷ 2」にします
// （カーネルの引数 input_block_shift = 1）。1 要素あたり 9 bit（4bit × 2 + FP16 × 2 / 32）です。

// KindPQ8 は PQ8 のテンソルの保存形式です。部分は PQ4 と同じ "packed" と "scales" で、ブロックの数が 2 倍です。
const KindPQ8 TensorKind = "pq8"

// PQ8 の値の範囲と分け方です。
const (
	pq8Minimum        = -136 // m の最小（16 × (0 − 8) + (0 − 8)）
	pq8Maximum        = 119  // m の最大（16 × (15 − 8) + (15 − 8)）
	pq8Offset         = 136  // m + pq8Offset = 16h + l（0〜255）
	pq8NibbleLevels   = 16   // 4bit の値の数。上位の係数は下位の 16 倍
	pq8BlocksPerInput = 2    // 入力の 1 ブロックあたりの重みのブロック数（上位・下位）
)

// newPQ8Matrix は、形を確かめずに PQ8 の Matrix を組み立てます。
func newPQ8Matrix(rows, cols int, packed []byte, scales []uint16) *Matrix {
	matrix := newMatrix(rows, cols, packed, scales)
	matrix.pq8 = true
	return matrix
}

// NewPQ8Matrix は、既存の本体と係数（ファイルから読んだものなど）を PQ8 の行列として包みます（コピーしない）。
func NewPQ8Matrix(rows, cols int, packed []byte, scales []uint16) (*Matrix, error) {
	if err := checkShape(rows, cols); err != nil {
		return nil, err
	}
	blocks := rows * (cols / BlockSize) * pq8BlocksPerInput
	if len(packed) != blocks*BlockBytes || len(scales) != blocks {
		return nil, fmt.Errorf("pq4: %dx%d PQ8 matrix needs %d packed bytes and %d scales, got %d and %d",
			rows, cols, blocks*BlockBytes, blocks, len(packed), len(scales))
	}
	return newPQ8Matrix(rows, cols, packed, scales), nil
}

// QuantizePQ8Matrix は、FP32 の行優先の重み（rows × cols）を PQ8 に量子化します。行ごとに並列で処理します。
// 例: QuantizePQ8Matrix(2048, 7680, w2)
func QuantizePQ8Matrix(rows, cols int, values []float32) (*Matrix, error) {
	if err := checkShape(rows, cols); err != nil {
		return nil, err
	}
	if len(values) != rows*cols {
		return nil, fmt.Errorf("pq4: %dx%d matrix needs %d values, got %d", rows, cols, rows*cols, len(values))
	}
	inputBlocks := cols / BlockSize
	weightBlocks := inputBlocks * pq8BlocksPerInput
	packed := make([]byte, rows*weightBlocks*BlockBytes)
	scales := make([]uint16, rows*weightBlocks)
	linear.ParallelRows(rows, func(begin, end int) {
		for row := begin; row < end; row++ {
			for block := 0; block < inputBlocks; block++ {
				high := row*weightBlocks + block*pq8BlocksPerInput
				low := high + 1
				source := values[row*cols+block*BlockSize : row*cols+(block+1)*BlockSize]
				scales[high], scales[low] = quantizePQ8Block(source,
					packed[high*BlockBytes:(high+1)*BlockBytes], packed[low*BlockBytes:(low+1)*BlockBytes])
			}
		}
	})
	return newPQ8Matrix(rows, cols, packed, scales), nil
}

// quantizePQ8Block は 32 要素を PQ8 にし、上位・下位のブロック（各 16 バイト）に書いて、両方の係数の FP16 を返します。
// 係数 d は、正の最大が 119 × d、負の最大が −136 × d を超えないように決めます。
// 例: 最大 1.19、最小 −0.5 なら d = 0.01、1.19 は m = 119（h = 15、l = 15）
func quantizePQ8Block(values []float32, highPacked, lowPacked []byte) (uint16, uint16) {
	var largestPositive, largestNegative float32
	for _, value := range values {
		largestPositive = max(largestPositive, value)
		largestNegative = max(largestNegative, -value)
	}
	scale := numeric.Float16ToFloat32(numeric.Float32ToFloat16(max(largestPositive/pq8Maximum, largestNegative/-pq8Minimum)))
	if scale == 0 {
		clear(highPacked)
		clear(lowPacked)
		return numeric.Float32ToFloat16(0), numeric.Float32ToFloat16(0)
	}
	high := make([]int, BlockSize)
	low := make([]int, BlockSize)
	for index, value := range values {
		m := int(math.RoundToEven(float64(value / scale)))
		m = min(max(m, pq8Minimum), pq8Maximum)
		shifted := m + pq8Offset
		high[index] = shifted / pq8NibbleLevels
		low[index] = shifted % pq8NibbleLevels
	}
	for index := 0; index < halfBlock; index++ {
		highPacked[index] = byte(high[index] | high[index+halfBlock]<<highNibbleShift)
		lowPacked[index] = byte(low[index] | low[index+halfBlock]<<highNibbleShift)
	}
	// 上位の係数 16d は d の 2 のべき乗倍なので、FP16 でも丸めの差は出ない。
	return numeric.Float32ToFloat16(scale * pq8NibbleLevels), numeric.Float32ToFloat16(scale)
}

// IsPQ8 は、この行列が PQ8 かを返します。
func (m *Matrix) IsPQ8() bool { return m.pq8 }

// weightBlocksPerInputBlock は、入力の 1 ブロックあたりの重みのブロック数です（PQ4 は 1、PQ8 は 2）。
func (m *Matrix) weightBlocksPerInputBlock() int {
	if m.pq8 {
		return pq8BlocksPerInput
	}
	return 1
}

// InputBlockShift は、GPU のカーネルで「活性値のブロック番号 = 重みのブロック番号 >> shift」とするための値です。
// PQ4 は 0、PQ8 は 1 です。
func (m *Matrix) InputBlockShift() int {
	if m.pq8 {
		return 1
	}
	return 0
}
