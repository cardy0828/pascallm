package pq4

import (
	"fmt"
	"math"

	"github.com/cardy0828/pascallm/internal/linear"
	"github.com/cardy0828/pascallm/internal/numeric"
)

// Matrix は PQ4 形式の行列（出力次元 Rows × 入力次元 Cols）です。linear.Matrix を満たします。
type Matrix struct {
	rows, cols int
	// packed は 4bit 本体 [rows][cols/32][16 バイト]。
	packed []byte
	// scales は係数 [rows][cols/32]（FP16 のビット列）。
	scales []uint16
	// scaleValues は scales を float32 に変換したもの（CPU の MulVec で毎回変換しないため）。
	scaleValues []float32
	// pq8 が true なら PQ8（pq8.go）です。重みのブロックが入力の 1 ブロックにつき 2 個（上位・下位）あります。
	pq8 bool
}

// newMatrix は、形を確かめずに Matrix を組み立て、係数の float32 版を用意します。
func newMatrix(rows, cols int, packed []byte, scales []uint16) *Matrix {
	values := make([]float32, len(scales))
	for index, bits := range scales {
		values[index] = numeric.Float16ToFloat32(bits)
	}
	return &Matrix{rows: rows, cols: cols, packed: packed, scales: scales, scaleValues: values}
}

// BlocksPerRow は 1 行あたりの重みのブロック数を返します。例: cols = 2048 -> PQ4 は 64、PQ8 は 128
func (m *Matrix) BlocksPerRow() int { return m.cols / BlockSize * m.weightBlocksPerInputBlock() }

// Rows は出力次元を返します。
func (m *Matrix) Rows() int { return m.rows }

// Cols は入力次元を返します。
func (m *Matrix) Cols() int { return m.cols }

// Packed は 4bit 本体のバイト列を返します（コピーしない）。
func (m *Matrix) Packed() []byte { return m.packed }

// Scales は係数の配列を返します（コピーしない）。
func (m *Matrix) Scales() []uint16 { return m.scales }

// NewMatrix は、既存の本体と係数（ファイルから読んだものなど）を行列として包みます（コピーしない）。
func NewMatrix(rows, cols int, packed []byte, scales []uint16) (*Matrix, error) {
	if err := checkShape(rows, cols); err != nil {
		return nil, err
	}
	blocks := rows * (cols / BlockSize)
	if len(packed) != blocks*BlockBytes || len(scales) != blocks {
		return nil, fmt.Errorf("pq4: %dx%d matrix needs %d packed bytes and %d scales, got %d and %d",
			rows, cols, blocks*BlockBytes, blocks, len(packed), len(scales))
	}
	return newMatrix(rows, cols, packed, scales), nil
}

// checkShape は、PQ4 で表せる形（入力次元が 32 の倍数）かを確かめます。
func checkShape(rows, cols int) error {
	if rows <= 0 || cols <= 0 || cols%BlockSize != 0 {
		return fmt.Errorf("pq4: cannot store a %dx%d matrix (cols must be a positive multiple of %d)", rows, cols, BlockSize)
	}
	return nil
}

// QuantizeMatrix は、FP32 の行優先の重み（rows × cols）を PQ4 に量子化します（重要度なし、PQ4S の模擬なし）。
// 例: QuantizeMatrix(7680, 2048, w1, MethodMSE)
func QuantizeMatrix(rows, cols int, values []float32, method Method) (*Matrix, error) {
	return QuantizeMatrixWith(rows, cols, values, QuantizeOptions{Method: method})
}

// DequantizeRow は row 行目を FP32 に復元して dst（長さ cols）に書きます。
// PQ8 では、同じ入力のブロックを受け持つ上位・下位のブロックの復元値を足します。
func (m *Matrix) DequantizeRow(row int, dst []float32) {
	blocksPerRow := m.BlocksPerRow()
	perInput := m.weightBlocksPerInputBlock()
	clear(dst[:m.cols])
	part := make([]float32, BlockSize)
	for block := 0; block < blocksPerRow; block++ {
		index := row*blocksPerRow + block
		DecodeBlock(m.packed[index*BlockBytes:(index+1)*BlockBytes], m.scales[index], part)
		target := dst[block/perInput*BlockSize : (block/perInput+1)*BlockSize]
		for element, value := range part {
			target[element] += value
		}
	}
}

// ActivationBlock は、int8 に量子化した活性値 1 ブロック分の付帯情報です（GPU の ActBlockInfo と同じ）。
// Scale: x ≒ Scale × q、ScaledSum: Scale × Σq（重みの中心 8 を引く補正に使う）。
type ActivationBlock struct {
	Scale     float32
	ScaledSum float32
}

// activationLevels は int8 化で使う最大の整数値です（-127〜127 を使う）。
const activationLevels = 127

// QuantizeActivations は、x を 32 要素ごとに int8 化します。GPU の quantize_activation_kernel と同じ手順です。
// quantized は長さ len(x)、blocks は長さ len(x)/32 の出力先です。
// 例: ブロック内の最大絶対値が 2.54 なら Scale = 0.02、要素 1.0 は round(50) = 50
func QuantizeActivations(x []float32, quantized []int8, blocks []ActivationBlock) {
	for block := range blocks {
		source := x[block*BlockSize : (block+1)*BlockSize]
		var maxAbs float32
		for _, value := range source {
			maxAbs = max(maxAbs, float32(math.Abs(float64(value))))
		}
		scale := maxAbs / activationLevels
		var inverse float32
		if scale > 0 {
			inverse = 1 / scale
		}
		sum := 0
		for index, value := range source {
			q := int(math.RoundToEven(float64(value * inverse)))
			quantized[block*BlockSize+index] = int8(q)
			sum += q
		}
		blocks[block] = ActivationBlock{Scale: scale, ScaledSum: scale * float32(sum)}
	}
}

// MulVec は dst = W·x を、GPU の DP4A カーネルと同じ手順で計算します。
//
//  1. x を 32 要素ごとに int8 化する（QuantizeActivations）
//  2. 各ブロックで整数の内積 Σ q_w × q_a を求める（q_w は 0〜15 のまま）
//  3. d_w × (d_a × 内積 − 8 × d_a × Σq_a) を足し合わせる
func (m *Matrix) MulVec(dst, x []float32) {
	linear.CheckShapes(m, dst, x)
	blocksPerRow := m.BlocksPerRow()
	perInput := m.weightBlocksPerInputBlock()
	quantized := make([]int8, m.cols)
	activations := make([]ActivationBlock, m.cols/BlockSize)
	QuantizeActivations(x, quantized, activations)
	scales := m.scaleValues

	linear.ParallelRows(m.rows, func(begin, end int) {
		for row := begin; row < end; row++ {
			var sum float32
			for block := 0; block < blocksPerRow; block++ {
				index := row*blocksPerRow + block
				packed := m.packed[index*BlockBytes : (index+1)*BlockBytes]
				input := block / perInput // PQ8 では上位・下位の 2 ブロックが同じ活性値のブロックを使う
				act := quantized[input*BlockSize : (input+1)*BlockSize]
				dot := int32(0)
				for byteIndex := 0; byteIndex < halfBlock; byteIndex++ {
					value := packed[byteIndex]
					dot += int32(value&lowNibbleMask) * int32(act[byteIndex])
					dot += int32(value>>highNibbleShift) * int32(act[byteIndex+halfBlock])
				}
				info := activations[input]
				sum += scales[index] * (info.Scale*float32(dot) - quantCenter*info.ScaledSum)
			}
			dst[row] = sum
		}
	})
}
