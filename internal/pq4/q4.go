package pq4

import (
	"encoding/binary"
	"fmt"

	"github.com/cardy0828/pascallm/internal/linear"
)

// この file は、llama.cpp の Q4_0 と PQ4 の橋渡しです（M7）。
//
// Q4_0 と PQ4 は同じ量子化です。どちらも 32 要素ごとに FP16 の係数 d を 1 個持ち、4bit の値 q（0〜15）を
// (q − 8) × d で復元します。バイト j の下位 4bit が要素 j、上位 4bit が要素 j + 16 という並びも同じです。
// 違いは配置だけで、Q4_0 は [係数 2 バイト][本体 16 バイト] を交互に並べ、PQ4 は本体と係数を別の配列にします。
// そのため、並べ替えるだけで値を変えずに変換できます。

// Q4_0 の 1 ブロックの寸法です（gguf パッケージの定数と同じ値。pq4 は gguf に依存しないため、ここにも置く）。
const (
	q4ScaleBytes = 2                         // ブロックの先頭にある FP16 の係数
	q4BlockBytes = q4ScaleBytes + BlockBytes // 18 バイト
)

// FromQ4_0 は、Q4_0 のバイト列（行優先で [行][ブロック][係数 2 + 本体 16]）を PQ4 の行列にします。
// 値は変わりません（本体と係数を別の配列に写すだけ）。
// 例: FromQ4_0(4096, 3840, data) は、4096 × 120 ブロック × 18 バイト = 8,847,360 バイトの data を受け取る
func FromQ4_0(rows, cols int, data []byte) (*Matrix, error) {
	if err := checkShape(rows, cols); err != nil {
		return nil, err
	}
	blocks := rows * (cols / BlockSize)
	if len(data) != blocks*q4BlockBytes {
		return nil, fmt.Errorf("pq4: Q4_0 data for a %dx%d matrix needs %d bytes, got %d", rows, cols, blocks*q4BlockBytes, len(data))
	}
	packed := make([]byte, blocks*BlockBytes)
	scales := make([]uint16, blocks)
	linear.ParallelRows(blocks, func(begin, end int) {
		for block := begin; block < end; block++ {
			source := data[block*q4BlockBytes : (block+1)*q4BlockBytes]
			scales[block] = binary.LittleEndian.Uint16(source[:q4ScaleBytes])
			copy(packed[block*BlockBytes:(block+1)*BlockBytes], source[q4ScaleBytes:])
		}
	})
	return newMatrix(rows, cols, packed, scales), nil
}

// MulVecExact は dst = W·x を、活性値を int8 にせずに計算します（重みは復元値 (q − 8) × d をそのまま使う）。
//
// llama.cpp が Q4_0 の重みで行う計算と同じ意味になるため、参照実装を llama.cpp と突き合わせるときに使います。
// MulVec（GPU と同じ int8 の活性値）との差が、活性値の int8 化による誤差です。
//
//	Σ (q − 8)·d·x = d · (Σ q·x − 8·Σ x)
//
// PQ8 では、上位・下位のブロックが同じ入力のブロックを使います（MulVec と同じ）。
func (m *Matrix) MulVecExact(dst, x []float32) {
	linear.CheckShapes(m, dst, x)
	blocksPerRow := m.BlocksPerRow()
	perInput := m.weightBlocksPerInputBlock()
	// 入力のブロックごとの合計（中心 8 を引く補正に使う）は、全行で共通なので先に求めます。
	inputSums := make([]float32, m.cols/BlockSize)
	for block := range inputSums {
		var sum float32
		for _, value := range x[block*BlockSize : (block+1)*BlockSize] {
			sum += value
		}
		inputSums[block] = sum
	}
	scales := m.scaleValues

	linear.ParallelRows(m.rows, func(begin, end int) {
		for row := begin; row < end; row++ {
			var sum float32
			for block := 0; block < blocksPerRow; block++ {
				index := row*blocksPerRow + block
				packed := m.packed[index*BlockBytes : (index+1)*BlockBytes]
				input := block / perInput
				values := x[input*BlockSize : (input+1)*BlockSize]
				var dot float32
				for byteIndex := 0; byteIndex < halfBlock; byteIndex++ {
					value := packed[byteIndex]
					dot += float32(value&lowNibbleMask) * values[byteIndex]
					dot += float32(value>>highNibbleShift) * values[byteIndex+halfBlock]
				}
				sum += scales[index] * (dot - quantCenter*inputSums[input])
			}
			dst[row] = sum
		}
	})
}
