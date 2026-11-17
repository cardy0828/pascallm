package lfm3

import (
	"github.com/cardy0828/pascallm/internal/linear"
	"github.com/cardy0828/pascallm/internal/pq4"
)

// DequantizeMatrices は、weights の中の PQ4 の行列をすべて FP32 に復元した Dense に置き換えます。
//
// 誤差の切り分けに使います。置き換えた後は活性値の int8 化をせずに計算するため、
// 「重みの 4bit 化だけ」による誤差を測れます（置き換え前との差が、活性値の int8 化による誤差）。
// メモリは FP32 の重み全体（LFM3 1.7B で約 7 GB）を使います。
func DequantizeMatrices(weights *Weights) {
	replace := func(matrix *linear.Matrix) {
		quantized, ok := (*matrix).(*pq4.Matrix)
		if !ok {
			return
		}
		values := make([]float32, quantized.Rows()*quantized.Cols())
		linear.ParallelRows(quantized.Rows(), func(begin, end int) {
			for row := begin; row < end; row++ {
				quantized.DequantizeRow(row, values[row*quantized.Cols():(row+1)*quantized.Cols()])
			}
		})
		dense, err := linear.NewDense(quantized.Rows(), quantized.Cols(), values)
		if err != nil {
			panic(err) // 形は元の行列と同じなので、ここに来るのは実装の誤り
		}
		*matrix = dense
	}

	replace(&weights.LMHead)
	for index := range weights.Layers {
		layer := &weights.Layers[index]
		if layer.Attention != nil {
			for _, matrix := range []*linear.Matrix{&layer.Attention.Q, &layer.Attention.K, &layer.Attention.V,
				&layer.Attention.O, &layer.Attention.Gate} {
				replace(matrix)
			}
		}
		if layer.Conv != nil {
			replace(&layer.Conv.InProj)
			replace(&layer.Conv.OutProj)
		}
		replace(&layer.FeedForward.W1)
		replace(&layer.FeedForward.W2)
		replace(&layer.FeedForward.W3)
	}
}
