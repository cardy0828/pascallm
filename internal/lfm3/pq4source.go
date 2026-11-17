package lfm3

import (
	"fmt"

	"github.com/cardy0828/pascallm/internal/linear"
	"github.com/cardy0828/pascallm/internal/numeric"
	"github.com/cardy0828/pascallm/internal/pq4"
)

// LMHeadTensorName は、PQ4 ファイルで語彙出力の行列を保存する名前です。
// LFM3 は埋め込み表と共有しているため Hugging Face 形式には存在せず、変換時に埋め込み表から作ります。
const LMHeadTensorName = "lm_head.weight"

// PQ4Source は、.pq4m ファイルから重みを読む TensorSource です。
// PQ4 の行列は CPU 上でも GPU と同じ手順（int8 活性値と整数の内積）で計算されます。
type PQ4Source struct {
	File *pq4.File
}

// OpenPQ4Model は .pq4m ファイルを開き、ヘッダーに保存された config を読みます。
func OpenPQ4Model(path string) (*Config, *PQ4Source, error) {
	file, err := pq4.OpenFile(path)
	if err != nil {
		return nil, nil, err
	}
	config, err := ParseConfig(file.Header.ModelConfig)
	if err != nil {
		file.Close()
		return nil, nil, err
	}
	return config, &PQ4Source{File: file}, nil
}

// Vector は F32 で保存したテンソルを返します。形状が一致するかも確かめます。
func (s *PQ4Source) Vector(name string, shape []int) ([]float32, error) {
	values, stored, err := s.File.Float32(name)
	if err != nil {
		return nil, err
	}
	if !equalShape(stored, shape) {
		return nil, fmt.Errorf("lfm3: tensor %s has shape %v, want %v", name, stored, shape)
	}
	return values, nil
}

// Matrix は、PQ4 のテンソルはそのまま、F16 のテンソル（小さい gate_proj など）は FP32 に変換して返します。
func (s *PQ4Source) Matrix(name string, rows, cols int) (linear.Matrix, error) {
	record, err := s.File.Record(name)
	if err != nil {
		return nil, err
	}
	switch record.Kind {
	case pq4.KindPQ4, pq4.KindPQ8:
		return s.File.Matrix(name, rows, cols)
	case pq4.KindF16:
		bits, shape, err := s.File.Float16(name)
		if err != nil {
			return nil, err
		}
		if !equalShape(shape, []int{rows, cols}) {
			return nil, fmt.Errorf("lfm3: tensor %s has shape %v, want [%d %d]", name, shape, rows, cols)
		}
		return linear.NewDense(rows, cols, float16ToFloat32(bits))
	default:
		return nil, fmt.Errorf("lfm3: tensor %s has kind %s, which cannot be used as a matrix", name, record.Kind)
	}
}

// Embedding は F16 で保存した埋め込み表を返します。行は参照のたびに FP32 へ変換します。
func (s *PQ4Source) Embedding(name string, vocab, hidden int) (EmbeddingTable, error) {
	bits, shape, err := s.File.Float16(name)
	if err != nil {
		return nil, err
	}
	if !equalShape(shape, []int{vocab, hidden}) {
		return nil, fmt.Errorf("lfm3: embedding %s has shape %v, want [%d %d]", name, shape, vocab, hidden)
	}
	return Float16Embedding{bits: bits, hidden: hidden}, nil
}

// LMHead は、変換時に作った語彙出力の行列（LMHeadTensorName）を返します。
func (s *PQ4Source) LMHead(_ string, vocab, hidden int) (linear.Matrix, error) {
	return s.Matrix(LMHeadTensorName, vocab, hidden)
}

// Float16Embedding は FP16 の埋め込み表です。
type Float16Embedding struct {
	bits   []uint16
	hidden int
}

// Lookup は token 番目の行を FP32 に変換して dst に書きます。
func (e Float16Embedding) Lookup(token int, dst []float32) {
	row := e.bits[token*e.hidden : (token+1)*e.hidden]
	for index, bits := range row {
		dst[index] = numeric.Float16ToFloat32(bits)
	}
}

// float16ToFloat32 は FP16 のビット列の配列を FP32 の新しい配列に変換します。
func float16ToFloat32(bits []uint16) []float32 {
	values := make([]float32, len(bits))
	for index, value := range bits {
		values[index] = numeric.Float16ToFloat32(value)
	}
	return values
}

// equalShape は 2 つの形状が同じかを返します。
func equalShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for index := range a {
		if a[index] != b[index] {
			return false
		}
	}
	return true
}
