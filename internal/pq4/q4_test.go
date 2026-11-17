package pq4

import (
	"encoding/binary"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/cardy0828/pascallm/internal/numeric"
)

// randomQ4Data は、rows × cols の行列の Q4_0 のバイト列（[係数 2 バイト][本体 16 バイト] の繰り返し）を乱数で作ります。
func randomQ4Data(random *rand.Rand, rows, cols int) []byte {
	blocks := rows * (cols / BlockSize)
	data := make([]byte, blocks*q4BlockBytes)
	for block := 0; block < blocks; block++ {
		part := data[block*q4BlockBytes : (block+1)*q4BlockBytes]
		binary.LittleEndian.PutUint16(part, numeric.Float32ToFloat16(float32(random.Float64()*0.1+0.01)))
		for index := q4ScaleBytes; index < q4BlockBytes; index++ {
			part[index] = byte(random.IntN(256))
		}
	}
	return data
}

// TestFromQ4Layout は、Q4_0 から詰め替えた行列の復元値が、Q4_0 の定義（(q − 8) × 係数、下位 4bit が要素 j、
// 上位 4bit が要素 j + 16）と同じことを確かめます。
func TestFromQ4Layout(t *testing.T) {
	const rows, cols = 3, 64
	random := rand.New(rand.NewPCG(1, 2))
	data := randomQ4Data(random, rows, cols)
	matrix, err := FromQ4_0(rows, cols, data)
	if err != nil {
		t.Fatal(err)
	}
	restored := make([]float32, cols)
	for row := 0; row < rows; row++ {
		matrix.DequantizeRow(row, restored)
		for col := 0; col < cols; col++ {
			block := data[(row*(cols/BlockSize)+col/BlockSize)*q4BlockBytes:]
			scale := numeric.Float16ToFloat32(binary.LittleEndian.Uint16(block))
			element := col % BlockSize
			packed := block[q4ScaleBytes+element%halfBlock]
			nibble := int(packed & lowNibbleMask)
			if element >= halfBlock {
				nibble = int(packed >> highNibbleShift)
			}
			if want := float32(nibble-quantCenter) * scale; restored[col] != want {
				t.Fatalf("row %d col %d = %g, want %g", row, col, restored[col], want)
			}
		}
	}
	if _, err := FromQ4_0(rows, cols, data[:len(data)-1]); err == nil {
		t.Error("FromQ4_0 accepted data of the wrong length")
	}
}

// TestMulVecExactMatchesDense は、MulVecExact が、復元した重みと FP32 の入力の内積と（丸めの範囲で）同じことを確かめます。
func TestMulVecExactMatchesDense(t *testing.T) {
	const rows, cols = 8, 96
	random := rand.New(rand.NewPCG(3, 4))
	matrix, err := FromQ4_0(rows, cols, randomQ4Data(random, rows, cols))
	if err != nil {
		t.Fatal(err)
	}
	x := make([]float32, cols)
	for index := range x {
		x[index] = float32(random.NormFloat64())
	}
	got := make([]float32, rows)
	matrix.MulVecExact(got, x)
	restored := make([]float32, cols)
	const relativeTolerance = 1e-4
	for row := 0; row < rows; row++ {
		matrix.DequantizeRow(row, restored)
		var want float64
		for col, weight := range restored {
			want += float64(weight) * float64(x[col])
		}
		if math.Abs(float64(got[row])-want) > relativeTolerance*(1+math.Abs(want)) {
			t.Errorf("row %d = %g, want %g", row, got[row], want)
		}
	}
}
