package pq4

import (
	"math"
	"math/rand/v2"
	"path/filepath"
	"testing"

	"github.com/cardy0828/pascallm/internal/numeric"
)

// randomValues は、正規分布に近い乱数を n 個返します（seed 固定で毎回同じ値）。
func randomValues(n int, seed uint64) []float32 {
	random := rand.New(rand.NewPCG(seed, 7))
	values := make([]float32, n)
	for index := range values {
		values[index] = float32(random.NormFloat64())
	}
	return values
}

// TestBlockLayout は、ニブルの並び（バイト j の下位が要素 j、上位が要素 j+16）と復元値を確かめます。
func TestBlockLayout(t *testing.T) {
	values := make([]float32, BlockSize)
	values[0] = -8 // 最大絶対値。absmax では q = 0 になる
	values[16] = 7 // q = 15
	values[1] = 1  // q = 9
	packed := make([]byte, BlockBytes)
	scaleBits := QuantizeBlock(values, packed, MethodAbsMax)
	if scale := numeric.Float16ToFloat32(scaleBits); scale != 1 {
		t.Fatalf("scale = %g, want 1", scale)
	}
	if packed[0] != 0x0|0xF<<4 {
		t.Errorf("packed[0] = 0x%02X, want 0xF0", packed[0])
	}
	if packed[1] != 0x9|0x8<<4 {
		t.Errorf("packed[1] = 0x%02X, want 0x89", packed[1])
	}
	restored := make([]float32, BlockSize)
	DecodeBlock(packed, scaleBits, restored)
	for index := range values {
		if restored[index] != values[index] {
			t.Errorf("restored[%d] = %g, want %g", index, restored[index], values[index])
		}
	}
}

// TestMSEIsNotWorseThanAbsMax は、mse 方式のブロック誤差が absmax 方式以下になることを確かめます
// （mse の候補には absmax と同じ係数が含まれるため、悪くなることはない）。
func TestMSEIsNotWorseThanAbsMax(t *testing.T) {
	values := randomValues(BlockSize*256, 1)
	packed := make([]byte, BlockBytes)
	restored := make([]float32, BlockSize)
	blockError := func(block []float32, method Method) float64 {
		bits := QuantizeBlock(block, packed, method)
		DecodeBlock(packed, bits, restored)
		var sum float64
		for index, value := range block {
			difference := float64(value - restored[index])
			sum += difference * difference
		}
		return sum
	}
	for begin := 0; begin < len(values); begin += BlockSize {
		block := values[begin : begin+BlockSize]
		if mse, absmax := blockError(block, MethodMSE), blockError(block, MethodAbsMax); mse > absmax*(1+1e-6) {
			t.Fatalf("block %d: mse error %g > absmax error %g", begin/BlockSize, mse, absmax)
		}
	}
}

// TestMulVecMatchesDequantized は、整数の内積で計算する MulVec が、
// 「復元した重み × int8 化した活性値を復元したもの」を FP64 で計算した値と一致することを確かめます。
func TestMulVecMatchesDequantized(t *testing.T) {
	const rows, cols = 37, 256
	weights := randomValues(rows*cols, 2)
	x := randomValues(cols, 3)
	matrix, err := QuantizeMatrix(rows, cols, weights, MethodMSE)
	if err != nil {
		t.Fatal(err)
	}
	got := make([]float32, rows)
	matrix.MulVec(got, x)

	quantized := make([]int8, cols)
	blocks := make([]ActivationBlock, cols/BlockSize)
	QuantizeActivations(x, quantized, blocks)
	row := make([]float32, cols)
	for r := 0; r < rows; r++ {
		matrix.DequantizeRow(r, row)
		var want float64
		for c := 0; c < cols; c++ {
			activation := float64(blocks[c/BlockSize].Scale) * float64(quantized[c])
			want += float64(row[c]) * activation
		}
		if math.Abs(float64(got[r])-want) > 1e-4*math.Max(1, math.Abs(want)) {
			t.Fatalf("row %d: MulVec = %g, dequantized = %g", r, got[r], want)
		}
	}
}

// TestFileRoundTrip は、.pq4m に書いたテンソルが同じ値で読み戻せること、CRC32C の検査が働くことを確かめます。
func TestFileRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.pq4m")
	matrix, err := QuantizeMatrix(3, 64, randomValues(3*64, 4), MethodAbsMax)
	if err != nil {
		t.Fatal(err)
	}
	vector := []float32{1, 2, 3}
	half := []uint16{numeric.Float32ToFloat16(0.5), numeric.Float32ToFloat16(-1)}
	header := FileHeader{ModelConfig: []byte(`{"hidden_size":64}`), Source: map[string]string{"test": "yes"}}
	tensors := []TensorData{
		{Name: "m", Kind: KindPQ4, Shape: []int{3, 64}, PQ4: matrix},
		{Name: "v", Kind: KindF32, Shape: []int{3}, F32: vector},
		{Name: "h", Kind: KindF16, Shape: []int{2}, F16: half},
	}
	if err := WriteFile(path, header, tensors); err != nil {
		t.Fatal(err)
	}
	file, err := OpenFile(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()

	readMatrix, err := file.Matrix("m", 3, 64)
	if err != nil {
		t.Fatal(err)
	}
	for index, value := range matrix.Packed() {
		if readMatrix.Packed()[index] != value {
			t.Fatalf("packed[%d] differs", index)
		}
	}
	for index, value := range matrix.Scales() {
		if readMatrix.Scales()[index] != value {
			t.Fatalf("scales[%d] differs", index)
		}
	}
	readVector, _, err := file.Float32("v")
	if err != nil || len(readVector) != 3 || readVector[2] != 3 {
		t.Fatalf("Float32(v) = %v, %v", readVector, err)
	}
	readHalf, _, err := file.Float16("h")
	if err != nil || len(readHalf) != 2 || readHalf[1] != half[1] {
		t.Fatalf("Float16(h) = %v, %v", readHalf, err)
	}
	if _, err := file.Matrix("m", 3, 32); err == nil {
		t.Fatal("Matrix with a wrong shape should fail")
	}
	if string(file.Header.ModelConfig) != `{"hidden_size":64}` {
		t.Fatalf("model config = %s", file.Header.ModelConfig)
	}
}

// TestSearchIsNotWorseThanMSE は、MethodSearch の二乗誤差が MethodMSE 以下であることを確かめます
// （search の候補は mse の候補を含むため）。
func TestSearchIsNotWorseThanMSE(t *testing.T) {
	values := randomValues(BlockSize*256, 3)
	packed := make([]byte, BlockBytes)
	restored := make([]float32, BlockSize)
	blockError := func(block []float32, method Method) float64 {
		bits := QuantizeBlock(block, packed, method)
		DecodeBlock(packed, bits, restored)
		var sum float64
		for index, value := range block {
			difference := float64(value - restored[index])
			sum += difference * difference
		}
		return sum
	}
	var searchTotal, mseTotal float64
	for begin := 0; begin < len(values); begin += BlockSize {
		block := values[begin : begin+BlockSize]
		search, mse := blockError(block, MethodSearch), blockError(block, MethodMSE)
		if search > mse*(1+1e-6) {
			t.Fatalf("block %d: search error %g > mse error %g", begin/BlockSize, search, mse)
		}
		searchTotal += search
		mseTotal += mse
	}
	if searchTotal >= mseTotal {
		t.Fatalf("search total error %g is not smaller than mse %g", searchTotal, mseTotal)
	}
}

// TestImportanceLowersWeightedError は、重要度を使うと、重み付きの誤差が重要度なしより小さいか同じになることを確かめます。
func TestImportanceLowersWeightedError(t *testing.T) {
	const rows, cols = 16, 256
	values := randomValues(rows*cols, 5)
	importance := make([]float32, cols)
	for index := range importance {
		importance[index] = float32(1 + (index%7)*(index%7)) // 列ごとに 1〜37 の重み
	}
	weighted := func(matrix *Matrix) float64 {
		row := make([]float32, cols)
		var sum float64
		for r := 0; r < rows; r++ {
			matrix.DequantizeRow(r, row)
			for c, restored := range row {
				difference := float64(values[r*cols+c] - restored)
				sum += float64(importance[c]) * difference * difference
			}
		}
		return sum
	}
	plain, err := QuantizeMatrixWith(rows, cols, values, QuantizeOptions{Method: MethodSearch})
	if err != nil {
		t.Fatal(err)
	}
	withImportance, err := QuantizeMatrixWith(rows, cols, values, QuantizeOptions{Method: MethodSearch, Importance: importance})
	if err != nil {
		t.Fatal(err)
	}
	if a, b := weighted(withImportance), weighted(plain); a > b*(1+1e-6) {
		t.Fatalf("weighted error with importance %g > without %g", a, b)
	}
}

// TestEmulatePQ4SScales は、PQ4S の模擬で、8 ブロックごとの係数が共通の係数 D の整数倍（|u| ≤ 31）になることを確かめます。
func TestEmulatePQ4SScales(t *testing.T) {
	const rows, cols = 4, 512
	values := randomValues(rows*cols, 9)
	matrix, err := QuantizeMatrixWith(rows, cols, values, QuantizeOptions{Method: MethodMSE, EmulatePQ4S: true})
	if err != nil {
		t.Fatal(err)
	}
	scales := matrix.Scales()
	for first := 0; first < len(scales); first += SuperBlockBlocks {
		var largest float64
		for _, bits := range scales[first : first+SuperBlockBlocks] {
			largest = math.Max(largest, math.Abs(float64(numeric.Float16ToFloat32(bits))))
		}
		superScale := largest / subScaleLimit
		for _, bits := range scales[first : first+SuperBlockBlocks] {
			u := float64(numeric.Float16ToFloat32(bits)) / superScale
			if math.Abs(u-math.Round(u)) > 0.05 || math.Abs(u) > subScaleLimit+0.05 {
				t.Fatalf("super block %d: scale %g is not an integer multiple of %g (u = %g)",
					first/SuperBlockBlocks, numeric.Float16ToFloat32(bits), superScale, u)
			}
		}
	}
}

// TestPQ8RoundTrip は、PQ8 の復元値の誤差が係数の半分以下で、CPU の MulVec が復元した重みでの計算と一致することを確かめます。
func TestPQ8RoundTrip(t *testing.T) {
	const rows, cols = 8, 128
	values := randomValues(rows*cols, 11)
	matrix, err := QuantizePQ8Matrix(rows, cols, values)
	if err != nil {
		t.Fatal(err)
	}
	if matrix.BlocksPerRow() != cols/BlockSize*pq8BlocksPerInput || matrix.InputBlockShift() != 1 {
		t.Fatalf("blocks per row %d, shift %d", matrix.BlocksPerRow(), matrix.InputBlockShift())
	}
	row := make([]float32, cols)
	for r := 0; r < rows; r++ {
		matrix.DequantizeRow(r, row)
		for c, restored := range row {
			scale := numeric.Float16ToFloat32(matrix.Scales()[(r*matrix.BlocksPerRow()+c/BlockSize*pq8BlocksPerInput)+1])
			if math.Abs(float64(values[r*cols+c]-restored)) > float64(scale)/2*1.001 {
				t.Fatalf("row %d col %d: %g restored as %g (scale %g)", r, c, values[r*cols+c], restored, scale)
			}
		}
	}
	x := randomValues(cols, 12)
	got := make([]float32, rows)
	matrix.MulVec(got, x)
	quantized := make([]int8, cols)
	blocks := make([]ActivationBlock, cols/BlockSize)
	QuantizeActivations(x, quantized, blocks)
	for r := 0; r < rows; r++ {
		matrix.DequantizeRow(r, row)
		var want float64
		for c := range row {
			want += float64(row[c]) * float64(quantized[c]) * float64(blocks[c/BlockSize].Scale)
		}
		if math.Abs(float64(got[r])-want) > 1e-3*math.Max(1, math.Abs(want)) {
			t.Fatalf("row %d: MulVec %g, dequantized %g", r, got[r], want)
		}
	}
}
