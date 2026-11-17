package lfm3gpu

import (
	"fmt"

	"github.com/cardy0828/pascallm/internal/cuda"
	"github.com/cardy0828/pascallm/internal/lfm3"
	"github.com/cardy0828/pascallm/internal/linear"
	"github.com/cardy0828/pascallm/internal/numeric"
	"github.com/cardy0828/pascallm/internal/pq4"
)

// deviceMatrix は GPU 上の PQ4・PQ8 行列です（4bit 本体と FP16 係数の分離配置、CPU 版と同じ並び）。
// PQ8 は、入力の 1 ブロックにつき PQ4 のブロック 2 個（上位・下位）を持ちます（internal/pq4/pq8.go）。
type deviceMatrix struct {
	packed cuda.DevicePtr // [rows][重みのブロック][16 バイト]
	scales cuda.DevicePtr // [rows][重みのブロック] の FP16
	rows   int
	cols   int
	// weightBlocks は 1 行あたりの重みのブロック数（PQ4 は cols/32、PQ8 は 2 × cols/32）です。
	weightBlocks int
	// inputBlockShift は、カーネルで活性値のブロック番号を「重みのブロック番号 >> shift」とする値（PQ4 は 0、PQ8 は 1）です。
	inputBlockShift int
}

// blocksPerRow は 1 行あたりの重みのブロック数です。例: cols = 2048 -> PQ4 は 64、PQ8 は 128
func (d deviceMatrix) blocksPerRow() int { return d.weightBlocks }

// format は形式の名前です（計測の表示用）。例: "pq4"
func (d deviceMatrix) format() string {
	if d.inputBlockShift > 0 {
		return "pq8"
	}
	return "pq4"
}

// deviceLayer は GPU 上の 1 層分の重みと状態です。層の種類に合わない項目は 0 のままです。
type deviceLayer struct {
	kind lfm3.LayerKind

	operatorNorm     cuda.DevicePtr // [hidden] float32
	postOperatorNorm cuda.DevicePtr
	ffnNorm          cuda.DevicePtr
	postFFNNorm      cuda.DevicePtr

	// shortconv 層
	inProj     deviceMatrix   // [3 × hidden, hidden]
	outProj    deviceMatrix   // [hidden, hidden]
	convKernel cuda.DevicePtr // [kernel_size][hidden] float32（読み込み時に転置。カーネルが連続して読むため）
	// convHistory は過去 kernel_size-1 個の入力の輪状バッファ [kernel_size-1][hidden] float32（状態）です。
	// 位置 p の入力は行 p mod (kernel_size-1) に入ります（kernels/lfm3.cu の short_conv_quantize）。
	convHistory cuda.DevicePtr

	// attention 層
	qkv         deviceMatrix   // q・k・v を行方向につないだもの [q + 2 × kv, hidden]
	o           deviceMatrix   // [hidden, num_heads × head_dim]
	gate        cuda.DevicePtr // [num_heads][hidden] FP16
	qNorm       cuda.DevicePtr // [head_dim] float32
	kNorm       cuda.DevicePtr
	ropeInverse cuda.DevicePtr // [head_dim/2] float32
	keyCache    cuda.DevicePtr // [max_context][kv_heads × head_dim] FP16
	valueCache  cuda.DevicePtr
	window      int // 窓付き層なら sliding_window、全文脈層なら 0

	// FFN（全層）
	gateUp deviceMatrix // W1 と W3 を行方向につないだもの [2 × intermediate, hidden]
	down   deviceMatrix // W2 [hidden, intermediate]
}

// asPQ4 は、読み込んだ行列が PQ4（または PQ8）であることを確かめて返します。
// GPU 実装は、それ以外の大きな行列（F16 の語彙出力など）に対応しません。
func asPQ4(matrix linear.Matrix, name string) (*pq4.Matrix, error) {
	quantized, ok := matrix.(*pq4.Matrix)
	if !ok {
		return nil, fmt.Errorf("lfm3gpu: %s is %T, but the GPU engine needs PQ4", name, matrix)
	}
	return quantized, nil
}

// uploadMatrices は、入力次元が同じ PQ4 行列を、行方向につないだ 1 つの行列として GPU に置きます。
// 1 回の GEMV で複数の行列を計算するためです（出力は渡した順に並ぶ）。
// 例: uploadMatrices(memory, "ffn", w1, w3) -> 出力の前半が W1·x、後半が W3·x
func uploadMatrices(memory *deviceMemory, name string, parts ...*pq4.Matrix) (deviceMatrix, error) {
	cols := parts[0].Cols()
	shift := parts[0].InputBlockShift()
	var rows, packedBytes, scaleCount int
	for _, part := range parts {
		if part.Cols() != cols {
			return deviceMatrix{}, fmt.Errorf("lfm3gpu: %s: cannot join matrices with %d and %d columns", name, cols, part.Cols())
		}
		if part.InputBlockShift() != shift {
			return deviceMatrix{}, fmt.Errorf("lfm3gpu: %s: cannot join PQ4 and PQ8 matrices (use the same format for all of them)", name)
		}
		rows += part.Rows()
		packedBytes += len(part.Packed())
		scaleCount += len(part.Scales())
	}
	if rows%gemvRowsPerWarp != 0 {
		return deviceMatrix{}, fmt.Errorf("lfm3gpu: %s: rows %d is not a multiple of %d (gemv_pq4)", name, rows, gemvRowsPerWarp)
	}
	packed, err := memory.alloc(packedBytes)
	if err != nil {
		return deviceMatrix{}, err
	}
	scales, err := memory.alloc(2 * scaleCount)
	if err != nil {
		return deviceMatrix{}, err
	}
	packedOffset, scaleOffset := 0, 0
	for _, part := range parts {
		if err := memory.copyBytes(packed+cuda.DevicePtr(packedOffset), part.Packed()); err != nil {
			return deviceMatrix{}, err
		}
		if err := memory.copyBytes(scales+cuda.DevicePtr(scaleOffset), asBytes(part.Scales())); err != nil {
			return deviceMatrix{}, err
		}
		packedOffset += len(part.Packed())
		scaleOffset += 2 * len(part.Scales())
	}
	return deviceMatrix{packed: packed, scales: scales, rows: rows, cols: cols, weightBlocks: parts[0].BlocksPerRow(),
		inputBlockShift: shift}, nil
}

// float16Rows は、FP32 の Dense 行列を FP16 のビット列に戻します。
// attention のゲートは .pq4m に F16 で保存され、読み込み時に FP32 へ広げられているため、戻しても値は変わりません。
func float16Rows(matrix linear.Matrix, name string) ([]uint16, error) {
	dense, ok := matrix.(*linear.Dense)
	if !ok {
		return nil, fmt.Errorf("lfm3gpu: %s is %T, want a dense (F16) matrix", name, matrix)
	}
	bits := make([]uint16, 0, dense.Rows()*dense.Cols())
	for row := 0; row < dense.Rows(); row++ {
		for _, value := range dense.Row(row) {
			bits = append(bits, numeric.Float32ToFloat16(value))
		}
	}
	return bits, nil
}

// layerUploader は 1 層分の転送で、最初に起きたエラーだけを覚えておく道具です（lfm3 の layerLoader と同じ考え方）。
// エラーの後の転送は何もしないので、呼び出し側は最後に err を 1 回確かめれば済みます。
type layerUploader struct {
	memory *deviceMemory
	layer  int
	err    error
}

// vector は float32 の配列を GPU に置きます。例: uploader.vector(weights.FFNNorm)
func (u *layerUploader) vector(values []float32) cuda.DevicePtr {
	if u.err != nil {
		return 0
	}
	pointer, err := upload(u.memory, values)
	u.err = err
	return pointer
}

// matrices は PQ4 行列（複数ならつないだもの）を GPU に置きます。例: uploader.matrices("qkv", q, k, v)
func (u *layerUploader) matrices(name string, parts ...linear.Matrix) deviceMatrix {
	if u.err != nil {
		return deviceMatrix{}
	}
	quantized := make([]*pq4.Matrix, len(parts))
	for index, part := range parts {
		if quantized[index], u.err = asPQ4(part, fmt.Sprintf("layer %d %s", u.layer, name)); u.err != nil {
			return deviceMatrix{}
		}
	}
	matrix, err := uploadMatrices(u.memory, name, quantized...)
	u.err = err
	return matrix
}

// transposeConvKernel は、畳み込みの係数 [hidden][kernelSize] を [kernelSize][hidden] に並べ替えた新しいスライスを返します。
// GPU ではスレッドがチャネルを 1 個ずつ受け持つため、同じ係数番号のチャネルが隣り合う並びにすると、読み出しがまとまります。
// 例: hidden = 2、kernelSize = 3、kernel = [a0 a1 a2 b0 b1 b2] なら [a0 b0 a1 b1 a2 b2]
func transposeConvKernel(kernel []float32, hidden, kernelSize int) []float32 {
	transposed := make([]float32, len(kernel))
	for channel := 0; channel < hidden; channel++ {
		for tap := 0; tap < kernelSize; tap++ {
			transposed[tap*hidden+channel] = kernel[channel*kernelSize+tap]
		}
	}
	return transposed
}

// zeroed は size バイトの GPU メモリを確保し、0 で埋めます（状態・キャッシュ用）。
func (u *layerUploader) zeroed(size int) cuda.DevicePtr {
	if u.err != nil {
		return 0
	}
	pointer, err := u.memory.alloc(size)
	if err == nil {
		err = cuda.MemsetBytes(u.memory.stream, pointer, 0, size)
	}
	u.err = err
	return pointer
}

// uploadLayer は、CPU 側で読み込んだ 1 層分の重みを GPU に置き、状態（畳み込みの履歴・KV キャッシュ）を確保します。
// maxContext は KV キャッシュに入る最大の位置数です。例: 4096
func uploadLayer(memory *deviceMemory, config *lfm3.Config, layer int, weights *lfm3.LayerWeights,
	maxContext int) (deviceLayer, error) {
	u := &layerUploader{memory: memory, layer: layer}
	out := deviceLayer{kind: weights.Kind}
	out.operatorNorm = u.vector(weights.OperatorNorm)
	out.postOperatorNorm = u.vector(weights.PostOperatorNorm)
	out.ffnNorm = u.vector(weights.FFNNorm)
	out.postFFNNorm = u.vector(weights.PostFFNNorm)

	if weights.Kind == lfm3.LayerConv {
		conv := weights.Conv
		out.inProj = u.matrices("conv.in_proj", conv.InProj)
		out.outProj = u.matrices("conv.out_proj", conv.OutProj)
		out.convKernel = u.vector(transposeConvKernel(conv.Kernel, config.HiddenSize, config.ConvKernelSize))
		out.convHistory = u.zeroed(4 * config.HiddenSize * (config.ConvKernelSize - 1))
	} else {
		attention := weights.Attention
		out.qkv = u.matrices("self_attn.qkv", attention.Q, attention.K, attention.V)
		out.o = u.matrices("self_attn.o_proj", attention.O)
		out.qNorm = u.vector(attention.QNorm)
		out.kNorm = u.vector(attention.KNorm)
		if u.err == nil {
			var gateBits []uint16
			if gateBits, u.err = float16Rows(attention.Gate, fmt.Sprintf("layer %d gate_proj", layer)); u.err == nil {
				out.gate, u.err = upload(memory, gateBits)
			}
		}
		if u.err == nil {
			var inverse []float32
			if inverse, u.err = config.RopeInverseFrequencies(weights.Kind); u.err == nil {
				out.ropeInverse = u.vector(inverse)
			}
		}
		cacheBytes := 2 * maxContext * config.NumKVHeads * config.HeadDim
		out.keyCache = u.zeroed(cacheBytes)
		out.valueCache = u.zeroed(cacheBytes)
		if weights.Kind == lfm3.LayerSlidingAttention {
			out.window = config.SlidingWindow
		}
	}
	ffn := weights.FeedForward
	out.gateUp = u.matrices("feed_forward.w1_w3", ffn.W1, ffn.W3)
	out.down = u.matrices("feed_forward.w2", ffn.W2)
	if u.err != nil {
		return deviceLayer{}, fmt.Errorf("lfm3gpu: layer %d: %w", layer, u.err)
	}
	return out, nil
}
