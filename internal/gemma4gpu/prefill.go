package gemma4gpu

import (
	"fmt"

	"github.com/cardy0828/pascallm/internal/cuda"
	"github.com/cardy0828/pascallm/internal/gpucore"
	"github.com/cardy0828/pascallm/internal/pq4"
)

// この file は入力処理（prefill）です。プロンプトの複数のトークンを、区切り（最大 PrefillChunkTokens 個）ごとに
// まとめて処理し、KV キャッシュを、1 トークンずつ Feed した場合と同じ場所に書きます。
// 語彙出力は計算しません。プロンプトの最後のトークンは Feed・FeedCandidates で入れて、次のトークンを選びます。
//
// attention は、トークンごとに 1 ブロックが見える範囲の K・V を読みます（生成用と同じ計算）。
// 複数のトークンで K・V の読み出しを共有する形にはしていません（文脈が長いと attention の割合が増える）。

// PrefillChunkTokens は、Prefill が 1 回の区切りでまとめて処理する最大のトークン数です。
// 局所の層の輪状バッファの余裕（localRingSlack）以下であること。
const PrefillChunkTokens = 512

// prefillKernelSet は入力処理のカーネルです。
const smallGemmLimit = 32
const smallGemmTileTokens = 4
const mtpGemmTileTokens = 8

// prefillKernelSet は入力処理で使うカーネルを保持します。
type prefillKernelSet struct {
	gemmSmall        *cuda.Function
	gemmSmall8       *cuda.Function
	embedRows        *cuda.Function
	residualNormRows *cuda.Function
	gemm             *cuda.Function
	prepareRows      *cuda.Function
	localRows        *cuda.Function
	globalRows       *cuda.Function
	quantizeRows     *cuda.Function
	geluMulRows      *cuda.Function
}

// prefillBuffers は、1 区切り（最大 PrefillChunkTokens 個）の作業領域です。行はトークンごとに連続して並びます。
type prefillBuffers struct {
	tokens    cuda.DevicePtr // [chunk] int32
	hidden    cuda.DevicePtr // [chunk][hidden] float32
	delta     cuda.DevicePtr // [chunk][hidden] float32
	actQ      cuda.DevicePtr // [chunk][最大の入力次元] int8
	actInfo   cuda.DevicePtr // [chunk][最大の入力次元 / 32] ActBlockInfo
	qkv       cuda.DevicePtr // [chunk][最大の qkv の長さ] float32
	query     cuda.DevicePtr // [chunk][最大の q の長さ] float32（正規化・RoPE の後）
	attention cuda.DevicePtr // [chunk][最大の q の長さ] float32（attention の出力）
	gateUp    cuda.DevicePtr // [chunk][2 × ffn] float32
}

// preparePrefill は、入力処理のカーネルを読み込み、作業領域と固定メモリを確保します（Open から呼ぶ）。
func (m *Model) preparePrefill() error {
	if PrefillChunkTokens > localRingSlack {
		return fmt.Errorf("gemma4gpu: PrefillChunkTokens %d is larger than the ring slack %d", PrefillChunkTokens, localRingSlack)
	}
	set := &m.prefillKernels
	err := loadNamed(m.module, []namedKernel{
		{&set.embedRows, "g4_embed_rows"},
		{&set.residualNormRows, "g4_residual_norm_quantize_rows"},
		{&set.gemm, "gemm_pq4"},
		{&set.gemmSmall, "g4_gemm_small"},
		{&set.gemmSmall8, "g4_gemm_small8"},
		{&set.prepareRows, "g4_attention_prepare_rows"},
		{&set.localRows, "g4_attention_local_rows"},
		{&set.globalRows, "g4_attention_global_rows"},
		{&set.quantizeRows, "quantize_rows"},
		{&set.geluMulRows, "g4_gelu_mul_quantize_rows"},
	})
	if err != nil {
		return err
	}
	config := m.config
	hidden := config.HiddenSize
	maxQKV, maxQuery := 0, 0
	for index := range m.layers {
		maxQKV = max(maxQKV, m.layers[index].qkvWidth)
		maxQuery = max(maxQuery, m.layers[index].queryWidth(config.NumHeads))
	}
	maxInput := max(hidden, config.FeedForwardSize, maxQuery)
	rows := PrefillChunkTokens
	p := &m.prefill
	sizes := []struct {
		target *cuda.DevicePtr
		bytes  int
	}{
		{&p.tokens, int32Bytes * rows},
		{&p.hidden, float32Bytes * rows * hidden},
		{&p.delta, float32Bytes * rows * hidden},
		{&p.actQ, rows * maxInput},
		{&p.actInfo, gpucore.ActivationInfoBytes * rows * (maxInput / pq4.BlockSize)},
		{&p.qkv, float32Bytes * rows * maxQKV},
		{&p.query, float32Bytes * rows * maxQuery},
		{&p.attention, float32Bytes * rows * maxQuery},
		{&p.gateUp, float32Bytes * rows * 2 * config.FeedForwardSize},
	}
	for _, entry := range sizes {
		pointer, err := m.memory.Alloc(entry.bytes)
		if err != nil {
			return err
		}
		*entry.target = pointer
	}
	m.hostPrefillTokens, err = cuda.AllocHost(int32Bytes * rows)
	return err
}

// Prefill は tokens を位置 Position() から順にまとめて処理し、位置を len(tokens) 進めます。
// logits は計算しないため、次のトークンを選ぶには、続けて最後のトークンを Feed・FeedCandidates・Forward で入れてください。
// ContinueGreedy は Prefill の直後には使えません。
// 例: Prefill(prompt[:len(prompt)-1]) の後に FeedCandidates(prompt[len(prompt)-1], false, params)
func (m *Model) Prefill(tokens []int) error {
	if len(tokens) > 0 {
		m.invalidateMTP()
	}
	if len(tokens) == 0 {
		return nil
	}
	if m.position+len(tokens) > m.maxContext {
		return fmt.Errorf("gemma4gpu: prefill of %d tokens at position %d exceeds MaxContext %d", len(tokens), m.position, m.maxContext)
	}
	for _, token := range tokens {
		if token < 0 || token >= m.config.VocabSize {
			return fmt.Errorf("gemma4gpu: token %d is outside the vocabulary (size %d)", token, m.config.VocabSize)
		}
	}
	for offset := 0; offset < len(tokens); offset += PrefillChunkTokens {
		end := min(offset+PrefillChunkTokens, len(tokens))
		if _, _, err := m.prefillChunk(tokens[offset:end], nil); err != nil {
			return err
		}
	}
	return nil
}

// MeasurePrefill は、tokens（PrefillChunkTokens 個以下）を Prefill と同じく 1 区切りで処理し、カーネルの種類ごとの
// GPU 時間（マイクロ秒）と起動回数を返します。位置は len(tokens) 進みます（確認用）。
func (m *Model) MeasurePrefill(tokens []int) (map[string]float64, map[string]int, error) {
	if len(tokens) == 0 || len(tokens) > PrefillChunkTokens {
		return nil, nil, fmt.Errorf("gemma4gpu: MeasurePrefill needs 1 to %d tokens, got %d", PrefillChunkTokens, len(tokens))
	}
	if m.position+len(tokens) > m.maxContext {
		return nil, nil, fmt.Errorf("gemma4gpu: prefill of %d tokens at position %d exceeds MaxContext %d", len(tokens), m.position, m.maxContext)
	}
	timer := &gpucore.KernelTimer{}
	defer timer.Close()
	return m.prefillChunk(tokens, timer)
}

// prefillChunk は、最大 PrefillChunkTokens 個のトークンを 1 区切りとして処理し、終わるまで待ちます。
// timer が nil でなければ、カーネルごとの時間を測って返します。
func (m *Model) prefillChunk(tokens []int, timer *gpucore.KernelTimer) (map[string]float64, map[string]int, error) {
	host := m.hostPrefillTokens.Int32s()
	for index, token := range tokens {
		host[index] = int32(token)
	}
	if err := m.hostPrefillTokens.CopyPrefixToDeviceAsync(m.stream, m.prefill.tokens, int32Bytes*len(tokens)); err != nil {
		return nil, nil, err
	}
	l := &gpucore.Launcher{Stream: m.stream, Timer: timer}
	m.enqueuePrefill(l, len(tokens), len(m.layers))
	if l.Err != nil {
		return nil, nil, l.Err
	}
	// トークン番号の固定メモリを次の区切りで書き換えるため、区切りごとに完了を待つ。
	if err := m.stream.Synchronize(); err != nil {
		return nil, nil, err
	}
	m.position += len(tokens)
	if timer == nil {
		return nil, nil, nil
	}
	return timer.Totals()
}

// enqueuePrefill は、位置 m.position から count 個のトークン（prefill.tokens に転送済み）について、埋め込みと
// 最初の layerCount 層のカーネルを積みます（通常は全層）。手順は enqueueToken と同じですが、最後の層の後の
// 正規化と語彙出力は行いません（logits を使わないため）。
func (m *Model) enqueuePrefill(l *gpucore.Launcher, count, layerCount int) {
	m.enqueueRows(l, count, layerCount, false)
}

// enqueueRows は入力処理または MTP 検証の本体を積みます。verify のとき attention も単一生成と同じ加算順です。
func (m *Model) enqueueRows(l *gpucore.Launcher, count, layerCount int, verify bool) {
	config := m.config
	hidden := config.HiddenSize
	start := m.position
	p := &m.prefill
	l.Launch(m.prefillKernels.embedRows, gpucore.Dim2(hidden/threadsPerBlock, count), gpucore.Dim1(threadsPerBlock),
		new(cuda.KernelArgs).Pointer(m.embedding.Packed).Pointer(m.embedding.Scales).Pointer(p.tokens).Pointer(p.hidden).
			Int32(int32(hidden)).Float32(config.EmbeddingScale()))
	m.residualNormRows(l, count, 0, 0, m.layers[0].attnNorm, noScale)

	for index := 0; index < layerCount; index++ {
		layer := &m.layers[index]
		m.gemmRows(l, layer.qkv, p.qkv, count)
		if verify {
			m.attendVerify(l, layer, start, count)
		} else {
			m.attendRows(l, layer, start, count)
		}
		m.gemmRows(l, layer.o, p.delta, count)
		m.residualNormRows(l, count, p.delta, layer.postAttnNorm, layer.ffnNorm, noScale)

		m.gemmRows(l, layer.gateUp, p.gateUp, count)
		size := config.FeedForwardSize
		l.Launch(m.prefillKernels.geluMulRows, gpucore.Dim2(size/threadsPerBlock, count), gpucore.Dim1(threadsPerBlock),
			new(cuda.KernelArgs).Pointer(p.gateUp).Int32(int32(size)).Pointer(p.actQ).Pointer(p.actInfo))
		m.gemmRows(l, layer.down, p.delta, count)
		// 次の層の入口の正規化まで行う。最後の層の後（最終正規化）は、logits を使わないため行わない。
		if index+1 < len(m.layers) {
			m.residualNormRows(l, count, p.delta, layer.postFFNNorm, m.layers[index+1].attnNorm, layer.outScale)
		}
	}
}

// residualNormRows は、トークンの行ごとに残差の加算・係数・正規化・int8 化を積みます（delta が 0 なら加算なし）。
func (m *Model) residualNormRows(l *gpucore.Launcher, count int, delta, deltaWeight, nextWeight cuda.DevicePtr, hiddenScale float32) {
	p := &m.prefill
	args := new(cuda.KernelArgs).
		Pointer(p.hidden).Pointer(delta).Pointer(deltaWeight).Pointer(nextWeight).
		Float32(float32(m.config.NormEps)).Int32(int32(m.config.HiddenSize)).Float32(hiddenScale).
		Pointer(p.actQ).Pointer(p.actInfo)
	l.Launch(m.prefillKernels.residualNormRows, gpucore.Dim1(count), gpucore.Dim1(normThreads), args)
}

// gemmRows は、out[t] = matrix · (トークン t の int8 化済みの活性値) を count トークン分積みます。
func (m *Model) gemmRows(l *gpucore.Launcher, matrix gpucore.Matrix, out cuda.DevicePtr, count int) {
	if count > smallGemmLimit {
		gpucore.Gemm(l, m.prefillKernels.gemm, matrix, m.prefill.actQ, m.prefill.actInfo, out, count)
		return
	}
	args := new(cuda.KernelArgs).Pointer(matrix.Packed).Pointer(matrix.Scales).
		Pointer(m.prefill.actQ).Pointer(m.prefill.actInfo).Pointer(out).
		Int32(int32(matrix.Rows)).Int32(int32(matrix.WeightBlocks)).Int32(int32(count)).Int32(int32(matrix.InputBlockShift))
	kernel, tile := m.prefillKernels.gemmSmall, smallGemmTileTokens
	if count > smallGemmTileTokens && count <= mtpGemmTileTokens {
		kernel = m.prefillKernels.gemmSmall8
		tile = mtpGemmTileTokens
	}
	l.LaunchLabeled(matrix.Label("g4_gemm_small"), kernel,
		gpucore.Dim2(gpucore.CeilDiv(matrix.Rows, 16), gpucore.CeilDiv(count, tile)),
		cuda.Dim3{X: 32, Y: 4, Z: 1}, args)
}

// attendRows は、attention の前処理・本体・出力の int8 化を、位置 start から count トークン分積みます
// （入力は prefill.qkv）。前処理が区切りの全トークンの k・v を KV キャッシュに書いてから、本体が読みます。
func (m *Model) attendRows(l *gpucore.Launcher, layer *deviceLayer, start, count int) {
	config := m.config
	p := &m.prefill
	headDim := layer.config.HeadDim
	kvHeads := layer.config.KVHeads
	prepare := new(cuda.KernelArgs).
		Pointer(p.qkv).Pointer(layer.qNorm).Pointer(layer.kNorm).Pointer(layer.ropeInverse).
		Float32(float32(config.NormEps)).Int32(int32(start)).
		Int32(int32(headDim)).Int32(int32(config.NumHeads)).Int32(int32(kvHeads)).Int32(int32(layer.valueOffset)).
		Int32(int32(layer.qkvWidth)).
		Pointer(p.query).Pointer(layer.keyCache).Pointer(layer.valueCache).Int32(int32(layer.ringSize))
	l.Launch(m.prefillKernels.prepareRows, gpucore.Dim2(m.prepareVectorBlocks(layer), count), gpucore.Dim1(threadsPerBlock), prepare)

	body := new(cuda.KernelArgs).
		Pointer(p.query).Pointer(layer.keyCache).Pointer(layer.valueCache).
		Int32(int32(start)).Int32(int32(layer.window)).Int32(int32(layer.ringSize)).Int32(int32(kvHeads)).
		Int32(int32(config.NumHeads / kvHeads)).Int32(int32(config.NumHeads)).Float32(attentionScale).Pointer(p.attention)
	kernel := m.prefillKernels.globalRows
	if layer.config.Sliding {
		kernel = m.prefillKernels.localRows
	}
	l.Launch(kernel, gpucore.Dim2(config.NumHeads/layer.groupSize, count), gpucore.Dim1(threadsPerBlock), body)

	width := layer.queryWidth(config.NumHeads)
	l.Launch(m.prefillKernels.quantizeRows, gpucore.Dim2(width/threadsPerBlock, count), gpucore.Dim1(threadsPerBlock),
		new(cuda.KernelArgs).Pointer(p.attention).Int32(int32(width)).Pointer(p.actQ).Pointer(p.actInfo))
}

// DebugPrefillHidden は、tokens（PrefillChunkTokens 個以下）を現在の位置から 1 区切りとして、最初の layers 層だけ
// 処理し、その時点の残差の流れ（[トークン][hidden] を並べたもの）を返します。位置は進めません。
// 生成の経路（DebugHidden）と層ごとに比べるための確認用です。layers は全層より少なくしてください
// （最後の層の後の加算は、入力処理では行わないため）。
// 例: DebugPrefillHidden(tokens[:40], 1) は、40 トークン分の 1 層目の後の hidden
func (m *Model) DebugPrefillHidden(tokens []int, layers int) ([]float32, error) {
	if len(tokens) == 0 || len(tokens) > PrefillChunkTokens {
		return nil, fmt.Errorf("gemma4gpu: DebugPrefillHidden needs 1 to %d tokens, got %d", PrefillChunkTokens, len(tokens))
	}
	if layers < 0 || layers >= len(m.layers) {
		return nil, fmt.Errorf("gemma4gpu: DebugPrefillHidden needs 0 to %d layers, got %d", len(m.layers)-1, layers)
	}
	if m.position+len(tokens) > m.maxContext {
		return nil, fmt.Errorf("gemma4gpu: prefill of %d tokens at position %d exceeds MaxContext %d", len(tokens), m.position, m.maxContext)
	}
	host := m.hostPrefillTokens.Int32s()
	for index, token := range tokens {
		host[index] = int32(token)
	}
	if err := m.hostPrefillTokens.CopyPrefixToDeviceAsync(m.stream, m.prefill.tokens, int32Bytes*len(tokens)); err != nil {
		return nil, err
	}
	l := &gpucore.Launcher{Stream: m.stream}
	m.enqueuePrefill(l, len(tokens), layers)
	if l.Err != nil {
		return nil, l.Err
	}
	hidden := make([]float32, len(tokens)*m.config.HiddenSize)
	if err := cuda.CopyFromDevice(m.stream, hidden, m.prefill.hidden); err != nil {
		return nil, err
	}
	return hidden, nil
}
