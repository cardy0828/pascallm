package lfm3gpu

import (
	"fmt"

	"github.com/cardy0828/pascallm/internal/cuda"
	"github.com/cardy0828/pascallm/internal/lfm3"
	"github.com/cardy0828/pascallm/internal/pq4"
)

// この file は入力処理（prefill、M4）です。プロンプトの複数のトークンを、区切り（最大 PrefillChunkTokens 個）ごとに
// まとめて処理し、KV キャッシュと shortconv の履歴を、1 トークンずつ Feed した場合と同じ形で書きます。
// 語彙出力（lm_head）は計算しません。プロンプトの最後のトークンは、これまでどおり Feed・FeedCandidates で入れて、
// 次のトークンを選びます。設計は docs/ARCHITECTURE.md の 5.5 節です。

// PrefillChunkTokens は、Prefill が 1 回の区切りでまとめて処理する最大のトークン数です。
const PrefillChunkTokens = 512

// 入力処理のカーネルの寸法です（kernels/lfm3.cu の kGemm*・kPrefill* と合わせる）。
const (
	gemmTileRows             = 64  // gemm_pq4 の 1 ブロックが受け持つ重みの行数
	gemmTileTokens           = 128 // gemm_pq4 の 1 ブロックが受け持つトークン数
	gemmThreads              = 256 // gemm_pq4 のスレッド数
	gemmStageBlocks          = 4   // gemm_pq4 が K 方向に 1 回で読むブロック数（blocks_per_row はこの倍数）
	prefillAttentionTokens   = 8   // attention_prefill の 1 ブロックが受け持つトークン数
	prefillAttentionThreads  = 256 // attention_prefill・attention_prepare_rows のスレッド数
	prefillTokenBytes        = 4   // トークン番号（int32）のバイト数
	prefillFloatBytes        = 4   // float32 のバイト数
	prefillMinimumTokenCount = 1   // Prefill に渡す区切りの最小のトークン数
)

// prefillKernelSet は入力処理のカーネルです。
type prefillKernelSet struct {
	embedRows            *cuda.Function
	residualNormRows     *cuda.Function
	gemm                 *cuda.Function
	shortConvRows        *cuda.Function
	attentionPrepareRows *cuda.Function
	attentionPrefill     *cuda.Function
	quantizeRows         *cuda.Function
	siluMulRows          *cuda.Function
}

// prefillBuffers は、1 区切り（最大 PrefillChunkTokens 個）の作業領域です。行はトークンごとに連続して並びます。
type prefillBuffers struct {
	tokens    cuda.DevicePtr // [chunk] int32
	hidden    cuda.DevicePtr // [chunk][hidden] float32
	normed    cuda.DevicePtr // [chunk][hidden] float32（attention のゲートの入力）
	delta     cuda.DevicePtr // [chunk][hidden] float32
	actQ      cuda.DevicePtr // [chunk][最大の入力次元] int8
	actInfo   cuda.DevicePtr // [chunk][最大の入力次元 / 32] ActBlockInfo
	convIn    cuda.DevicePtr // [chunk][3 × hidden] float32
	qkv       cuda.DevicePtr // [chunk][q + 2 × kv] float32
	query     cuda.DevicePtr // [chunk][num_heads × head_dim] float32（正規化・RoPE の後）
	gate      cuda.DevicePtr // [chunk][num_heads] float32
	attention cuda.DevicePtr // [chunk][num_heads × head_dim] float32（ゲートを掛けた attention の出力）
	gateUp    cuda.DevicePtr // [chunk][2 × intermediate] float32
}

// loadPrefillKernels は、モジュールから入力処理のカーネルを名前で取り出します。
func loadPrefillKernels(module *cuda.Module) (prefillKernelSet, error) {
	var set prefillKernelSet
	targets := []struct {
		target **cuda.Function
		name   string
	}{
		{&set.embedRows, "embed_rows"},
		{&set.residualNormRows, "residual_norm_quantize_rows"},
		{&set.gemm, "gemm_pq4"},
		{&set.shortConvRows, "short_conv_quantize_rows"},
		{&set.attentionPrepareRows, "attention_prepare_rows"},
		{&set.attentionPrefill, "attention_prefill"},
		{&set.quantizeRows, "quantize_rows"},
		{&set.siluMulRows, "silu_mul_quantize_rows"},
	}
	for _, entry := range targets {
		function, err := module.Function(entry.name)
		if err != nil {
			return prefillKernelSet{}, err
		}
		*entry.target = function
	}
	return set, nil
}

// checkGemmShape は、行列が gemm_pq4 の前提（行数が 64 の倍数、1 行のブロック数が 8 の倍数）を満たすかを確かめます。
// 例: 15360 行 × 2048 列（64 ブロック）→ nil
func checkGemmShape(matrix deviceMatrix, name string) error {
	if matrix.rows%gemmTileRows != 0 || matrix.blocksPerRow()%gemmStageBlocks != 0 {
		return fmt.Errorf("lfm3gpu: %s (%d × %d) does not fit gemm_pq4 (rows multiple of %d, blocks per row multiple of %d)",
			name, matrix.rows, matrix.cols, gemmTileRows, gemmStageBlocks)
	}
	return nil
}

// namedMatrix は、エラーの表示用に名前を付けた行列です。
type namedMatrix struct {
	matrix deviceMatrix
	name   string
}

// layerMatrices は、層の種類に応じて、入力処理で gemm_pq4 に渡す行列の一覧を返します。
// 例: shortconv 層なら in_proj・out_proj・w1_w3・w2
func layerMatrices(layer *deviceLayer) []namedMatrix {
	matrices := []namedMatrix{{layer.gateUp, "w1_w3"}, {layer.down, "w2"}}
	if layer.kind == lfm3.LayerConv {
		return append(matrices, namedMatrix{layer.inProj, "in_proj"}, namedMatrix{layer.outProj, "out_proj"})
	}
	return append(matrices, namedMatrix{layer.qkv, "qkv"}, namedMatrix{layer.o, "o_proj"})
}

// preparePrefill は、入力処理のカーネルを読み込み、行列の形を確かめ、作業領域と固定メモリを確保します（Open から呼ぶ）。
func (m *Model) preparePrefill() error {
	kernels, err := loadPrefillKernels(m.module)
	if err != nil {
		return err
	}
	m.prefillKernels = kernels
	for index, layer := range m.layers {
		for _, entry := range layerMatrices(&layer) {
			if err := checkGemmShape(entry.matrix, fmt.Sprintf("layer %d %s", index, entry.name)); err != nil {
				return err
			}
		}
	}

	config := m.config
	hidden := config.HiddenSize
	attentionWidth := config.NumHeads * config.HeadDim
	kvWidth := config.NumKVHeads * config.HeadDim
	maxInput := max(hidden, config.IntermediateSize, attentionWidth)
	rows := PrefillChunkTokens
	sizes := []struct {
		target *cuda.DevicePtr
		bytes  int
	}{
		{&m.prefill.tokens, prefillTokenBytes * rows},
		{&m.prefill.hidden, prefillFloatBytes * rows * hidden},
		{&m.prefill.normed, prefillFloatBytes * rows * hidden},
		{&m.prefill.delta, prefillFloatBytes * rows * hidden},
		{&m.prefill.actQ, rows * maxInput},
		{&m.prefill.actInfo, activationInfoSize * rows * (maxInput / pq4.BlockSize)},
		{&m.prefill.convIn, prefillFloatBytes * rows * 3 * hidden},
		{&m.prefill.qkv, prefillFloatBytes * rows * (attentionWidth + 2*kvWidth)},
		{&m.prefill.query, prefillFloatBytes * rows * attentionWidth},
		{&m.prefill.gate, prefillFloatBytes * rows * config.NumHeads},
		{&m.prefill.attention, prefillFloatBytes * rows * attentionWidth},
		{&m.prefill.gateUp, prefillFloatBytes * rows * 2 * config.IntermediateSize},
	}
	for _, entry := range sizes {
		pointer, err := m.memory.alloc(entry.bytes)
		if err != nil {
			return err
		}
		*entry.target = pointer
	}
	m.hostPrefillTokens, err = cuda.AllocHost(prefillTokenBytes * rows)
	return err
}

// Prefill は tokens を位置 Position() から順にまとめて処理し、位置を len(tokens) 進めます。
// KV キャッシュと shortconv の履歴は、同じトークンを 1 個ずつ Feed した場合と同じ場所に書きます（値は、合計の順序の
// 違いで、わずかに違うことがあります）。logits は計算しないため、次のトークンを選ぶには、続けて最後のトークンを
// Feed・FeedCandidates・Forward で入れてください。ContinueGreedy は Prefill の直後には使えません。
// 例: Prefill(prompt[:len(prompt)-1]) の後に FeedCandidates(prompt[len(prompt)-1], false, params)
func (m *Model) Prefill(tokens []int) error {
	if len(tokens) < prefillMinimumTokenCount {
		return nil
	}
	if m.position+len(tokens) > m.maxContext {
		return fmt.Errorf("lfm3gpu: prefill of %d tokens at position %d exceeds MaxContext %d", len(tokens), m.position, m.maxContext)
	}
	for _, token := range tokens {
		if token < 0 || token >= m.config.VocabSize {
			return fmt.Errorf("lfm3gpu: token %d is outside the vocabulary (size %d)", token, m.config.VocabSize)
		}
	}
	for offset := 0; offset < len(tokens); offset += PrefillChunkTokens {
		end := min(offset+PrefillChunkTokens, len(tokens))
		if err := m.prefillChunk(tokens[offset:end]); err != nil {
			return err
		}
	}
	return nil
}

// prefillChunk は、最大 PrefillChunkTokens 個のトークンを 1 区切りとして処理し、終わるまで待ちます。
// トークン番号の固定メモリを次の区切りで書き換えるため、区切りごとに完了を待ちます。
func (m *Model) prefillChunk(tokens []int) error {
	host := m.hostPrefillTokens.Int32s()
	for index, token := range tokens {
		host[index] = int32(token)
	}
	if err := m.hostPrefillTokens.CopyPrefixToDeviceAsync(m.stream, m.prefill.tokens, prefillTokenBytes*len(tokens)); err != nil {
		return err
	}
	l := &launcher{stream: m.stream}
	m.enqueuePrefill(l, len(tokens))
	if l.err != nil {
		return l.err
	}
	if err := m.stream.Synchronize(); err != nil {
		return err
	}
	m.position += len(tokens)
	return nil
}

// enqueuePrefill は、位置 m.position から count 個のトークン（prefill.tokens に転送済み）の全カーネルを積みます。
// 手順は enqueueTokenWith と同じですが、最後の層の後の正規化と語彙出力は行いません（logits を使わないため）。
func (m *Model) enqueuePrefill(l *launcher, count int) {
	config := m.config
	start := m.position
	p := &m.prefill
	l.launch(m.prefillKernels.embedRows,
		cuda.Dim3{X: uint32(config.HiddenSize / threadsPerBlock), Y: uint32(count), Z: 1}, dim1(threadsPerBlock),
		new(cuda.KernelArgs).Pointer(m.embedding).Pointer(p.tokens).Pointer(p.hidden).Int32(int32(config.HiddenSize)))
	m.residualNormRows(l, count, 0, 0, m.layers[0].operatorNorm)

	for index := range m.layers {
		layer := &m.layers[index]
		if layer.kind == lfm3.LayerConv {
			m.gemmRows(l, layer.inProj, p.convIn, count)
			m.shortConvRows(l, layer, start, count)
			m.gemmRows(l, layer.outProj, p.delta, count)
		} else {
			m.gemmRows(l, layer.qkv, p.qkv, count)
			m.attentionRows(l, layer, start, count)
			m.gemmRows(l, layer.o, p.delta, count)
		}
		m.residualNormRows(l, count, p.delta, layer.postOperatorNorm, layer.ffnNorm)
		m.gemmRows(l, layer.gateUp, p.gateUp, count)
		intermediate := config.IntermediateSize
		l.launch(m.prefillKernels.siluMulRows,
			cuda.Dim3{X: uint32(intermediate / threadsPerBlock), Y: uint32(count), Z: 1}, dim1(threadsPerBlock),
			new(cuda.KernelArgs).Pointer(p.gateUp).Int32(int32(intermediate)).Pointer(p.actQ).Pointer(p.actInfo))
		m.gemmRows(l, layer.down, p.delta, count)
		// 次の層の入口の正規化まで行う。最後の層の後（最終正規化）は、logits を使わないため行わない。
		if index+1 < len(m.layers) {
			m.residualNormRows(l, count, p.delta, layer.postFFNNorm, m.layers[index+1].operatorNorm)
		}
	}
}

// residualNormRows は、トークンの行ごとに残差の加算・正規化・int8 化を積みます（delta が 0 なら加算なし）。
func (m *Model) residualNormRows(l *launcher, count int, delta, deltaWeight, nextWeight cuda.DevicePtr) {
	p := &m.prefill
	args := new(cuda.KernelArgs).
		Pointer(p.hidden).Pointer(delta).Pointer(deltaWeight).Pointer(nextWeight).
		Float32(float32(m.config.NormEps)).Int32(int32(m.config.HiddenSize)).
		Pointer(p.normed).Pointer(p.actQ).Pointer(p.actInfo)
	l.launch(m.prefillKernels.residualNormRows, dim1(count), dim1(normThreads), args)
}

// gemmRows は、out[t] = matrix · (トークン t の int8 化済みの活性値) を count トークン分積みます。
func (m *Model) gemmRows(l *launcher, matrix deviceMatrix, out cuda.DevicePtr, count int) {
	p := &m.prefill
	args := new(cuda.KernelArgs).
		Pointer(matrix.packed).Pointer(matrix.scales).Pointer(p.actQ).Pointer(p.actInfo).
		Pointer(out).Int32(int32(matrix.rows)).Int32(int32(matrix.blocksPerRow())).Int32(int32(count)).
		Int32(int32(matrix.inputBlockShift))
	grid := cuda.Dim3{X: uint32(matrix.rows / gemmTileRows), Y: uint32(ceilDiv(count, gemmTileTokens)), Z: 1}
	l.launchLabeled(fmt.Sprintf("gemm_pq4 %dx%d %s", matrix.rows, matrix.cols, matrix.format()), m.prefillKernels.gemm, grid,
		dim1(gemmThreads), args)
}

// shortConvRows は、shortconv 層の畳み込みを位置 start から count トークン分積みます（入力は prefill.convIn）。
func (m *Model) shortConvRows(l *launcher, layer *deviceLayer, start, count int) {
	config := m.config
	p := &m.prefill
	args := new(cuda.KernelArgs).
		Pointer(p.convIn).Pointer(layer.convHistory).Pointer(layer.convKernel).
		Int32(int32(start)).Int32(int32(count)).Int32(int32(config.HiddenSize)).Int32(int32(config.ConvKernelSize)).
		Pointer(p.actQ).Pointer(p.actInfo)
	l.launch(m.prefillKernels.shortConvRows, dim1(config.HiddenSize/threadsPerBlock), dim1(threadsPerBlock), args)
}

// attentionRows は、attention 層の前処理（正規化・RoPE・KV キャッシュへの書き込み・ゲート）、attention 本体、
// 出力の int8 化を、位置 start から count トークン分積みます（入力は prefill.qkv と prefill.normed）。
func (m *Model) attentionRows(l *launcher, layer *deviceLayer, start, count int) {
	config := m.config
	p := &m.prefill
	prepare := new(cuda.KernelArgs).
		Pointer(p.qkv).Pointer(layer.qNorm).Pointer(layer.kNorm).Pointer(layer.ropeInverse).
		Pointer(layer.gate).Pointer(p.normed).Int32(int32(config.HiddenSize)).Float32(float32(config.NormEps)).
		Pointer(layer.keyCache).Pointer(layer.valueCache).Int32(int32(start)).Int32(int32(config.NumKVHeads)).
		Pointer(p.query).Pointer(p.gate)
	l.launch(m.prefillKernels.attentionPrepareRows, dim1(count), dim1(prefillAttentionThreads), prepare)

	attend := new(cuda.KernelArgs).
		Pointer(p.query).Pointer(layer.keyCache).Pointer(layer.valueCache).Pointer(p.gate).
		Int32(int32(start)).Int32(int32(count)).Int32(int32(layer.window)).Int32(int32(config.NumKVHeads)).
		Float32(attentionScale(config.HeadDim)).Pointer(p.attention)
	grid := cuda.Dim3{X: uint32(config.NumKVHeads), Y: uint32(ceilDiv(count, prefillAttentionTokens)), Z: 1}
	l.launch(m.prefillKernels.attentionPrefill, grid, dim1(prefillAttentionThreads), attend)

	width := config.NumHeads * config.HeadDim
	l.launch(m.prefillKernels.quantizeRows,
		cuda.Dim3{X: uint32(width / threadsPerBlock), Y: uint32(count), Z: 1}, dim1(threadsPerBlock),
		new(cuda.KernelArgs).Pointer(p.attention).Int32(int32(width)).Pointer(p.actQ).Pointer(p.actInfo))
}

// kernelTimer は、カーネルの起動ごとに前後のイベントを置き、ラベルごとの GPU 時間を集めます（計測用）。
type kernelTimer struct {
	labels []string
	events []*cuda.Event // 起動 i の前が events[2i]、後が events[2i+1]
}

// measure は、launch（カーネル 1 個の起動）の前後にイベントを置きます。
func (t *kernelTimer) measure(label string, stream *cuda.Stream, launch func() error) error {
	pair := make([]*cuda.Event, 2)
	for index := range pair {
		event, err := cuda.NewEvent()
		if err != nil {
			return err
		}
		t.events = append(t.events, event)
		pair[index] = event
	}
	t.labels = append(t.labels, label)
	if err := pair[0].Record(stream); err != nil {
		return err
	}
	if err := launch(); err != nil {
		return err
	}
	return pair[1].Record(stream)
}

// totals は、ラベルごとの GPU 時間の合計（マイクロ秒）を返します。ストリームの完了を待った後に呼びます。
func (t *kernelTimer) totals() (map[string]float64, error) {
	const microsecondsPerMillisecond = 1000
	result := make(map[string]float64)
	for index, label := range t.labels {
		milliseconds, err := cuda.ElapsedMilliseconds(t.events[2*index], t.events[2*index+1])
		if err != nil {
			return nil, err
		}
		result[label] += milliseconds * microsecondsPerMillisecond
	}
	return result, nil
}

// close はイベントをすべて破棄します。
func (t *kernelTimer) close() {
	for _, event := range t.events {
		event.Close()
	}
}

// MeasurePrefill は、tokens（PrefillChunkTokens 個以下）を Prefill と同じく 1 区切りで処理し、カーネルの種類ごとの
// GPU 時間（マイクロ秒、ラベル → 時間）を返します。位置は len(tokens) 進みます（確認用。起動ごとにイベントを置くため、
// 通常の Prefill より少し遅くなります）。例: times, err := model.MeasurePrefill(tokens[:512])
func (m *Model) MeasurePrefill(tokens []int) (map[string]float64, error) {
	if len(tokens) < prefillMinimumTokenCount || len(tokens) > PrefillChunkTokens {
		return nil, fmt.Errorf("lfm3gpu: MeasurePrefill needs 1 to %d tokens, got %d", PrefillChunkTokens, len(tokens))
	}
	if m.position+len(tokens) > m.maxContext {
		return nil, fmt.Errorf("lfm3gpu: prefill of %d tokens at position %d exceeds MaxContext %d", len(tokens), m.position, m.maxContext)
	}
	host := m.hostPrefillTokens.Int32s()
	for index, token := range tokens {
		host[index] = int32(token)
	}
	if err := m.hostPrefillTokens.CopyPrefixToDeviceAsync(m.stream, m.prefill.tokens, prefillTokenBytes*len(tokens)); err != nil {
		return nil, err
	}
	timer := &kernelTimer{}
	defer timer.close()
	l := &launcher{stream: m.stream, timer: timer}
	m.enqueuePrefill(l, len(tokens))
	if l.err != nil {
		return nil, l.err
	}
	if err := m.stream.Synchronize(); err != nil {
		return nil, err
	}
	m.position += len(tokens)
	return timer.totals()
}
