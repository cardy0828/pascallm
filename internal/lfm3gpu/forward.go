package lfm3gpu

import (
	"github.com/cardy0828/pascallm/internal/cuda"
	"github.com/cardy0828/pascallm/internal/lfm3"
)

// launcher は、カーネルを順に積む間の最初のエラーだけを覚えておく道具です。
// エラーの後の起動は何もしないので、呼び出し側は最後に err を 1 回確かめれば済みます。
//
// 計測用に、次の 2 つの使い方もできます（MeasureKernelShares）。
//   - skip に含まれるラベルの起動を積まない（そのカーネルを抜いた Graph の時間から、そのカーネルの実際の負担を求める）
//   - counts が nil でなければ、何も積まずにラベルごとの起動回数だけを数える
type launcher struct {
	stream *cuda.Stream
	skip   map[string]bool
	counts map[string]int
	timer  *kernelTimer // nil でなければ、起動ごとに前後のイベントを置いて GPU 時間を測る（MeasurePrefill）
	err    error
}

// launch は function を grid × block で積みます。計測用のラベルはカーネル名です。
// 例: l.launch(k.embedLookup, dim1(8), dim1(256), args)
func (l *launcher) launch(function *cuda.Function, grid, block cuda.Dim3, args *cuda.KernelArgs) {
	l.launchLabeled(function.Name(), function, grid, block, args)
}

// launchLabeled は launch と同じですが、計測用のラベルを label にします（GEMV を形ごとに分けるため）。
func (l *launcher) launchLabeled(label string, function *cuda.Function, grid, block cuda.Dim3, args *cuda.KernelArgs) {
	if l.err != nil {
		return
	}
	if l.counts != nil {
		l.counts[label]++
		return
	}
	if l.skip[label] {
		return
	}
	if l.timer != nil {
		l.err = l.timer.measure(label, l.stream, func() error { return function.Launch(grid, block, 0, l.stream, args) })
		return
	}
	l.err = function.Launch(grid, block, 0, l.stream, args)
}

// dim1 は 1 次元の大きさ（x だけ）を返します。例: dim1(256) = {256, 1, 1}
func dim1(x int) cuda.Dim3 {
	return cuda.Dim3{X: uint32(x), Y: 1, Z: 1}
}

// ceilDiv は value / divisor の切り上げです。例: ceilDiv(10, 4) = 3
func ceilDiv(value, divisor int) int {
	return (value + divisor - 1) / divisor
}

// gemv は out = matrix · (int8 化済みの活性値) を積みます。活性値は buffers.actQ・actInfo から読みます。
func (m *Model) gemv(l *launcher, matrix deviceMatrix, out cuda.DevicePtr) {
	args := new(cuda.KernelArgs).
		Pointer(matrix.packed).Pointer(matrix.scales).
		Pointer(m.buffers.actQ).Pointer(m.buffers.actInfo).
		Pointer(out).Int32(int32(matrix.rows)).Int32(int32(matrix.blocksPerRow())).Int32(int32(matrix.inputBlockShift))
	grid := dim1(ceilDiv(matrix.rows, gemvWarpsPerBlock*gemvRowsPerWarp))
	block := cuda.Dim3{X: warpSize, Y: gemvWarpsPerBlock, Z: 1}
	l.launchLabeled(gemvLabel(matrix), m.kernels.gemvPQ4, grid, block, args)
}

// residualNorm は、delta があれば hidden += RMSNorm(delta) × deltaWeight を行い、
// normed = RMSNorm(hidden) × nextWeight を書いて int8 化する処理を積みます。
// delta が 0 なら加算をせず、正規化と int8 化だけを行います（最初の層の入口）。
func (m *Model) residualNorm(l *launcher, delta, deltaWeight, nextWeight cuda.DevicePtr) {
	args := new(cuda.KernelArgs).
		Pointer(m.buffers.hidden).Pointer(delta).Pointer(deltaWeight).Pointer(nextWeight).
		Float32(float32(m.config.NormEps)).Int32(int32(m.config.HiddenSize)).
		Pointer(m.buffers.normed).Pointer(m.buffers.actQ).Pointer(m.buffers.actInfo)
	l.launch(m.kernels.residualNormQuantize, dim1(1), dim1(normThreads), args)
}

// convMix は、shortconv 層の畳み込み・乗算ゲート・int8 化（1 カーネル）を積みます。入力は buffers.convIn です。
// 履歴の輪状バッファのどの行を使うかは、カーネルが状態の位置から決めます。
func (m *Model) convMix(l *launcher, layer *deviceLayer) {
	config := m.config
	args := new(cuda.KernelArgs).
		Pointer(m.buffers.convIn).Pointer(layer.convHistory).Pointer(layer.convKernel).Pointer(m.buffers.state).
		Int32(int32(config.HiddenSize)).Int32(int32(config.ConvKernelSize)).
		Pointer(m.buffers.actQ).Pointer(m.buffers.actInfo)
	l.launch(m.kernels.shortConvQuantize, dim1(config.HiddenSize/threadsPerBlock), dim1(threadsPerBlock), args)
}

// shortConv は shortconv 層の演算部分（in_proj → 畳み込み → out_proj）を積み、結果を buffers.delta に書きます。
func (m *Model) shortConv(l *launcher, layer *deviceLayer) {
	m.gemv(l, layer.inProj, m.buffers.convIn)
	m.convMix(l, layer)
	m.gemv(l, layer.outProj, m.buffers.delta)
}

// attend は、q・k の正規化と RoPE、KV キャッシュへの書き込み、ヘッドごとのゲート、attention 本体、
// ゲートの乗算、int8 化（1 カーネル）を積みます。入力は buffers.qkv（q・k・v の GEMV の出力）と
// buffers.normed（ゲートの入力）です。
// スレッドブロックは KV ヘッド × 位置の範囲の分割（最大 attentionMaxSplits）です。例: KV 4 ヘッドなら grid = (4, 16)
func (m *Model) attend(l *launcher, layer *deviceLayer) {
	config := m.config
	args := new(cuda.KernelArgs).
		Pointer(m.buffers.qkv).Pointer(layer.qNorm).Pointer(layer.kNorm).Pointer(layer.ropeInverse).
		Pointer(layer.gate).Pointer(m.buffers.normed).Int32(int32(config.HiddenSize)).Float32(float32(config.NormEps)).
		Pointer(layer.keyCache).Pointer(layer.valueCache).
		Pointer(m.buffers.state).Int32(int32(layer.window)).
		Int32(int32(config.NumKVHeads)).Float32(attentionScale(config.HeadDim)).
		Pointer(m.buffers.attentionPartial).Pointer(m.buffers.attentionDone).
		Pointer(m.buffers.actQ).Pointer(m.buffers.actInfo)
	grid := cuda.Dim3{X: uint32(config.NumKVHeads), Y: attentionMaxSplits, Z: 1}
	l.launch(m.kernels.attentionDecode, grid, dim1(attentionThreads), args)
}

// attention は attention 層の演算部分（qkv → 正規化・RoPE・KV 書き込み・ゲート・attention 本体 → o_proj）
// を積み、結果を buffers.delta に書きます。
func (m *Model) attention(l *launcher, layer *deviceLayer) {
	m.gemv(l, layer.qkv, m.buffers.qkv)
	m.attend(l, layer)
	m.gemv(l, layer.o, m.buffers.delta)
}

// siluMul は、FFN の SiLU と乗算・int8 化（1 カーネル）を積みます。入力は buffers.gateUp です。
func (m *Model) siluMul(l *launcher) {
	intermediate := m.config.IntermediateSize
	args := new(cuda.KernelArgs).
		Pointer(m.buffers.gateUp).Int32(int32(intermediate)).Pointer(m.buffers.actQ).Pointer(m.buffers.actInfo)
	l.launch(m.kernels.siluMulQuantize, dim1(intermediate/threadsPerBlock), dim1(threadsPerBlock), args)
}

// feedForward は FFN（W1・W3 → SiLU と乗算 → W2）を積み、結果を buffers.delta に書きます。
func (m *Model) feedForward(l *launcher, layer *deviceLayer) {
	m.gemv(l, layer.gateUp, m.buffers.gateUp)
	m.siluMul(l)
	m.gemv(l, layer.down, m.buffers.delta)
}

// embed は、入力トークンの埋め込みを hidden に書く処理（1 カーネル）を積みます。
func (m *Model) embed(l *launcher) {
	hidden := m.config.HiddenSize
	args := new(cuda.KernelArgs).
		Pointer(m.embedding).Pointer(m.buffers.state).Pointer(m.buffers.hidden).Int32(int32(hidden))
	l.launch(m.kernels.embedLookup, dim1(hidden/threadsPerBlock), dim1(threadsPerBlock), args)
}

// argmaxPartial は、logits の部分ごとの最大値を求める処理（1 カーネル）を積みます。
func (m *Model) argmaxPartial(l *launcher) {
	args := new(cuda.KernelArgs).
		Pointer(m.buffers.logits).Int32(int32(m.config.VocabSize)).Pointer(m.buffers.partValue).Pointer(m.buffers.partIndex)
	l.launch(m.kernels.argmaxPartial, dim1(argmaxParts), dim1(threadsPerBlock), args)
}

// argmaxFinal は、全体の argmax を状態に書いて位置を 1 進める処理（1 カーネル）を積みます。
func (m *Model) argmaxFinal(l *launcher) {
	args := new(cuda.KernelArgs).
		Pointer(m.buffers.partValue).Pointer(m.buffers.partIndex).Int32(argmaxParts).Pointer(m.buffers.state)
	l.launch(m.kernels.argmaxFinalAdvance, dim1(1), dim1(threadsPerBlock), args)
}

// enqueueToken は、1 トークン分の全カーネルを順にストリームに積みます（完了は待たない）。
// 記録中のストリームに積めば CUDA Graph になります。手順は CPU 版の lfm3.Model.Forward と同じです。
//
//	hidden = embed(token)
//	各層: hidden += post_norm(operator(norm(hidden)))、hidden += post_ffn_norm(ffn(ffn_norm(hidden)))
//	logits = lm_head(final_norm(hidden))、argmax を状態に書いて位置を進める
func (m *Model) enqueueToken() error {
	return m.enqueueTokenWith(&launcher{stream: m.stream})
}

// enqueueTokenWith は enqueueToken と同じ処理を、渡された launcher で積みます（計測用の設定を使うため）。
func (m *Model) enqueueTokenWith(l *launcher) error {
	m.embed(l)
	m.residualNorm(l, 0, 0, m.layers[0].operatorNorm)

	for index := range m.layers {
		layer := &m.layers[index]
		if layer.kind == lfm3.LayerConv {
			m.shortConv(l, layer)
		} else {
			m.attention(l, layer)
		}
		m.residualNorm(l, m.buffers.delta, layer.postOperatorNorm, layer.ffnNorm)
		m.feedForward(l, layer)
		// FFN の後の正規化に続けて、次の層の入口の正規化（最後の層の後は最終正規化）まで行う。
		next := m.finalNorm
		if index+1 < len(m.layers) {
			next = m.layers[index+1].operatorNorm
		}
		m.residualNorm(l, m.buffers.delta, layer.postFFNNorm, next)
	}

	m.gemv(l, m.lmHead, m.buffers.logits)
	m.argmaxPartial(l)
	m.argmaxFinal(l)
	return l.err
}
