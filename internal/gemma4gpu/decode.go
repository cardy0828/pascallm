package gemma4gpu

import (
	"fmt"

	"github.com/cardy0828/pascallm/internal/cuda"
	"github.com/cardy0828/pascallm/internal/gpucore"
)

// この file は、1 トークン分のカーネルを順に積む処理です。記録中のストリームに積めば CUDA Graph になります。
// 手順は CPU 版の gemma4.Model.Forward と同じです。

// gemv は out = matrix · (int8 化済みの活性値) を積みます。活性値は buffers.actQ・actInfo から読みます。
func (m *Model) gemv(l *gpucore.Launcher, matrix gpucore.Matrix, out cuda.DevicePtr) {
	gpucore.Gemv(l, m.kernels.gemv, matrix, m.buffers.actQ, m.buffers.actInfo, out)
}

// residualNorm は、delta があれば hidden = (hidden + RMSNorm(delta) × deltaWeight) × hiddenScale を行い、
// RMSNorm(hidden) × nextWeight を int8 化して actQ・actInfo に書く処理を積みます。
// delta が 0 なら加算をせず、正規化と int8 化だけを行います（最初の層の入口）。
func (m *Model) residualNorm(l *gpucore.Launcher, delta, deltaWeight, nextWeight cuda.DevicePtr, hiddenScale float32) {
	args := new(cuda.KernelArgs).
		Pointer(m.buffers.hidden).Pointer(delta).Pointer(deltaWeight).Pointer(nextWeight).
		Float32(float32(m.config.NormEps)).Int32(int32(m.config.HiddenSize)).Float32(hiddenScale).
		Pointer(m.buffers.actQ).Pointer(m.buffers.actInfo)
	l.Launch(m.kernels.residualNorm, gpucore.Dim1(1), gpucore.Dim1(normThreads), args)
}

// embed は、入力トークンの埋め込み × sqrt(hidden) を hidden に書く処理を積みます。
func (m *Model) embed(l *gpucore.Launcher) {
	hidden := m.config.HiddenSize
	args := new(cuda.KernelArgs).
		Pointer(m.embedding.Packed).Pointer(m.embedding.Scales).Pointer(m.buffers.state).Pointer(m.buffers.hidden).
		Int32(int32(hidden)).Float32(m.config.EmbeddingScale())
	l.Launch(m.kernels.embed, gpucore.Dim1(hidden/threadsPerBlock), gpucore.Dim1(threadsPerBlock), args)
}

// prepareVectorBlocks は、attention の前処理のスレッドブロック数を返します（1 ブロック 8 本のベクトル）。
// 例: 局所の層（q 16 本 + k 8 本 + v 8 本）なら 4、全文脈の層（16 + 1 + 1）なら 3
func (m *Model) prepareVectorBlocks(layer *deviceLayer) int {
	vectors := m.config.NumHeads + keyAndValueVectors*layer.config.KVHeads
	return gpucore.CeilDiv(vectors, warpsPerBlock)
}

// attend は、attention の q・k・v の後から o の射影の前までを積みます（3 個のカーネル）。
//
//  1. 前処理: q・k の正規化と RoPE、v の正規化、KV キャッシュへの書き込み
//  2. 本体: 位置の範囲を分けて、スコア・重み・重み付き和の途中結果を求める
//  3. 合成: 途中結果をまとめ、出力を int8 化する（o の射影の入力）
//
// 入力は buffers.qkv（q・k・v の GEMV の出力）です。
func (m *Model) attend(l *gpucore.Launcher, layer *deviceLayer) {
	config := m.config
	b := &m.buffers
	headDim := layer.config.HeadDim
	kvHeads := layer.config.KVHeads
	prepare := new(cuda.KernelArgs).
		Pointer(b.qkv).Pointer(layer.qNorm).Pointer(layer.kNorm).Pointer(layer.ropeInverse).
		Float32(float32(config.NormEps)).Pointer(b.state).
		Int32(int32(headDim)).Int32(int32(config.NumHeads)).Int32(int32(kvHeads)).Int32(int32(layer.valueOffset)).
		Pointer(b.query).Pointer(layer.keyCache).Pointer(layer.valueCache).Int32(int32(layer.ringSize))
	l.Launch(m.kernels.attentionPrepare, gpucore.Dim1(m.prepareVectorBlocks(layer)), gpucore.Dim1(threadsPerBlock), prepare)

	body := new(cuda.KernelArgs).
		Pointer(b.query).Pointer(layer.keyCache).Pointer(layer.valueCache).Pointer(b.state).
		Int32(int32(layer.window)).Int32(int32(layer.ringSize)).Int32(int32(kvHeads)).
		Int32(int32(config.NumHeads / kvHeads)).Float32(attentionScale).Pointer(b.partial)
	kernel := m.kernels.attentionGlobal
	if layer.config.Sliding {
		kernel = m.kernels.attentionLocal
	}
	l.Launch(kernel, gpucore.Dim2(config.NumHeads/layer.groupSize, layer.maxSplits), gpucore.Dim1(threadsPerBlock), body)

	combine := new(cuda.KernelArgs).
		Pointer(b.partial).Pointer(b.state).Int32(int32(layer.window)).Int32(int32(layer.maxSplits)).
		Int32(int32(headDim)).Int32(int32(layer.groupSize)).Int32(int32(config.NumHeads)).
		Pointer(b.actQ).Pointer(b.actInfo)
	l.Launch(m.kernels.attentionCombine, gpucore.Dim1(layer.queryWidth(config.NumHeads)/threadsPerBlock),
		gpucore.Dim1(threadsPerBlock), combine)
}

// geluMul は、FFN の GELU と乗算・int8 化（1 カーネル）を積みます。入力は buffers.gateUp です。
func (m *Model) geluMul(l *gpucore.Launcher) {
	size := m.config.FeedForwardSize
	args := new(cuda.KernelArgs).Pointer(m.buffers.gateUp).Int32(int32(size)).Pointer(m.buffers.actQ).Pointer(m.buffers.actInfo)
	l.Launch(m.kernels.geluMul, gpucore.Dim1(size/threadsPerBlock), gpucore.Dim1(threadsPerBlock), args)
}

// softcap は、logits に cap × tanh(logits / cap) を適用する処理を積みます。cap が 0 なら何も積みません。
func (m *Model) softcap(l *gpucore.Launcher) {
	limit := float32(m.config.FinalLogitSoftcap)
	if limit == 0 {
		return
	}
	vocab := m.config.VocabSize
	args := new(cuda.KernelArgs).Pointer(m.buffers.logits).Int32(int32(vocab)).Float32(limit)
	l.Launch(m.kernels.softcap, gpucore.Dim1(gpucore.CeilDiv(vocab, threadsPerBlock)), gpucore.Dim1(threadsPerBlock), args)
}

// argmax は、logits の argmax を状態に書いて位置を 1 進める処理（2 個のカーネル）を積みます。
func (m *Model) argmax(l *gpucore.Launcher) {
	b := &m.buffers
	partial := new(cuda.KernelArgs).
		Pointer(b.logits).Int32(int32(m.config.VocabSize)).Pointer(b.partValue).Pointer(b.partIndex)
	l.Launch(m.kernels.argmaxPartial, gpucore.Dim1(argmaxParts), gpucore.Dim1(threadsPerBlock), partial)
	final := new(cuda.KernelArgs).Pointer(b.partValue).Pointer(b.partIndex).Int32(argmaxParts).Pointer(b.state)
	l.Launch(m.kernels.argmaxFinalAdvance, gpucore.Dim1(1), gpucore.Dim1(threadsPerBlock), final)
}

// noScale は、residualNorm で層の出力の係数を掛けないとき（attention の後）の値です。
const noScale = 1.0

// enqueueToken は、1 トークン分の全カーネルを順にストリームに積みます（完了は待たない）。
//
//	hidden = embed(token) × sqrt(hidden)
//	各層: hidden += post_attention_norm(attention(attn_norm(hidden)))
//	      hidden = (hidden + post_ffw_norm(ffn(ffn_norm(hidden)))) × layer_output_scale
//	logits = softcap(lm_head(output_norm(hidden)))、argmax を状態に書いて位置を進める
func (m *Model) enqueueToken(l *gpucore.Launcher) error {
	m.enqueueLayers(l, len(m.layers))
	m.gemv(l, m.embedding, m.buffers.logits)
	m.softcap(l)
	m.argmax(l)
	return l.Err
}

// enqueueLayers は、埋め込みと、最初の count 層を積みます（count が全層なら、最後に最終正規化まで行う）。
// 終わった時点の buffers.hidden は、count 層を終えた残差の流れです。
func (m *Model) enqueueLayers(l *gpucore.Launcher, count int) {
	b := &m.buffers
	m.embed(l)
	m.residualNorm(l, 0, 0, m.layers[0].attnNorm, noScale)

	for index := 0; index < count; index++ {
		layer := &m.layers[index]
		m.gemv(l, layer.qkv, b.qkv)
		m.attend(l, layer)
		m.gemv(l, layer.o, b.delta)
		m.residualNorm(l, b.delta, layer.postAttnNorm, layer.ffnNorm, noScale)

		m.gemv(l, layer.gateUp, b.gateUp)
		m.geluMul(l)
		m.gemv(l, layer.down, b.delta)
		// FFN の後の正規化と係数に続けて、次の層の入口の正規化（最後の層の後は最終正規化）まで行う。
		next := m.outputNorm
		if index+1 < len(m.layers) {
			next = m.layers[index+1].attnNorm
		}
		m.residualNorm(l, b.delta, layer.postFFNNorm, next, layer.outScale)
	}
}

// DebugHidden は、token を現在の位置で最初の layers 層だけ処理し、その時点の残差の流れ（長さ hidden）を返します。
// 位置は進めません（KV キャッシュの現在の位置には書くが、次の Forward が同じ位置に書き直す）。
// CPU の参照実装（gemma4.Model.Trace）と層ごとに比べるための確認用です。
// 例: DebugHidden(token, 0) は埋め込み × sqrt(hidden)、DebugHidden(token, 1) は 1 層目の後
func (m *Model) DebugHidden(token, layers int) ([]float32, error) {
	if err := m.checkInput(token); err != nil {
		return nil, err
	}
	if layers < 0 || layers > len(m.layers) {
		return nil, fmt.Errorf("gemma4gpu: DebugHidden needs 0 to %d layers, got %d", len(m.layers), layers)
	}
	if err := m.writeInput(token); err != nil {
		return nil, err
	}
	l := &gpucore.Launcher{Stream: m.stream}
	m.enqueueLayers(l, layers)
	if l.Err != nil {
		return nil, l.Err
	}
	hidden := make([]float32, m.config.HiddenSize)
	if err := cuda.CopyFromDevice(m.stream, hidden, m.buffers.hidden); err != nil {
		return nil, err
	}
	return hidden, nil
}

// MeasureToken は、token を 1 個、カーネルを 1 個ずつ起動して処理し、カーネルの種類ごとの GPU 時間
// （マイクロ秒）と起動回数を返します。位置は 1 進みます（確認用。起動ごとにイベントを置くため、通常より少し遅い）。
// 例: times, launches, err := model.MeasureToken(token)
func (m *Model) MeasureToken(token int) (map[string]float64, map[string]int, error) {
	if err := m.checkInput(token); err != nil {
		return nil, nil, err
	}
	if err := m.writeInput(token); err != nil {
		return nil, nil, err
	}
	timer := &gpucore.KernelTimer{}
	defer timer.Close()
	if err := m.enqueueToken(&gpucore.Launcher{Stream: m.stream, Timer: timer}); err != nil {
		return nil, nil, err
	}
	if err := m.stream.Synchronize(); err != nil {
		return nil, nil, err
	}
	m.position++
	return timer.Totals()
}
