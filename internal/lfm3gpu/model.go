// Package lfm3gpu は、LFM3 の 1 トークン生成（decode）を GTX 1080 Ti（sm_61）で実行します（M2）。
//
// 重みは .pq4m（PQ4）から読み、計算は kernels/lfm3.cu のカーネルで行います。1 トークン分のカーネル
// （約 200 個）は CUDA Graph に 1 回だけ記録し、毎トークン再生します。トークンと位置は GPU メモリ上の
// 「状態」（int32 × 4）から読むため、位置が変わっても Graph を作り直す必要はありません。
//
// 計算の手順は CPU の参照実装（internal/lfm3 と internal/pq4）と同じです。違いは次の 2 点です。
//   - KV キャッシュを FP16 で持つ（CPU 版は FP32）
//   - 合計の順序が違う（並列に足すため）
//
// CUDA のコンテキストは OS スレッドごとに決まるため、Open から Close まで同じゴルーチンで呼び、
// そのゴルーチンは最初に runtime.LockOSThread を呼んでください。
package lfm3gpu

import (
	"errors"
	"fmt"
	"math"

	"github.com/cardy0828/pascallm/internal/cuda"
	"github.com/cardy0828/pascallm/internal/kernels"
	"github.com/cardy0828/pascallm/internal/lfm3"
	"github.com/cardy0828/pascallm/internal/pq4"
)

// 状態（GPU メモリ上の int32 × 4）の各要素の位置です。kernels/lfm3.cu の state と同じ並びです。
const (
	stateToken    = 0 // 入力トークン
	statePosition = 1 // 位置（0 始まり）
	stateArgmax   = 2 // 直前の 1 トークン処理での argmax（貪欲生成の次のトークン）
	stateCount    = 4
)

// カーネルの起動の大きさに関する定数です（kernels/lfm3.cu の前提と合わせる）。
const (
	threadsPerBlock    = 256  // 要素ごとのカーネルのスレッド数
	normThreads        = 1024 // residual_norm_quantize のスレッド数（1 スレッドブロックでベクトル全体を処理）
	normMaxPerThread   = 4    // residual_norm_quantize で 1 スレッドが持つ最大の要素数（kernels/lfm3.cu と同じ）
	gemvWarpsPerBlock  = 4    // GEMV の 1 スレッドブロックのワープ数
	gemvRowsPerWarp    = 4    // GEMV の 1 ワープが担当する行数（行列の行数はこの倍数であること）
	warpSize           = 32
	argmaxParts        = 256 // argmax の 1 段目のスレッドブロック数
	supportedHeadDim   = 128 // attention_decode が対応する head_dim
	activationInfoSize = 8   // ActBlockInfo（float32 × 2）のバイト数
	convMaxKernel      = 8   // short_conv_quantize が対応する最大の畳み込みの長さ（kernels/lfm3.cu の kConvMaxKernel）
)

// attention_decode の起動に関する定数です（kernels/lfm3.cu の kAttention* と合わせる）。
const (
	attentionThreads = 256 // 1 スレッドブロックのスレッド数
	attentionGroup   = 4   // 1 個の KV ヘッドを共有する query ヘッドの数（num_heads / num_kv_heads）
	// attentionMaxSplits は、見る位置の範囲を分ける最大の数（grid の y）です。実際に使う数はカーネルが
	// 範囲の長さから決めます（1 分割 128 位置以上）。16 にすると、全文脈で 2048 位置までは 1 ブロック 128 位置になります。
	// 計測（位置 4000、全文脈）では 8・16・32 の差は小さく、短い範囲の時間は分割の上限によりません。
	attentionMaxSplits = 16
	// attentionPartialFloats は、部分結果 1 個（1 ブロック・1 ヘッド）の float の数です（最大値・合計・重み付き和 128）。
	attentionPartialFloats = 2 + supportedHeadDim
)

// Options は GPU 実装の設定です。
type Options struct {
	// DeviceUUID は使う GPU の UUID です。例: ""（GTX 1080 Ti）
	DeviceUUID string
	// MaxContext は扱える最大の位置数（KV キャッシュの大きさ）です。例: 4096
	MaxContext int
}

// kernelSet は、使うカーネルの一覧です。
type kernelSet struct {
	embedLookup          *cuda.Function
	residualNormQuantize *cuda.Function
	gemvPQ4              *cuda.Function
	shortConvQuantize    *cuda.Function
	attentionDecode      *cuda.Function
	siluMulQuantize      *cuda.Function
	argmaxPartial        *cuda.Function
	argmaxFinalAdvance   *cuda.Function
	candidatesPartial    *cuda.Function
	candidatesFinal      *cuda.Function
}

// loadKernels はモジュールから全カーネルを名前で取り出します。
func loadKernels(module *cuda.Module) (kernelSet, error) {
	var set kernelSet
	targets := []struct {
		target **cuda.Function
		name   string
	}{
		{&set.embedLookup, "embed_lookup"},
		{&set.residualNormQuantize, "residual_norm_quantize"},
		{&set.gemvPQ4, "gemv_pq4"},
		{&set.shortConvQuantize, "short_conv_quantize"},
		{&set.attentionDecode, "attention_decode"},
		{&set.siluMulQuantize, "silu_mul_quantize"},
		{&set.argmaxPartial, "argmax_partial"},
		{&set.argmaxFinalAdvance, "argmax_final_advance"},
		{&set.candidatesPartial, "candidates_partial"},
		{&set.candidatesFinal, "candidates_final"},
	}
	for _, entry := range targets {
		function, err := module.Function(entry.name)
		if err != nil {
			return kernelSet{}, err
		}
		*entry.target = function
	}
	return set, nil
}

// activations は、1 トークンの計算で使い回す GPU 上の作業領域です。
type activations struct {
	state     cuda.DevicePtr // int32 × 4
	hidden    cuda.DevicePtr // 残差の流れ [hidden]
	normed    cuda.DevicePtr // 正規化した入力（FP32）[hidden]。attention のゲートが読む
	delta     cuda.DevicePtr // conv・attention・FFN の出力 [hidden]
	actQ      cuda.DevicePtr // int8 化した活性値 [最大の入力次元]
	actInfo   cuda.DevicePtr // int8 化の付帯情報 [最大の入力次元 / 32]
	convIn    cuda.DevicePtr // in_proj の出力 [3 × hidden]
	qkv       cuda.DevicePtr // q・k・v の GEMV の出力 [q + 2 × kv]
	gateUp    cuda.DevicePtr // FFN の W1・W3 の出力 [2 × intermediate]
	logits    cuda.DevicePtr // [vocab]
	partValue cuda.DevicePtr // argmax の途中結果 [argmaxParts] float32
	partIndex cuda.DevicePtr // [argmaxParts] int32

	// attentionPartial は、attention の位置の範囲を分けたときの部分結果
	// [num_kv_heads][attentionMaxSplits][attentionGroup][attentionPartialFloats] float32 です。
	attentionPartial cuda.DevicePtr
	// attentionDone は、KV ヘッドごとに終わったスレッドブロックの数 [num_kv_heads] int32 です。
	// 0 で始め、最後のブロックが 0 に戻します。
	attentionDone cuda.DevicePtr

	// 選び方の候補を絞る処理（candidates.go）の領域です。
	seenFlags           cuda.DevicePtr // ペナルティの対象の印 [vocab] uint8
	tokenCounts         cuda.DevicePtr // 生成した回数 [vocab] int32
	candidatePartValues cuda.DevicePtr // 1 段目の候補の値 [candidateParts][CandidateCount] float32
	candidatePartIDs    cuda.DevicePtr // 1 段目の候補の番号 [candidateParts][CandidateCount] int32
	candidatePartMax    cuda.DevicePtr // 1 段目の区間ごとの最大値 [candidateParts] float32
	candidatePartSum    cuda.DevicePtr // 1 段目の区間ごとの exp の合計 [candidateParts] float32
	candidateResult     cuda.DevicePtr // 2 段目の結果 [candidateResultFloats] × 4 バイト
}

// Model は GPU 上の LFM3 です。同時に 1 系列だけを扱います。
type Model struct {
	config     *lfm3.Config
	maxContext int
	position   int // 次に処理するトークンの位置（ホスト側の写し）
	// savedPosition は SaveState で保存した位置（なければ noSavedState）、savedHistory はそのときの
	// 全 shortconv 層の履歴の写し [conv 層][kernel_size-1][hidden] float32 です（snapshot.go）。
	savedPosition int
	savedHistory  cuda.DevicePtr
	useGraph      bool // true なら CUDA Graph を再生し、false ならカーネルを 1 個ずつ起動する

	context *cuda.Context
	stream  *cuda.Stream
	module  *cuda.Module
	kernels kernelSet
	memory  *deviceMemory

	embedding cuda.DevicePtr // [vocab][hidden] FP16
	finalNorm cuda.DevicePtr
	lmHead    deviceMatrix
	layers    []deviceLayer
	buffers   activations

	hostState  *cuda.HostBuffer // 状態の受け渡し用の固定メモリ（int32 × 4）
	hostLogits *cuda.HostBuffer // logits の受け取り用の固定メモリ（float32 × vocab）
	// hostSeen はペナルティの対象の印を送る固定メモリ（uint8 × vocab）、hostCandidates は候補の受け取り用です。
	hostSeen       *cuda.HostBuffer
	hostCandidates *cuda.HostBuffer
	graph          *cuda.GraphExec

	// 入力処理（prefill.go）のカーネル・作業領域と、トークン番号を送る固定メモリ（int32 × PrefillChunkTokens）です。
	prefillKernels    prefillKernelSet
	prefill           prefillBuffers
	hostPrefillTokens *cuda.HostBuffer
}

// validateConfig は、GPU のカーネルが前提とする大きさかを確かめます。
func validateConfig(config *lfm3.Config) error {
	switch {
	case config.HeadDim != supportedHeadDim:
		return fmt.Errorf("lfm3gpu: head_dim %d is not supported (only %d)", config.HeadDim, supportedHeadDim)
	case config.NumHeads != attentionGroup*config.NumKVHeads:
		return fmt.Errorf("lfm3gpu: num_attention_heads %d must be %d × num_key_value_heads %d",
			config.NumHeads, attentionGroup, config.NumKVHeads)
	case config.HiddenSize%threadsPerBlock != 0:
		return fmt.Errorf("lfm3gpu: hidden_size %d is not a multiple of %d", config.HiddenSize, threadsPerBlock)
	case config.HiddenSize > normThreads*normMaxPerThread:
		return fmt.Errorf("lfm3gpu: hidden_size %d is larger than %d", config.HiddenSize, normThreads*normMaxPerThread)
	case config.IntermediateSize%threadsPerBlock != 0:
		return fmt.Errorf("lfm3gpu: intermediate_size %d is not a multiple of %d", config.IntermediateSize, threadsPerBlock)
	case config.ConvKernelSize < 2:
		return fmt.Errorf("lfm3gpu: conv kernel size %d is too small", config.ConvKernelSize)
	case config.ConvKernelSize > convMaxKernel:
		return fmt.Errorf("lfm3gpu: conv kernel size %d is larger than %d", config.ConvKernelSize, convMaxKernel)
	}
	return nil
}

// Open は .pq4m を読み、重みを GPU に置き、1 トークン分の処理を CUDA Graph に記録します。
// 例: Open("./data/lfm3-1.7b-pq4-search-imatrix.pq4m", Options{DeviceUUID: "GPU-...", MaxContext: 4096})
func Open(path string, opts Options) (model *Model, err error) {
	if opts.MaxContext <= 0 {
		return nil, fmt.Errorf("lfm3gpu: MaxContext must be positive, got %d", opts.MaxContext)
	}
	config, source, err := lfm3.OpenPQ4Model(path)
	if err != nil {
		return nil, err
	}
	defer source.File.Close()
	if err := validateConfig(config); err != nil {
		return nil, err
	}
	weights, err := lfm3.LoadWeights(config, source)
	if err != nil {
		return nil, err
	}

	if err := cuda.Init(); err != nil {
		return nil, err
	}
	device, err := cuda.FindDeviceByUUID(opts.DeviceUUID)
	if err != nil {
		return nil, err
	}
	m := &Model{config: config, maxContext: opts.MaxContext, useGraph: true}
	// 途中で失敗したら、それまでに確保したものを解放する。
	defer func() {
		if err != nil {
			m.Close()
			model = nil
		}
	}()
	if m.context, err = cuda.NewPrimaryContext(device); err != nil {
		return nil, err
	}
	if m.stream, err = cuda.NewStream(); err != nil {
		return nil, err
	}
	if m.module, err = cuda.LoadModule(kernels.LFM3SM61); err != nil {
		return nil, err
	}
	if m.kernels, err = loadKernels(m.module); err != nil {
		return nil, err
	}
	if m.memory, err = newDeviceMemory(m.stream); err != nil {
		return nil, err
	}
	if err = m.uploadWeights(source, weights); err != nil {
		return nil, err
	}
	if err = m.allocateActivations(); err != nil {
		return nil, err
	}
	if err = m.preparePrefill(); err != nil {
		return nil, err
	}
	if err = m.prepareSnapshot(); err != nil {
		return nil, err
	}
	if m.hostState, err = cuda.AllocHost(4 * stateCount); err != nil {
		return nil, err
	}
	if m.hostLogits, err = cuda.AllocHost(4 * config.VocabSize); err != nil {
		return nil, err
	}
	if m.hostSeen, err = cuda.AllocHost(config.VocabSize); err != nil {
		return nil, err
	}
	if m.hostCandidates, err = cuda.AllocHost(4 * candidateResultFloats); err != nil {
		return nil, err
	}
	if err = m.recordGraph(); err != nil {
		return nil, err
	}
	return m, nil
}

// uploadWeights は、埋め込み表・各層・最終正規化・語彙出力を GPU に置きます。
func (m *Model) uploadWeights(source *lfm3.PQ4Source, weights *lfm3.Weights) error {
	embeddingBits, shape, err := source.File.Float16(lfm3.EmbeddingTensorName)
	if err != nil {
		return err
	}
	if len(shape) != 2 || shape[0] != m.config.VocabSize || shape[1] != m.config.HiddenSize {
		return fmt.Errorf("lfm3gpu: embedding has shape %v", shape)
	}
	if m.embedding, err = upload(m.memory, embeddingBits); err != nil {
		return err
	}
	if m.finalNorm, err = upload(m.memory, weights.FinalNorm); err != nil {
		return err
	}
	lmHead, err := asPQ4(weights.LMHead, "lm_head")
	if err != nil {
		return err
	}
	if m.lmHead, err = uploadMatrices(m.memory, "lm_head", lmHead); err != nil {
		return err
	}
	m.layers = make([]deviceLayer, len(weights.Layers))
	for index := range weights.Layers {
		if m.layers[index], err = uploadLayer(m.memory, m.config, index, &weights.Layers[index], m.maxContext); err != nil {
			return err
		}
	}
	return nil
}

// allocateActivations は作業領域を確保します。
func (m *Model) allocateActivations() error {
	config := m.config
	hidden := config.HiddenSize
	attentionWidth := config.NumHeads * config.HeadDim
	kvWidth := config.NumKVHeads * config.HeadDim
	maxInput := max(hidden, config.IntermediateSize, attentionWidth)
	sizes := []struct {
		target *cuda.DevicePtr
		bytes  int
	}{
		{&m.buffers.state, 4 * stateCount},
		{&m.buffers.hidden, 4 * hidden},
		{&m.buffers.normed, 4 * hidden},
		{&m.buffers.delta, 4 * hidden},
		{&m.buffers.actQ, maxInput},
		{&m.buffers.actInfo, activationInfoSize * (maxInput / pq4.BlockSize)},
		{&m.buffers.convIn, 4 * 3 * hidden},
		{&m.buffers.qkv, 4 * (attentionWidth + 2*kvWidth)},
		{&m.buffers.gateUp, 4 * 2 * config.IntermediateSize},
		{&m.buffers.logits, 4 * config.VocabSize},
		{&m.buffers.partValue, 4 * argmaxParts},
		{&m.buffers.partIndex, 4 * argmaxParts},
		{&m.buffers.attentionPartial, 4 * config.NumKVHeads * attentionMaxSplits * attentionGroup * attentionPartialFloats},
		{&m.buffers.attentionDone, m.attentionDoneBytes()},
		{&m.buffers.seenFlags, config.VocabSize},
		{&m.buffers.tokenCounts, 4 * config.VocabSize},
		{&m.buffers.candidatePartValues, 4 * candidateParts * CandidateCount},
		{&m.buffers.candidatePartIDs, 4 * candidateParts * CandidateCount},
		{&m.buffers.candidatePartMax, 4 * candidateParts},
		{&m.buffers.candidatePartSum, 4 * candidateParts},
		{&m.buffers.candidateResult, 4 * candidateResultFloats},
	}
	for _, entry := range sizes {
		pointer, err := m.memory.alloc(entry.bytes)
		if err != nil {
			return err
		}
		*entry.target = pointer
	}
	return cuda.MemsetBytes(m.stream, m.buffers.attentionDone, 0, m.attentionDoneBytes())
}

// attentionDoneBytes は buffers.attentionDone のバイト数（int32 × num_kv_heads）です。例: 4 ヘッドなら 16
func (m *Model) attentionDoneBytes() int {
	return 4 * m.config.NumKVHeads
}

// recordGraph は 1 トークン分の処理を CUDA Graph に記録し、実行可能な形にします。
func (m *Model) recordGraph() error {
	if err := m.stream.BeginCapture(); err != nil {
		return err
	}
	enqueueError := m.enqueueToken()
	// 積む途中で失敗しても、記録は必ず終える（ストリームを記録中のまま残さないため）。
	graph, endError := m.stream.EndCapture()
	if enqueueError != nil {
		if graph != nil {
			graph.Close()
		}
		return enqueueError
	}
	if endError != nil {
		return endError
	}
	defer graph.Close()
	executable, err := graph.Instantiate()
	if err != nil {
		return err
	}
	m.graph = executable
	return nil
}

// Config はモデルの設定を返します。
func (m *Model) Config() *lfm3.Config { return m.config }

// Position は、次に処理するトークンの位置を返します。
func (m *Model) Position() int { return m.position }

// MaxContext は、扱える最大の位置数（Options.MaxContext）を返します。例: 4096
func (m *Model) MaxContext() int { return m.maxContext }

// AllocatedBytes は、確保した GPU メモリの合計（重み・KV キャッシュ・作業領域）を返します。
func (m *Model) AllocatedBytes() int64 { return m.memory.totalBytes }

// SetUseGraph は、CUDA Graph を再生するか（true）、カーネルを 1 個ずつ起動するか（false）を選びます。
// 計算の内容は同じで、起動の方法だけが変わります（速度の比較用）。
func (m *Model) SetUseGraph(use bool) { m.useGraph = use }

// Reset は、畳み込みの履歴と attention の終了数を 0 にし、位置を 0 に戻します。
// 終了数は通常 attention_decode の最後のブロックが 0 に戻しますが、途中で失敗した場合に備えて消します。
// KV キャッシュは、位置より先の要素を読まないため消しません。SaveState で保存した状態は無効になります。
func (m *Model) Reset() error {
	m.savedPosition = noSavedState
	if err := cuda.MemsetBytes(m.stream, m.buffers.attentionDone, 0, m.attentionDoneBytes()); err != nil {
		return err
	}
	for _, layer := range m.layers {
		if layer.kind == lfm3.LayerConv {
			if err := cuda.MemsetBytes(m.stream, layer.convHistory, 0, m.convHistoryBytes()); err != nil {
				return err
			}
		}
	}
	m.position = 0
	return nil
}

// checkInput は、入力トークンと位置が扱える範囲かを確かめます。
func (m *Model) checkInput(token int) error {
	if token < 0 || token >= m.config.VocabSize {
		return fmt.Errorf("lfm3gpu: token %d is outside the vocabulary (size %d)", token, m.config.VocabSize)
	}
	if m.position >= m.maxContext {
		return fmt.Errorf("lfm3gpu: position %d reached MaxContext %d", m.position, m.maxContext)
	}
	return nil
}

// runToken は 1 トークン分の処理をストリームに積みます（Graph の再生、またはカーネルの個別起動）。
func (m *Model) runToken() error {
	if m.useGraph {
		return m.graph.Launch(m.stream)
	}
	return m.enqueueToken()
}

// writeInput は、入力トークンと現在の位置を GPU の状態に書く処理を積みます（完了は待たない）。
func (m *Model) writeInput(token int) error {
	state := m.hostState.Int32s()
	state[stateToken] = int32(token)
	state[statePosition] = int32(m.position)
	state[stateArgmax] = 0
	return m.hostState.CopyToDeviceAsync(m.stream, m.buffers.state)
}

// Forward は token を 1 個処理して位置を 1 進め、その位置の logits（長さ vocab）を返します。
// 戻り値は固定メモリを指し、次の呼び出しで上書きされるため、残す場合はコピーしてください。
// CPU の参照実装（lfm3.Model.Forward）と同じ使い方ができます。
func (m *Model) Forward(token int) ([]float32, error) {
	if err := m.checkInput(token); err != nil {
		return nil, err
	}
	if err := m.writeInput(token); err != nil {
		return nil, err
	}
	if err := m.runToken(); err != nil {
		return nil, err
	}
	if err := m.hostLogits.CopyFromDeviceAsync(m.stream, m.buffers.logits); err != nil {
		return nil, err
	}
	if err := m.stream.Synchronize(); err != nil {
		return nil, err
	}
	m.position++
	return m.hostLogits.Float32s(), nil
}

// CopyLogits は、直前に処理した位置の logits（長さ vocab）を GPU から読み戻して返します。
// Feed の後で argmax 以外の選び方が必要になったときに使います。戻り値は Forward と同じく固定メモリを指し、
// 次の呼び出しで上書きされます。例: Feed(token) の後に CopyLogits()
func (m *Model) CopyLogits() ([]float32, error) {
	if m.position == 0 {
		return nil, errors.New("lfm3gpu: CopyLogits needs a processed token")
	}
	if err := m.hostLogits.CopyFromDeviceAsync(m.stream, m.buffers.logits); err != nil {
		return nil, err
	}
	if err := m.stream.Synchronize(); err != nil {
		return nil, err
	}
	return m.hostLogits.Float32s(), nil
}

// Feed は token を 1 個処理して位置を 1 進め、logits の argmax（貪欲法での次のトークン）を返します。
// logits はコピーしないため、Forward より速く進みます。プロンプトの入力に使います。
func (m *Model) Feed(token int) (int, error) {
	if err := m.checkInput(token); err != nil {
		return 0, err
	}
	if err := m.writeInput(token); err != nil {
		return 0, err
	}
	return m.finishGreedyStep()
}

// ContinueGreedy は、直前の argmax を入力として 1 トークン処理し、次の argmax を返します。
// 状態は GPU 側で更新済みなので、ホストからは何も書きません（貪欲生成の 1 ステップ）。
// 直前に Feed または ContinueGreedy を呼んでいる必要があります。
func (m *Model) ContinueGreedy() (int, error) {
	if m.position == 0 {
		return 0, errors.New("lfm3gpu: ContinueGreedy needs a previous Feed")
	}
	if m.position >= m.maxContext {
		return 0, fmt.Errorf("lfm3gpu: position %d reached MaxContext %d", m.position, m.maxContext)
	}
	return m.finishGreedyStep()
}

// finishGreedyStep は、1 トークン分の処理を積み、状態を読み戻して argmax を返します。
func (m *Model) finishGreedyStep() (int, error) {
	if err := m.runToken(); err != nil {
		return 0, err
	}
	if err := m.hostState.CopyFromDeviceAsync(m.stream, m.buffers.state); err != nil {
		return 0, err
	}
	if err := m.stream.Synchronize(); err != nil {
		return 0, err
	}
	m.position++
	state := m.hostState.Int32s()
	if int(state[statePosition]) != m.position {
		return 0, fmt.Errorf("lfm3gpu: GPU position %d differs from host position %d", state[statePosition], m.position)
	}
	return int(state[stateArgmax]), nil
}

// Close は、Graph・GPU メモリ・固定メモリ・モジュール・ストリーム・コンテキストを解放します。
// 途中まで作った Model（Open の失敗時）にも使えます。最初のエラーを返します。
func (m *Model) Close() error {
	var first error
	keep := func(err error) {
		if err != nil && first == nil {
			first = err
		}
	}
	if m.graph != nil {
		keep(m.graph.Close())
		m.graph = nil
	}
	if m.memory != nil {
		keep(m.memory.close())
		m.memory = nil
	}
	for _, buffer := range []**cuda.HostBuffer{&m.hostState, &m.hostLogits, &m.hostSeen, &m.hostCandidates, &m.hostPrefillTokens} {
		if *buffer != nil {
			keep((*buffer).Close())
			*buffer = nil
		}
	}
	if m.module != nil {
		keep(m.module.Close())
		m.module = nil
	}
	if m.stream != nil {
		keep(m.stream.Close())
		m.stream = nil
	}
	if m.context != nil {
		keep(m.context.Close())
		m.context = nil
	}
	return first
}

// attentionScale は attention のスコアに掛ける 1/sqrt(head_dim) です（CPU 版と同じく float64 で計算して丸める）。
// 例: head_dim = 128 -> 0.0884
func attentionScale(headDim int) float32 {
	return float32(1 / math.Sqrt(float64(headDim)))
}
