// Package gemma4gpu は、Gemma 4（12B など）の生成と入力処理を GTX 1080 Ti（sm_61）で実行します（M7）。
//
// 重みは、GGUF の Q4_0 を詰め替えた .pq4m から読み、計算は kernels/gemma4.cu（と、そこが取り込む lfm3.cu の
// 共通のカーネル）で行います。1 トークン分のカーネルは CUDA Graph に 1 回だけ記録し、毎トークン再生します。
// 使い方は internal/lfm3gpu と同じです（Forward・Feed・ContinueGreedy・FeedCandidates・Prefill・SaveState など）。
//
// 計算の手順は CPU の参照実装（internal/gemma4 の ActivationsInt8）と同じです。違いは次の 2 点です。
//   - KV キャッシュは既定 FP16。K FP32 / V FP16 と診断用 K/V FP32 も選べる
//   - 合計の順序が違う（並列に足すため）
//
// KV キャッシュの持ち方:
//   - 全文脈の層は、最大の位置数（Options.MaxContext）の分を確保します。
//   - 局所の層は、直近の SlidingWindow 個しか見ないため、「窓 + 余裕」個の輪状バッファにします（位置 p は
//     スロット p mod 輪の大きさ）。余裕（localRingSlack）は、入力処理が 1 区切りをまとめて書けることと、
//     保存した位置に戻せること（RestoreState）のためのものです。
//
// CUDA のコンテキストは OS スレッドごとに決まるため、Open から Close まで同じゴルーチンで呼び、
// そのゴルーチンは最初に runtime.LockOSThread を呼んでください。
package gemma4gpu

import (
	"errors"
	"fmt"

	"github.com/cardy0828/pascallm/internal/cuda"
	"github.com/cardy0828/pascallm/internal/gemma4"
	"github.com/cardy0828/pascallm/internal/gpucore"
	"github.com/cardy0828/pascallm/internal/pq4"
)

// 状態（GPU メモリ上の int32 × 4）の各要素の位置です。kernels/lfm3.cu の state と同じ並びです。
const (
	stateToken    = 0 // 入力トークン
	statePosition = 1 // 位置（0 始まり）
	stateArgmax   = 2 // 直前の 1 トークン処理での argmax（貪欲生成の次のトークン）
	stateCount    = 4
)

// カーネルの起動の大きさに関する定数です（kernels/gemma4.cu・lfm3.cu の前提と合わせる）。
const (
	threadsPerBlock  = 256  // 要素ごとのカーネルと attention のカーネルのスレッド数
	normThreads      = 1024 // g4_residual_norm_quantize のスレッド数（1 スレッドブロックでベクトル全体を処理）
	normMaxPerThread = 4    // g4_residual_norm_quantize で 1 スレッドが持つ最大の要素数
	argmaxParts      = 256  // argmax の 1 段目のスレッドブロック数
	warpsPerBlock    = threadsPerBlock / gpucore.WarpSize
	float32Bytes     = 4
	int32Bytes       = 4
	float16Bytes     = 2
	// keyAndValueVectors は、attention の前処理で KV ヘッド 1 個につき作るベクトルの数（k と v）です。
	keyAndValueVectors = 2
)

// attention のカーネルの寸法です（kernels/gemma4.cu の g4_attention_local・g4_attention_global と合わせる）。
const (
	localHeadDim       = 256 // g4_attention_local が対応する head_dim
	localGroupSize     = 2   // g4_attention_local の 1 ブロックが受け持つ query ヘッドの数
	globalHeadDim      = 512 // g4_attention_global が対応する head_dim
	globalGroupSize    = 4   // g4_attention_global の 1 ブロックが受け持つ query ヘッドの数
	minPositionsPerCut = 128 // 位置の範囲を分けるとき、1 分割に入れる最小の位置数（kG4MinPositionsPerSplit）
	// globalMaxSplits は、全文脈の層で位置の範囲を分ける最大の数（grid の y）です。
	// 4 組 × 16 分割 = 64 ブロックで、2048 位置までは 1 分割 128 位置になります。
	globalMaxSplits = 16
	// partialHeaderFloats は、途中結果の先頭の float の数（最大値、合計）です（kG4PartialHeader）。
	partialHeaderFloats = 2
	// attentionScale は、attention のスコアに掛ける値です。Gemma 4 は q と k を正規化するため、1 です。
	attentionScale = 1.0
)

// localRingSlack は、局所の層の輪状バッファを窓より大きくする分（スロット数）です。
//
//   - 入力処理は設定した区切りの k・v をまとめて書いてから attention を計算します。区切りの最初のトークンが
//     見る位置（窓の分）を、同じ区切りの後ろのトークンが上書きしないために、余裕は区切りの長さ以上が必要です。
//   - RestoreState で保存した位置に戻せるのは、保存してから進んだ位置数が余裕以下のときです。
//
// CompactVRAM を選ぶとこの余裕を max(入力チャンク幅, MTP検証行数) に縮めます。
// 例: 窓 1024、余裕 1024 なら輪は 2048 スロット（12B で 40 層 × 16 MB = 655 MB）
const localRingSlack = 1024

// Options は GPU 実装の設定です。
type Options struct {
	// CompactVRAM は重み・KVを一括確保し、局所 KV の保存点の余裕を入力の区切りに合わせます。
	CompactVRAM bool
	// PrefillChunk は同時に処理する入力の行数です。0 は既定値、例は 256。
	PrefillChunk int
	// MTPGreedyOnly はサンプリング/文法で従来経路を使う比較設定です。
	MTPGreedyOnly bool
	// KVCache は fp16（既定）、k-fp32、fp32（診断用）の格納形式です。
	KVCache string
	// MTPPath は任意の Gemma4 12B assistant Q4_0 GGUF です。空なら先読みしません。
	MTPPath string
	// MTPDraftTokens は固定候補数（0 は 3）、MTPAdaptive は候補数の自動調整です。
	MTPDraftTokens int
	MTPAdaptive    bool
	// DeviceUUID は使う GPU の UUID です。例: ""（GTX 1080 Ti）
	DeviceUUID string
	// MaxContext は扱える最大の位置数（全文脈の層の KV キャッシュの大きさ）です。例: 8192
	MaxContext int
}

// kernelSet は、生成で使うカーネルの一覧です。
type kernelSet struct {
	embed              *cuda.Function
	residualNorm       *cuda.Function
	gemv               *cuda.Function
	attentionPrepare   *cuda.Function
	attentionLocal     *cuda.Function
	attentionGlobal    *cuda.Function
	attentionCombine   *cuda.Function
	geluMul            *cuda.Function
	softcap            *cuda.Function
	argmaxPartial      *cuda.Function
	argmaxFinalAdvance *cuda.Function
	candidatesPartial  *cuda.Function
	candidatesFinal    *cuda.Function
}

// namedKernel は、モジュールから取り出すカーネルの名前と、入れる先です。
type namedKernel struct {
	target **cuda.Function
	name   string
}

// loadNamed は、モジュールからカーネルを名前で取り出して、それぞれの入れる先に入れます。
func loadNamed(module *cuda.Module, entries []namedKernel) error {
	for _, entry := range entries {
		function, err := module.Function(entry.name)
		if err != nil {
			return err
		}
		*entry.target = function
	}
	return nil
}

// loadKernels はモジュールから生成用のカーネルを取り出します。
func loadKernels(module *cuda.Module) (kernelSet, error) {
	var set kernelSet
	err := loadNamed(module, []namedKernel{
		{&set.embed, "g4_embed_pq4"},
		{&set.residualNorm, "g4_residual_norm_quantize"},
		{&set.gemv, "gemv_pq4"},
		{&set.attentionPrepare, "g4_attention_prepare"},
		{&set.attentionLocal, "g4_attention_local"},
		{&set.attentionGlobal, "g4_attention_global"},
		{&set.attentionCombine, "g4_attention_combine"},
		{&set.geluMul, "g4_gelu_mul_quantize"},
		{&set.softcap, "g4_softcap"},
		{&set.argmaxPartial, "argmax_partial"},
		{&set.argmaxFinalAdvance, "argmax_final_advance"},
		{&set.candidatesPartial, "g4_candidates_partial"},
		{&set.candidatesFinal, "candidates_final"},
	})
	return set, err
}

// deviceLayer は GPU 上の 1 層分の重みと状態です。
type deviceLayer struct {
	config gemma4.LayerConfig

	attnNorm     cuda.DevicePtr // [hidden] float32
	postAttnNorm cuda.DevicePtr
	ffnNorm      cuda.DevicePtr
	postFFNNorm  cuda.DevicePtr
	qNorm        cuda.DevicePtr // [head_dim] float32
	kNorm        cuda.DevicePtr
	ropeInverse  cuda.DevicePtr // [head_dim / 2] float32
	outScale     float32        // 層の出力に掛ける係数

	qkv    gpucore.Matrix // q・k（・v）を行方向につないだもの
	o      gpucore.Matrix // [hidden, num_heads × head_dim]
	gateUp gpucore.Matrix // gate と up を行方向につないだもの [2 × ffn, hidden]
	down   gpucore.Matrix // [hidden, ffn]

	// valueOffset は、qkv の GEMV の出力の中で v の元がある位置（float の数）です。
	// v の行列がある層は k の後ろ、ない層（全文脈）は k と同じ位置です。
	valueOffset int
	qkvWidth    int // qkv の GEMV の出力の長さ

	keyCache   cuda.DevicePtr // [ringSize][kv_heads × head_dim]。形式は Model.keyBytes
	valueCache cuda.DevicePtr
	ringSize   int // KV キャッシュのスロット数
	window     int // 局所の層なら sliding_window、全文脈の層なら 0

	groupSize int // attention の 1 ブロックが受け持つ query ヘッドの数
	maxSplits int // attention の位置の範囲を分ける最大の数（grid の y）
}

// queryWidth は、層の q の出力の次元数を返します。例: 局所 4096、全文脈 8192
func (l *deviceLayer) queryWidth(numHeads int) int { return numHeads * l.config.HeadDim }

// kvWidth は、層の k（v）の出力の次元数を返します。例: 局所 2048、全文脈 512
func (l *deviceLayer) kvWidth() int { return l.config.KVHeads * l.config.HeadDim }

// partialFloats は、attention の途中結果の float の数を返します
// （[組][分割][ヘッド][最大値, 合計, 重み付き和 head_dim 個]）。
func (l *deviceLayer) partialFloats(numHeads int) int {
	return numHeads * l.maxSplits * (partialHeaderFloats + l.config.HeadDim)
}

// activations は、1 トークンの計算で使い回す GPU 上の作業領域です。
type activations struct {
	state     cuda.DevicePtr // int32 × 4
	hidden    cuda.DevicePtr // 残差の流れ [hidden]
	delta     cuda.DevicePtr // attention・FFN の出力 [hidden]
	actQ      cuda.DevicePtr // int8 化した活性値 [最大の入力次元]
	actInfo   cuda.DevicePtr // int8 化の付帯情報 [最大の入力次元 / 32]
	qkv       cuda.DevicePtr // q・k・v の GEMV の出力 [最大の qkv の長さ]
	query     cuda.DevicePtr // 正規化と RoPE の後の q [最大の q の長さ] float32
	partial   cuda.DevicePtr // attention の途中結果（層の最大の大きさ）
	gateUp    cuda.DevicePtr // FFN の gate・up の出力 [2 × ffn]
	logits    cuda.DevicePtr // [vocab]
	partValue cuda.DevicePtr // argmax の途中結果 [argmaxParts] float32
	partIndex cuda.DevicePtr // [argmaxParts] int32

	// 選び方の候補を絞る処理（candidates.go）の領域です。
	seenFlags           cuda.DevicePtr // ペナルティの対象の印 [vocab] uint8
	tokenCounts         cuda.DevicePtr // 生成した回数 [vocab] int32
	candidatePartValues cuda.DevicePtr // 1 段目の候補の値 [CandidateParts][CandidateCount] float32
	candidatePartIDs    cuda.DevicePtr // 1 段目の候補の番号 [CandidateParts][CandidateCount] int32
	candidatePartMax    cuda.DevicePtr // 1 段目の区間ごとの最大値 [CandidateParts] float32
	candidatePartSum    cuda.DevicePtr // 1 段目の区間ごとの exp の合計 [CandidateParts] float32
	candidateResult     cuda.DevicePtr // 2 段目の結果 [CandidateResultFloats] × 4 バイト
}

// Model は GPU 上の Gemma 4 です。同時に 1 系列だけを扱います。
type Model struct {
	kvCache      string
	keyBytes     int
	valueBytes   int
	mtp          *mtpState
	config       *gemma4.Config
	maxContext   int
	prefillRows  int
	restoreSlack int
	position     int // 次に処理するトークンの位置（ホスト側の写し）
	// savedPosition は SaveState で保存した位置です（なければ noSavedState）。
	savedPosition int
	useGraph      bool // true なら CUDA Graph を再生し、false ならカーネルを 1 個ずつ起動する

	context *cuda.Context
	stream  *cuda.Stream
	module  *cuda.Module
	kernels kernelSet
	memory  *gpucore.Memory

	embedding  gpucore.Matrix // [vocab, hidden]。埋め込みの参照と語彙出力の両方に使う
	outputNorm cuda.DevicePtr
	layers     []deviceLayer
	buffers    activations

	hostState      *cuda.HostBuffer // 状態の受け渡し用の固定メモリ（int32 × 4）
	hostLogits     *cuda.HostBuffer // logits の受け取り用の固定メモリ（float32 × vocab）
	hostSeen       *cuda.HostBuffer // ペナルティの対象の印を送る固定メモリ（uint8 × vocab）
	hostCandidates *cuda.HostBuffer // 候補の受け取り用
	graph          *cuda.GraphExec

	// 入力処理（prefill.go）のカーネル・作業領域と、トークン番号を送る固定メモリ（int32 × 作業領域の行数）です。
	prefillKernels    prefillKernelSet
	prefill           prefillBuffers
	hostPrefillTokens *cuda.HostBuffer
}

// validateConfig は、GPU のカーネルが前提とする大きさかを確かめます。
func validateConfig(config *gemma4.Config) error {
	switch {
	case config.HiddenSize%threadsPerBlock != 0:
		return fmt.Errorf("gemma4gpu: hidden_size %d is not a multiple of %d", config.HiddenSize, threadsPerBlock)
	case config.HiddenSize > normThreads*normMaxPerThread:
		return fmt.Errorf("gemma4gpu: hidden_size %d is larger than %d", config.HiddenSize, normThreads*normMaxPerThread)
	case config.FeedForwardSize%threadsPerBlock != 0:
		return fmt.Errorf("gemma4gpu: feed_forward_size %d is not a multiple of %d", config.FeedForwardSize, threadsPerBlock)
	case config.VocabSize > gpucore.CandidateParts*candidatePartSpan:
		return fmt.Errorf("gemma4gpu: vocab_size %d is larger than %d", config.VocabSize, gpucore.CandidateParts*candidatePartSpan)
	}
	for index, layer := range config.Layers {
		wantHeadDim, groupSize := globalHeadDim, globalGroupSize
		if layer.Sliding {
			wantHeadDim, groupSize = localHeadDim, localGroupSize
		}
		headsPerKV := config.NumHeads / layer.KVHeads
		switch {
		case layer.HeadDim != wantHeadDim:
			return fmt.Errorf("gemma4gpu: layer %d has head_dim %d (the kernel supports %d)", index, layer.HeadDim, wantHeadDim)
		case config.NumHeads%groupSize != 0 || headsPerKV%groupSize != 0:
			return fmt.Errorf("gemma4gpu: layer %d has %d heads and %d kv heads, which do not fit groups of %d",
				index, config.NumHeads, layer.KVHeads, groupSize)
		}
	}
	return nil
}

// Open は .pq4m を読み、重みを GPU に置き、1 トークン分の処理を CUDA Graph に記録します。
// 例: Open("./data/gemma4-12b-qat.pq4m", Options{DeviceUUID: "GPU-...", MaxContext: 8192})
func Open(path string, opts Options) (model *Model, err error) {
	if opts.MaxContext <= 0 {
		return nil, fmt.Errorf("gemma4gpu: MaxContext must be positive, got %d", opts.MaxContext)
	}
	if opts.PrefillChunk == 0 {
		opts.PrefillChunk = PrefillChunkTokens
	}
	if opts.PrefillChunk < 1 || opts.PrefillChunk > localRingSlack {
		return nil, fmt.Errorf("gemma4gpu: prefill chunk must be between 1 and %d, got %d", localRingSlack, opts.PrefillChunk)
	}
	storage, err := resolveKVStorage(opts.KVCache)
	if err != nil {
		return nil, err
	}
	weights, err := gemma4.OpenWeights(path)
	if err != nil {
		return nil, err
	}
	defer weights.Close()
	if err := validateConfig(weights.Config); err != nil {
		return nil, err
	}
	if err := cuda.Init(); err != nil {
		return nil, err
	}
	device, err := cuda.FindDeviceByUUID(opts.DeviceUUID)
	if err != nil {
		return nil, err
	}
	m := &Model{kvCache: storage.name, keyBytes: storage.keyBytes, valueBytes: storage.valueBytes, config: weights.Config, maxContext: opts.MaxContext, prefillRows: opts.PrefillChunk, useGraph: true, savedPosition: noSavedState}
	m.restoreSlack = localRingSlack
	if opts.CompactVRAM {
		m.restoreSlack = max(m.prefillRows, mtpVerifyTokens)
	}
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
	if m.module, err = cuda.LoadModule(storage.cubin); err != nil {
		return nil, err
	}
	if m.kernels, err = loadKernels(m.module); err != nil {
		return nil, err
	}
	if m.memory, err = gpucore.NewMemory(m.stream); err != nil {
		return nil, err
	}
	allocate := func() error {
		if err := m.uploadWeights(weights); err != nil {
			return err
		}
		if err := m.allocateActivations(); err != nil {
			return err
		}
		return m.allocatePrefill()
	}
	if opts.CompactVRAM {
		err = m.memory.Packed(allocate)
	} else {
		err = allocate()
	}
	if err != nil {
		return nil, err
	}
	if err = m.preparePrefill(); err != nil {
		return nil, err
	}
	hostBuffers := []struct {
		target **cuda.HostBuffer
		bytes  int
	}{
		{&m.hostPrefillTokens, int32Bytes * max(m.prefillRows, mtpVerifyTokens)},
		{&m.hostState, int32Bytes * stateCount},
		{&m.hostLogits, float32Bytes * m.config.VocabSize},
		{&m.hostSeen, m.config.VocabSize},
		{&m.hostCandidates, float32Bytes * gpucore.CandidateResultFloats},
	}
	for _, entry := range hostBuffers {
		if *entry.target, err = cuda.AllocHost(entry.bytes); err != nil {
			return nil, err
		}
	}
	if err = m.recordGraph(); err != nil {
		return nil, err
	}
	if opts.MTPPath != "" {
		if m.mtp, err = m.loadAssistant(opts.MTPPath); err != nil {
			return nil, err
		}
		m.mtp.greedyOnly = opts.MTPGreedyOnly
		if err = m.ConfigureMTP(opts.MTPDraftTokens, opts.MTPAdaptive); err != nil {
			return nil, err
		}
	}
	return m, nil
}

// uploadWeights は、埋め込み表（語彙出力と共有）・最終正規化・各層を GPU に置きます。
func (m *Model) uploadWeights(weights *gemma4.Weights) error {
	var err error
	if m.embedding, err = gpucore.UploadMatrices(m.memory, "token_embd", weights.Embedding); err != nil {
		return err
	}
	if m.outputNorm, err = gpucore.Upload(m.memory, weights.OutputNorm); err != nil {
		return err
	}
	m.layers = make([]deviceLayer, len(weights.Layers))
	for index := range weights.Layers {
		if m.layers[index], err = m.uploadLayer(index, &weights.Layers[index]); err != nil {
			return fmt.Errorf("gemma4gpu: layer %d: %w", index, err)
		}
	}
	return nil
}

// layerUploader は 1 層分の転送で、最初に起きたエラーだけを覚えておく道具です。
type layerUploader struct {
	memory *gpucore.Memory
	err    error
}

// vector は float32 の配列を GPU に置きます。
func (u *layerUploader) vector(values []float32) cuda.DevicePtr {
	if u.err != nil {
		return 0
	}
	pointer, err := gpucore.Upload(u.memory, values)
	u.err = err
	return pointer
}

// matrices は PQ4 行列（複数ならつないだもの）を GPU に置きます。
func (u *layerUploader) matrices(name string, parts ...*pq4.Matrix) gpucore.Matrix {
	if u.err != nil {
		return gpucore.Matrix{}
	}
	matrix, err := gpucore.UploadMatrices(u.memory, name, parts...)
	u.err = err
	return matrix
}

// zeroed は size バイトの GPU メモリを確保し、0 で埋めます。
func (u *layerUploader) zeroed(size int) cuda.DevicePtr {
	if u.err != nil {
		return 0
	}
	pointer, err := u.memory.AllocZeroed(size)
	u.err = err
	return pointer
}

// uploadLayer は、1 層分の重みを GPU に置き、KV キャッシュを確保します。
func (m *Model) uploadLayer(index int, weights *gemma4.LayerWeights) (deviceLayer, error) {
	config := m.config
	u := &layerUploader{memory: m.memory}
	queryWidth := config.QueryWidth(index)
	kvWidth := config.KVWidth(index)
	out := deviceLayer{
		config:       weights.Config,
		attnNorm:     u.vector(weights.AttnNorm),
		postAttnNorm: u.vector(weights.PostAttnNorm),
		ffnNorm:      u.vector(weights.FFNNorm),
		postFFNNorm:  u.vector(weights.PostFFNNorm),
		qNorm:        u.vector(weights.QNorm),
		kNorm:        u.vector(weights.KNorm),
		ropeInverse:  u.vector(weights.RopeInverse),
		outScale:     weights.OutScale,
		o:            u.matrices("attn_output", weights.O),
		gateUp:       u.matrices("ffn_gate_up", weights.Gate, weights.Up),
		down:         u.matrices("ffn_down", weights.Down),
	}
	if weights.V != nil {
		out.qkv = u.matrices("attn_qkv", weights.Q, weights.K, weights.V)
		out.valueOffset = queryWidth + kvWidth
		out.qkvWidth = queryWidth + keyAndValueVectors*kvWidth
	} else {
		out.qkv = u.matrices("attn_qk", weights.Q, weights.K)
		out.valueOffset = queryWidth // v は k の出力（正規化の前）を使う
		out.qkvWidth = queryWidth + kvWidth
	}
	if weights.Config.Sliding {
		out.window = config.SlidingWindow
		out.ringSize = config.SlidingWindow + m.restoreSlack
		out.groupSize = localGroupSize
		out.maxSplits = gpucore.CeilDiv(config.SlidingWindow, minPositionsPerCut)
	} else {
		out.ringSize = m.maxContext
		out.groupSize = globalGroupSize
		out.maxSplits = globalMaxSplits
	}
	out.keyCache = u.zeroed(m.keyBytes * out.ringSize * kvWidth)
	out.valueCache = u.zeroed(m.valueBytes * out.ringSize * kvWidth)
	return out, u.err
}

// allocateActivations は作業領域を確保します。層によって大きさが違うものは、最大の大きさにします。
func (m *Model) allocateActivations() error {
	config := m.config
	hidden := config.HiddenSize
	maxInput := max(hidden, config.FeedForwardSize)
	maxQKV, maxQuery, maxPartial := 0, 0, 0
	for index := range m.layers {
		layer := &m.layers[index]
		maxQKV = max(maxQKV, layer.qkvWidth)
		maxQuery = max(maxQuery, layer.queryWidth(config.NumHeads))
		maxPartial = max(maxPartial, layer.partialFloats(config.NumHeads))
	}
	maxInput = max(maxInput, maxQuery)
	b := &m.buffers
	sizes := []struct {
		target *cuda.DevicePtr
		bytes  int
	}{
		{&b.state, int32Bytes * stateCount},
		{&b.hidden, float32Bytes * hidden},
		{&b.delta, float32Bytes * hidden},
		{&b.actQ, maxInput},
		{&b.actInfo, gpucore.ActivationInfoBytes * (maxInput / pq4.BlockSize)},
		{&b.qkv, float32Bytes * maxQKV},
		{&b.query, float32Bytes * maxQuery},
		{&b.partial, float32Bytes * maxPartial},
		{&b.gateUp, float32Bytes * 2 * config.FeedForwardSize},
		{&b.logits, float32Bytes * config.VocabSize},
		{&b.partValue, float32Bytes * argmaxParts},
		{&b.partIndex, int32Bytes * argmaxParts},
		{&b.seenFlags, config.VocabSize},
		{&b.tokenCounts, int32Bytes * config.VocabSize},
		{&b.candidatePartValues, float32Bytes * gpucore.CandidateParts * gpucore.CandidateCount},
		{&b.candidatePartIDs, int32Bytes * gpucore.CandidateParts * gpucore.CandidateCount},
		{&b.candidatePartMax, float32Bytes * gpucore.CandidateParts},
		{&b.candidatePartSum, float32Bytes * gpucore.CandidateParts},
		{&b.candidateResult, float32Bytes * gpucore.CandidateResultFloats},
	}
	for _, entry := range sizes {
		pointer, err := m.memory.AllocZeroed(entry.bytes)
		if err != nil {
			return err
		}
		*entry.target = pointer
	}
	return nil
}

// recordGraph は 1 トークン分の処理を CUDA Graph に記録し、実行可能な形にします。
func (m *Model) recordGraph() error {
	if err := m.stream.BeginCapture(); err != nil {
		return err
	}
	enqueueError := m.enqueueToken(&gpucore.Launcher{Stream: m.stream})
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

// Config はモデルの構成を返します。
func (m *Model) Config() *gemma4.Config { return m.config }

// VocabSize は語彙の大きさを返します。例: 262144
func (m *Model) VocabSize() int { return m.config.VocabSize }

// Position は、次に処理するトークンの位置を返します。
func (m *Model) Position() int { return m.position }

// MaxContext は、扱える最大の位置数（Options.MaxContext）を返します。
func (m *Model) MaxContext() int { return m.maxContext }

// AllocatedBytes は、確保した GPU メモリの合計（重み・KV キャッシュ・作業領域）を返します。
func (m *Model) AllocatedBytes() int64 { return m.memory.TotalBytes() }

// SetUseGraph は、CUDA Graph を再生するか（true）、カーネルを 1 個ずつ起動するか（false）を選びます（速度の比較用）。
func (m *Model) SetUseGraph(use bool) { m.useGraph = use }

// Reset は位置を 0 に戻します。KV キャッシュは、見える範囲より外の要素を読まないため消しません。
// SaveState で保存した状態は無効になります。
func (m *Model) Reset() error {
	m.invalidateMTP()
	if m.mtp != nil {
		m.mtp.highWater = 0
		m.mtp.policy = [mtpContextBuckets]mtpPolicy{}
	}
	m.savedPosition = noSavedState
	m.position = 0
	return nil
}

// checkInput は、入力トークンと位置が扱える範囲かを確かめます。
func (m *Model) checkInput(token int) error {
	if token < 0 || token >= m.config.VocabSize {
		return fmt.Errorf("gemma4gpu: token %d is outside the vocabulary (size %d)", token, m.config.VocabSize)
	}
	if m.position >= m.maxContext {
		return fmt.Errorf("gemma4gpu: position %d reached MaxContext %d", m.position, m.maxContext)
	}
	return nil
}

// runToken は 1 トークン分の処理をストリームに積みます（Graph の再生、またはカーネルの個別起動）。
func (m *Model) runToken() error {
	if m.useGraph {
		return m.graph.Launch(m.stream)
	}
	return m.enqueueToken(&gpucore.Launcher{Stream: m.stream})
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
func (m *Model) Forward(token int) ([]float32, error) {
	if m.selectionMTPEnabled() {
		if _, err := m.Feed(token); err != nil {
			return nil, err
		}
		return m.CopyLogits()
	}
	m.invalidateMTP()
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
// 戻り値は Forward と同じく固定メモリを指し、次の呼び出しで上書きされます。
func (m *Model) CopyLogits() ([]float32, error) {
	if m.position == 0 {
		return nil, errors.New("gemma4gpu: CopyLogits needs a processed token")
	}
	logits := m.buffers.logits
	if m.mtp != nil && m.mtp.logitsRow >= 0 {
		logits = m.mtp.logits + cuda.DevicePtr(m.mtp.logitsRow*float32Bytes*m.config.VocabSize)
	}
	if err := m.hostLogits.CopyFromDeviceAsync(m.stream, logits); err != nil {
		return nil, err
	}
	if err := m.stream.Synchronize(); err != nil {
		return nil, err
	}
	return m.hostLogits.Float32s(), nil
}

// Feed は token を 1 個処理して位置を 1 進め、logits の argmax（貪欲法での次のトークン）を返します。
func (m *Model) Feed(token int) (int, error) {
	if err := m.checkInput(token); err != nil {
		return 0, err
	}
	if m.mtp != nil && m.mtp.enabled {
		return m.feedMTP(token)
	}
	return m.feedSingle(token)
}

// feedSingle は先読みなしの通常生成を実行します。MTP の初回入力と比較にも使います。
func (m *Model) feedSingle(token int) (int, error) {
	if err := m.checkInput(token); err != nil {
		return 0, err
	}
	if err := m.writeInput(token); err != nil {
		return 0, err
	}
	return m.finishGreedyStep()
}

// ContinueGreedy は、直前の argmax を入力として 1 トークン処理し、次の argmax を返します。
// 状態は GPU 側で更新済みなので、ホストからは何も書きません。直前に Feed または ContinueGreedy を呼んでいる必要があります。
func (m *Model) ContinueGreedy() (int, error) {
	if m.mtp != nil && m.mtp.enabled && m.mtp.seedValid {
		return m.Feed(m.mtp.expected)
	}
	if m.position == 0 {
		return 0, errors.New("gemma4gpu: ContinueGreedy needs a previous Feed")
	}
	if m.position >= m.maxContext {
		return 0, fmt.Errorf("gemma4gpu: position %d reached MaxContext %d", m.position, m.maxContext)
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
		return 0, fmt.Errorf("gemma4gpu: GPU position %d differs from host position %d", state[statePosition], m.position)
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
	if m.mtp != nil {
		keep(m.mtp.close())
		m.mtp = nil
	}
	if m.graph != nil {
		keep(m.graph.Close())
		m.graph = nil
	}
	if m.memory != nil {
		keep(m.memory.Close())
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
