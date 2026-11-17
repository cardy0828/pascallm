package lfm3

import (
	"fmt"
	"math"
)

// Model は、重みと、1 トークンずつ進める計算状態（KV キャッシュ・畳み込みの履歴）を持ちます。
// 同時に 1 系列だけを扱います。別の系列を始めるときは Reset を呼びます。
type Model struct {
	weights  *Weights
	config   *Config
	position int // 次に処理するトークンの位置（0 始まり）

	// ropeInverseFrequencies[kind] は、層の種類ごとの RoPE の逆周波数（長さ head_dim/2）です。
	ropeInverseFrequencies map[LayerKind][]float32

	// 層ごとの状態。種類に合わない層の要素は nil です。
	convHistory [][]float32   // [layer][hidden × (kernel_size-1)]。チャネル c の古い順の入力
	keyCache    [][][]float32 // [layer][position][kv_heads × head_dim]
	valueCache  [][][]float32 // [layer][position][kv_heads × head_dim]

	buffers workBuffers
}

// workBuffers は、1 トークンの計算で使い回す作業領域です（毎回の確保を避けるため）。
type workBuffers struct {
	hidden    []float32 // 残差の流れ（x）
	normed    []float32 // 正規化した入力
	operator  []float32 // conv または attention の出力
	convIn    []float32 // in_proj の出力 [3 × hidden]
	convMixed []float32 // 畳み込み後に C を掛けた値 [hidden]
	query     []float32 // [num_heads × head_dim]
	key       []float32 // [kv_heads × head_dim]
	value     []float32 // [kv_heads × head_dim]
	gate      []float32 // [num_heads]
	attention []float32 // ヘッドごとの出力を並べたもの [num_heads × head_dim]
	scores    []float32 // 1 ヘッド分の attention の重み（位置の数だけ伸びる）
	ffnGate   []float32 // W1·x [intermediate]
	ffnUp     []float32 // W3·x [intermediate]
	ffnOut    []float32 // W2 の出力 [hidden]
	logits    []float32 // [vocab]
	ropeCos   []float32 // [head_dim/2]
	ropeSin   []float32 // [head_dim/2]
}

// NewModel は、重みから計算状態を作ります。
func NewModel(weights *Weights) *Model {
	config := weights.Config
	model := &Model{
		weights:                weights,
		config:                 config,
		ropeInverseFrequencies: map[LayerKind][]float32{},
		convHistory:            make([][]float32, config.NumLayers),
		keyCache:               make([][][]float32, config.NumLayers),
		valueCache:             make([][][]float32, config.NumLayers),
	}
	for _, kind := range []LayerKind{LayerFullAttention, LayerSlidingAttention} {
		if rope, ok := config.RopeParameters[kind.String()]; ok {
			model.ropeInverseFrequencies[kind] = ropeInverseFrequencies(rope, config.HeadDim)
		}
	}
	hidden := config.HiddenSize
	model.buffers = workBuffers{
		hidden:    make([]float32, hidden),
		normed:    make([]float32, hidden),
		operator:  make([]float32, hidden),
		convIn:    make([]float32, 3*hidden),
		convMixed: make([]float32, hidden),
		query:     make([]float32, config.NumHeads*config.HeadDim),
		key:       make([]float32, config.NumKVHeads*config.HeadDim),
		value:     make([]float32, config.NumKVHeads*config.HeadDim),
		gate:      make([]float32, config.NumHeads),
		attention: make([]float32, config.NumHeads*config.HeadDim),
		ffnGate:   make([]float32, config.IntermediateSize),
		ffnUp:     make([]float32, config.IntermediateSize),
		ffnOut:    make([]float32, hidden),
		logits:    make([]float32, config.VocabSize),
		ropeCos:   make([]float32, config.HeadDim/2),
		ropeSin:   make([]float32, config.HeadDim/2),
	}
	model.Reset()
	return model
}

// Reset は、KV キャッシュと畳み込みの履歴を空にし、位置を 0 に戻します。
func (m *Model) Reset() {
	m.position = 0
	historyLength := m.config.HiddenSize * (m.config.ConvKernelSize - 1)
	for layer, weights := range m.weights.Layers {
		if weights.Kind == LayerConv {
			m.convHistory[layer] = make([]float32, historyLength)
		} else {
			m.keyCache[layer] = nil
			m.valueCache[layer] = nil
		}
	}
}

// Position は、次に処理するトークンの位置を返します。
func (m *Model) Position() int { return m.position }

// Forward は token を 1 個処理して位置を 1 進め、その位置の logits（長さ vocab）を返します。
// 戻り値のスライスは次の Forward で上書きされるため、残す場合はコピーしてください。
func (m *Model) Forward(token int) ([]float32, error) {
	if token < 0 || token >= m.config.VocabSize {
		return nil, fmt.Errorf("lfm3: token %d is outside the vocabulary (size %d)", token, m.config.VocabSize)
	}
	b := &m.buffers
	eps := float32(m.config.NormEps)
	m.weights.Embedding.Lookup(token, b.hidden)

	for layer := range m.weights.Layers {
		weights := &m.weights.Layers[layer]
		rmsNorm(b.normed, b.hidden, weights.OperatorNorm, eps)
		if weights.Kind == LayerConv {
			m.shortConv(layer, weights.Conv)
		} else {
			m.attention(layer, weights.Kind, weights.Attention)
		}
		rmsNorm(b.operator, b.operator, weights.PostOperatorNorm, eps)
		addInPlace(b.hidden, b.operator)

		rmsNorm(b.normed, b.hidden, weights.FFNNorm, eps)
		m.feedForward(&weights.FeedForward)
		rmsNorm(b.ffnOut, b.ffnOut, weights.PostFFNNorm, eps)
		addInPlace(b.hidden, b.ffnOut)
	}

	rmsNorm(b.normed, b.hidden, m.weights.FinalNorm, eps)
	m.weights.LMHead.MulVec(b.logits, b.normed)
	m.position++
	return b.logits, nil
}

// shortConv は gated short convolution を計算し、結果を buffers.operator に書きます。
//
//	[B, C, x] = in_proj(normed)、u = B * x
//	conv[c] = Σ_k kernel[c][k] · u[t-(K-1)+k][c]（k = K-1 が現在のトークン）
//	operator = out_proj(C * conv)
func (m *Model) shortConv(layer int, weights *ConvWeights) {
	b := &m.buffers
	hidden := m.config.HiddenSize
	kernelSize := m.config.ConvKernelSize
	historyLength := kernelSize - 1
	history := m.convHistory[layer]

	weights.InProj.MulVec(b.convIn, b.normed)
	gateB := b.convIn[0:hidden]
	gateC := b.convIn[hidden : 2*hidden]
	input := b.convIn[2*hidden : 3*hidden]
	for channel := 0; channel < hidden; channel++ {
		current := gateB[channel] * input[channel]
		taps := weights.Kernel[channel*kernelSize : (channel+1)*kernelSize]
		past := history[channel*historyLength : (channel+1)*historyLength]
		sum := taps[historyLength] * current
		for k := 0; k < historyLength; k++ {
			sum += taps[k] * past[k]
		}
		// 履歴を 1 つ古い方へずらし、最後に現在の値を入れる。
		copy(past, past[1:])
		past[historyLength-1] = current
		b.convMixed[channel] = gateC[channel] * sum
	}
	weights.OutProj.MulVec(b.operator, b.convMixed)
}

// attention は、q・k の正規化、RoPE、因果 attention（窓付き層は直近 SlidingWindow 個）、
// ヘッドごとの sigmoid ゲートを計算し、o_proj の結果を buffers.operator に書きます。
func (m *Model) attention(layer int, kind LayerKind, weights *AttentionWeights) {
	b := &m.buffers
	config := m.config
	headDim := config.HeadDim
	eps := float32(config.NormEps)

	weights.Q.MulVec(b.query, b.normed)
	weights.K.MulVec(b.key, b.normed)
	weights.V.MulVec(b.value, b.normed)
	weights.Gate.MulVec(b.gate, b.normed)

	m.ropeAngles(kind)
	for head := 0; head < config.NumHeads; head++ {
		vector := b.query[head*headDim : (head+1)*headDim]
		rmsNorm(vector, vector, weights.QNorm, eps)
		applyRope(vector, b.ropeCos, b.ropeSin)
	}
	for head := 0; head < config.NumKVHeads; head++ {
		vector := b.key[head*headDim : (head+1)*headDim]
		rmsNorm(vector, vector, weights.KNorm, eps)
		applyRope(vector, b.ropeCos, b.ropeSin)
	}
	m.keyCache[layer] = append(m.keyCache[layer], append([]float32(nil), b.key...))
	m.valueCache[layer] = append(m.valueCache[layer], append([]float32(nil), b.value...))

	first := m.firstVisiblePosition(kind)
	visible := m.position - first + 1
	if cap(b.scores) < visible {
		b.scores = make([]float32, visible, 2*visible)
	}
	scores := b.scores[:visible]
	scale := float32(1 / math.Sqrt(float64(headDim)))
	groupSize := config.NumHeads / config.NumKVHeads

	for head := 0; head < config.NumHeads; head++ {
		kvHead := head / groupSize
		query := b.query[head*headDim : (head+1)*headDim]
		for offset := range scores {
			key := m.keyCache[layer][first+offset][kvHead*headDim : (kvHead+1)*headDim]
			scores[offset] = dotFloat32(query, key) * scale
		}
		softmaxInPlace(scores)
		output := b.attention[head*headDim : (head+1)*headDim]
		clear(output)
		for offset, weight := range scores {
			value := m.valueCache[layer][first+offset][kvHead*headDim : (kvHead+1)*headDim]
			for dim := range output {
				output[dim] += weight * value[dim]
			}
		}
		gate := sigmoid(b.gate[head])
		for dim := range output {
			output[dim] *= gate
		}
	}
	weights.O.MulVec(b.operator, b.attention)
}

// firstVisiblePosition は、現在の位置から見える最も古い位置を返します。
// 窓付き層では「現在の位置 - 位置 < SlidingWindow」を満たす位置だけが見えます（Hugging Face の定義）。
// 例: SlidingWindow = 256、現在の位置 300 -> 45（45〜300 の 256 個）
func (m *Model) firstVisiblePosition(kind LayerKind) int {
	if kind != LayerSlidingAttention {
		return 0
	}
	return max(0, m.position-m.config.SlidingWindow+1)
}

// ropeAngles は、現在の位置での RoPE の cos・sin（長さ head_dim/2）を buffers に用意します。
// PyTorch と同じく、角度は float32 の「位置 × 逆周波数」から計算します。
func (m *Model) ropeAngles(kind LayerKind) {
	inverse := m.ropeInverseFrequencies[kind]
	position := float32(m.position)
	for index, frequency := range inverse {
		angle := position * frequency
		m.buffers.ropeCos[index] = float32(math.Cos(float64(angle)))
		m.buffers.ropeSin[index] = float32(math.Sin(float64(angle)))
	}
}

// feedForward は buffers.ffnOut = W2(silu(W1·normed) * (W3·normed)) を計算します。
func (m *Model) feedForward(weights *FeedForwardWeights) {
	b := &m.buffers
	weights.W1.MulVec(b.ffnGate, b.normed)
	weights.W3.MulVec(b.ffnUp, b.normed)
	for index := range b.ffnGate {
		b.ffnGate[index] = silu(b.ffnGate[index]) * b.ffnUp[index]
	}
	weights.W2.MulVec(b.ffnOut, b.ffnGate)
}
