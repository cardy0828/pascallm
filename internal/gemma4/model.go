package gemma4

import (
	"fmt"
	"math"

	"github.com/cardy0828/pascallm/internal/linear"
	"github.com/cardy0828/pascallm/internal/numeric"
	"github.com/cardy0828/pascallm/internal/pq4"
)

// ActivationMode は、行列とベクトルの積で活性値をどう扱うかです。
type ActivationMode int

const (
	// ActivationsExact は、活性値を FP32 のまま使います（llama.cpp が Q4_0 で行う計算と同じ意味）。
	// llama.cpp との突き合わせに使います。
	ActivationsExact ActivationMode = iota
	// ActivationsInt8 は、活性値を 32 要素ごとに int8 にしてから計算します（GPU の DP4A カーネルと同じ手順）。
	// GPU 実装との突き合わせに使います。
	ActivationsInt8
)

// GELU（tanh による近似）の定数です。llama.cpp（ggml）と同じ値です。
const (
	geluCoefficient = 0.044715
	sqrtTwoOverPi   = 0.79788456080286535587989211986876
	geluHalf        = 0.5
)

// noWindowLimit は、firstVisiblePosition で窓のない層（全文脈）を表す値です。
const noWindowLimit = 0

// Model は、重みと、1 トークンずつ進める計算状態（KV キャッシュ）を持ちます。同時に 1 系列だけを扱います。
type Model struct {
	weights  *Weights
	config   *Config
	mode     ActivationMode
	position int // 次に処理するトークンの位置（0 始まり）

	// HalfPrecisionCache が true なら、k・v を FP16 に丸めてから KV キャッシュに入れます（GPU 実装と llama.cpp の
	// 既定と同じ）。false（既定）なら FP32 のままです。FP16 の丸めによる差を調べるために使います。
	HalfPrecisionCache bool

	// Trace が nil でなければ、Forward が埋め込みの後（layers = 0）と各層の後（layers = 1, 2, …）に、
	// 残差の流れ hidden を渡して呼びます（GPU 実装との差を層ごとに調べるため）。hidden は次の層で上書きされます。
	Trace func(layers int, hidden []float32)

	keyCache   [][][]float32 // [layer][position][kv_heads × head_dim]（正規化と RoPE の後）
	valueCache [][][]float32 // [layer][position][kv_heads × head_dim]（重みなしの正規化の後）

	buffers workBuffers
}

// workBuffers は、1 トークンの計算で使い回す作業領域です。層によって大きさが違うものは、最大の大きさで確保します。
type workBuffers struct {
	hidden    []float32 // 残差の流れ [hidden]
	normed    []float32 // 正規化した入力 [hidden]
	delta     []float32 // attention・FFN の出力 [hidden]
	query     []float32 // [num_heads × head_dim]
	key       []float32 // [kv_heads × head_dim]
	value     []float32 // [kv_heads × head_dim]
	attention []float32 // ヘッドごとの出力を並べたもの [num_heads × head_dim]
	scores    []float32 // 1 ヘッド分の attention の重み（位置の数だけ伸びる）
	ffnGate   []float32 // Wgate·y [ffn]
	ffnUp     []float32 // Wup·y [ffn]
	logits    []float32 // [vocab]
	ropeCos   []float32 // [head_dim / 2]
	ropeSin   []float32
}

// NewModel は、重みから計算状態を作ります。
// 例: NewModel(weights, ActivationsExact)
func NewModel(weights *Weights, mode ActivationMode) *Model {
	config := weights.Config
	maxQuery, maxKV, maxHeadDim := 0, 0, 0
	for index, layer := range config.Layers {
		maxQuery = max(maxQuery, config.QueryWidth(index))
		maxKV = max(maxKV, config.KVWidth(index))
		maxHeadDim = max(maxHeadDim, layer.HeadDim)
	}
	model := &Model{
		weights:    weights,
		config:     config,
		mode:       mode,
		keyCache:   make([][][]float32, config.NumLayers()),
		valueCache: make([][][]float32, config.NumLayers()),
		buffers: workBuffers{
			hidden:    make([]float32, config.HiddenSize),
			normed:    make([]float32, config.HiddenSize),
			delta:     make([]float32, config.HiddenSize),
			query:     make([]float32, maxQuery),
			key:       make([]float32, maxKV),
			value:     make([]float32, maxKV),
			attention: make([]float32, maxQuery),
			ffnGate:   make([]float32, config.FeedForwardSize),
			ffnUp:     make([]float32, config.FeedForwardSize),
			logits:    make([]float32, config.VocabSize),
			ropeCos:   make([]float32, maxHeadDim/2),
			ropeSin:   make([]float32, maxHeadDim/2),
		},
	}
	return model
}

// Reset は KV キャッシュを空にし、位置を 0 に戻します。
func (m *Model) Reset() {
	m.position = 0
	for layer := range m.keyCache {
		m.keyCache[layer] = nil
		m.valueCache[layer] = nil
	}
}

// Position は、次に処理するトークンの位置を返します。
func (m *Model) Position() int { return m.position }

// mulVec は dst = matrix·x を、設定した活性値の扱いで計算します。
func (m *Model) mulVec(matrix *pq4.Matrix, dst, x []float32) {
	if m.mode == ActivationsInt8 {
		matrix.MulVec(dst, x)
		return
	}
	matrix.MulVecExact(dst, x)
}

// Forward は token を 1 個処理して位置を 1 進め、その位置の logits（長さ vocab）を返します。
// 戻り値のスライスは次の Forward で上書きされるため、残す場合はコピーしてください。
func (m *Model) Forward(token int) ([]float32, error) {
	if token < 0 || token >= m.config.VocabSize {
		return nil, fmt.Errorf("gemma4: token %d is outside the vocabulary (size %d)", token, m.config.VocabSize)
	}
	b := &m.buffers
	eps := float32(m.config.NormEps)
	m.weights.Embedding.DequantizeRow(token, b.hidden)
	scaleInPlace(b.hidden, m.config.EmbeddingScale())
	m.trace(0)

	for layer := range m.weights.Layers {
		weights := &m.weights.Layers[layer]
		rmsNorm(b.normed, b.hidden, weights.AttnNorm, eps)
		m.attention(layer, weights)
		rmsNorm(b.delta, b.delta, weights.PostAttnNorm, eps)
		addInPlace(b.hidden, b.delta)

		rmsNorm(b.normed, b.hidden, weights.FFNNorm, eps)
		m.feedForward(weights)
		rmsNorm(b.delta, b.delta, weights.PostFFNNorm, eps)
		addInPlace(b.hidden, b.delta)
		scaleInPlace(b.hidden, weights.OutScale)
		m.trace(layer + 1)
	}

	rmsNorm(b.normed, b.hidden, m.weights.OutputNorm, eps)
	m.mulVec(m.weights.Embedding, b.logits, b.normed)
	softcapInPlace(b.logits, float32(m.config.FinalLogitSoftcap))
	m.position++
	return b.logits, nil
}

// trace は、Trace が設定されていれば、layers 層を終えた時点の hidden を渡して呼びます。
func (m *Model) trace(layers int) {
	if m.Trace != nil {
		m.Trace(layers, m.buffers.hidden)
	}
}

// attention は、層 layer の attention（q・k・v の計算から o の射影まで）を計算し、結果を buffers.delta に書きます。
// 入力は buffers.normed です。スコアには 1/sqrt(head_dim) を掛けません（q と k を正規化しているため。llama.cpp と同じ）。
func (m *Model) attention(layer int, weights *LayerWeights) {
	b := &m.buffers
	config := weights.Config
	headDim := config.HeadDim
	eps := float32(m.config.NormEps)
	query := b.query[:m.config.QueryWidth(layer)]
	key := b.key[:m.config.KVWidth(layer)]
	value := b.value[:m.config.KVWidth(layer)]

	m.mulVec(weights.Q, query, b.normed)
	m.mulVec(weights.K, key, b.normed)
	if weights.V != nil {
		m.mulVec(weights.V, value, b.normed)
	} else {
		copy(value, key) // 全文脈の層: 正規化の前の k を v にも使う
	}

	m.ropeAngles(weights.RopeInverse)
	cosine := b.ropeCos[:headDim/2]
	sine := b.ropeSin[:headDim/2]
	for head := 0; head < m.config.NumHeads; head++ {
		vector := query[head*headDim : (head+1)*headDim]
		rmsNorm(vector, vector, weights.QNorm, eps)
		applyRope(vector, cosine, sine)
	}
	for head := 0; head < config.KVHeads; head++ {
		vector := key[head*headDim : (head+1)*headDim]
		rmsNorm(vector, vector, weights.KNorm, eps)
		applyRope(vector, cosine, sine)
		rmsNormPlain(value[head*headDim:(head+1)*headDim], eps)
	}
	if m.HalfPrecisionCache {
		roundToHalf(key)
		roundToHalf(value)
	}
	m.keyCache[layer] = append(m.keyCache[layer], append([]float32(nil), key...))
	m.valueCache[layer] = append(m.valueCache[layer], append([]float32(nil), value...))

	window := noWindowLimit
	if config.Sliding {
		window = m.config.SlidingWindow
	}
	first := firstVisiblePosition(m.position, window)
	visible := m.position - first + 1
	if cap(b.scores) < visible {
		b.scores = make([]float32, visible, 2*visible)
	}
	scores := b.scores[:visible]
	groupSize := m.config.NumHeads / config.KVHeads
	output := b.attention[:len(query)]

	for head := 0; head < m.config.NumHeads; head++ {
		kvHead := head / groupSize
		headQuery := query[head*headDim : (head+1)*headDim]
		for offset := range scores {
			cached := m.keyCache[layer][first+offset][kvHead*headDim : (kvHead+1)*headDim]
			scores[offset] = linear.Dot(headQuery, cached)
		}
		softmaxInPlace(scores)
		headOutput := output[head*headDim : (head+1)*headDim]
		clear(headOutput)
		for offset, weight := range scores {
			cached := m.valueCache[layer][first+offset][kvHead*headDim : (kvHead+1)*headDim]
			for dim := range headOutput {
				headOutput[dim] += weight * cached[dim]
			}
		}
	}
	m.mulVec(weights.O, b.delta, output)
}

// firstVisiblePosition は、位置 position から見える最も古い位置を返します。
// 窓のある層では「position − 位置 < window」の位置だけが見えます（llama.cpp の is_masked_swa と同じ）。
// 例: window 1024、position 1500 -> 477（477〜1500 の 1024 個）。window が 0 なら 0
func firstVisiblePosition(position, window int) int {
	if window == noWindowLimit {
		return 0
	}
	return max(0, position-window+1)
}

// ropeAngles は、現在の位置での RoPE の cos・sin を buffers に用意します（角度は float32 の「位置 × 逆周波数」）。
func (m *Model) ropeAngles(inverse []float32) {
	position := float32(m.position)
	for index, frequency := range inverse {
		angle := position * frequency
		m.buffers.ropeCos[index] = float32(math.Cos(float64(angle)))
		m.buffers.ropeSin[index] = float32(math.Sin(float64(angle)))
	}
}

// feedForward は buffers.delta = Wdown·(gelu(Wgate·normed) × (Wup·normed)) を計算します。
func (m *Model) feedForward(weights *LayerWeights) {
	b := &m.buffers
	m.mulVec(weights.Gate, b.ffnGate, b.normed)
	m.mulVec(weights.Up, b.ffnUp, b.normed)
	for index := range b.ffnGate {
		b.ffnGate[index] = gelu(b.ffnGate[index]) * b.ffnUp[index]
	}
	m.mulVec(weights.Down, b.delta, b.ffnGate)
}

// rmsNorm は dst = src / sqrt(mean(src²) + eps) × weight を計算します。dst と src は同じでもかまいません。
// 例: src = [3, 4]、weight = [1, 1]、eps = 0 -> [0.8485, 1.1314]
func rmsNorm(dst, src, weight []float32, eps float32) {
	inverseRMS := inverseRootMeanSquare(src, eps)
	for index, value := range src {
		dst[index] = weight[index] * (value * inverseRMS)
	}
}

// rmsNormPlain は、重みを掛けない RMSNorm を values にその場で適用します（attention の v 用）。
func rmsNormPlain(values []float32, eps float32) {
	inverseRMS := inverseRootMeanSquare(values, eps)
	for index := range values {
		values[index] *= inverseRMS
	}
}

// inverseRootMeanSquare は 1 / sqrt(mean(values²) + eps) を返します。二乗和は float64 で足します。
func inverseRootMeanSquare(values []float32, eps float32) float32 {
	var sumSquares float64
	for _, value := range values {
		sumSquares += float64(value) * float64(value)
	}
	mean := float32(sumSquares / float64(len(values)))
	return float32(1 / math.Sqrt(float64(mean+eps)))
}

// roundToHalf は、values の各要素を FP16 で表せる最も近い値に丸めます。例: 0.1 -> 0.0999755859375
func roundToHalf(values []float32) {
	for index, value := range values {
		values[index] = numeric.Float16ToFloat32(numeric.Float32ToFloat16(value))
	}
}

// addInPlace は dst[i] += src[i] を計算します（残差接続）。
func addInPlace(dst, src []float32) {
	for index, value := range src {
		dst[index] += value
	}
}

// scaleInPlace は values[i] *= factor を計算します。
func scaleInPlace(values []float32, factor float32) {
	for index := range values {
		values[index] *= factor
	}
}

// softmaxInPlace は values を softmax に置き換えます。最大値を引いてから指数を取り、桁あふれを防ぎます。
func softmaxInPlace(values []float32) {
	maximum := float32(math.Inf(-1))
	for _, value := range values {
		maximum = max(maximum, value)
	}
	var sum float64
	for index, value := range values {
		exponent := float32(math.Exp(float64(value - maximum)))
		values[index] = exponent
		sum += float64(exponent)
	}
	inverse := float32(1 / sum)
	for index := range values {
		values[index] *= inverse
	}
}

// gelu は GELU の tanh による近似です（llama.cpp と同じ式）。
//
//	0.5 × x × (1 + tanh(sqrt(2/π) × x × (1 + 0.044715 × x²)))
//
// 例: gelu(0) = 0、gelu(1) ≒ 0.8412
func gelu(x float32) float32 {
	inner := sqrtTwoOverPi * float64(x) * (1 + geluCoefficient*float64(x)*float64(x))
	return float32(geluHalf * float64(x) * (1 + math.Tanh(inner)))
}

// softcapInPlace は values[i] = limit × tanh(values[i] / limit) を計算します。limit が 0 なら何もしません。
// 値の大小の順は変わりません（argmax は同じ）。例: limit 30、値 60 -> 28.92
func softcapInPlace(values []float32, limit float32) {
	if limit == 0 {
		return
	}
	for index, value := range values {
		values[index] = limit * float32(math.Tanh(float64(value/limit)))
	}
}

// applyRope は、1 ヘッド分のベクトル（長さ head_dim）に RoPE を掛けます。
// 次元 i と i + head_dim / 2 を組にして回転します。cos・sin は長さ head_dim / 2 です。
func applyRope(vector, cos, sin []float32) {
	half := len(cos)
	for index := 0; index < half; index++ {
		first := vector[index]
		second := vector[index+half]
		vector[index] = first*cos[index] - second*sin[index]
		vector[index+half] = second*cos[index] + first*sin[index]
	}
}
