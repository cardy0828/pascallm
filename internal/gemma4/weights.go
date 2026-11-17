package gemma4

import (
	"fmt"
	"math"

	"github.com/cardy0828/pascallm/internal/pq4"
)

// テンソル名です（GGUF のまま）。%d には層の番号（0 始まり）が入ります。
const (
	NameEmbedding    = "token_embd.weight" // [vocab, hidden]。語彙出力と共有
	nameOutputNorm   = "output_norm.weight"
	nameRopeFreqs    = "rope_freqs.weight" // 全文脈の層の RoPE の周波数を割る係数 [head_dim / 2]
	nameAttnNorm     = "blk.%d.attn_norm.weight"
	namePostAttnNorm = "blk.%d.post_attention_norm.weight"
	nameFFNNorm      = "blk.%d.ffn_norm.weight"
	namePostFFNNorm  = "blk.%d.post_ffw_norm.weight"
	nameAttnQ        = "blk.%d.attn_q.weight"
	nameAttnK        = "blk.%d.attn_k.weight"
	nameAttnV        = "blk.%d.attn_v.weight" // 全文脈の層にはない
	nameAttnO        = "blk.%d.attn_output.weight"
	nameAttnQNorm    = "blk.%d.attn_q_norm.weight"
	nameAttnKNorm    = "blk.%d.attn_k_norm.weight"
	nameFFNGate      = "blk.%d.ffn_gate.weight"
	nameFFNUp        = "blk.%d.ffn_up.weight"
	nameFFNDown      = "blk.%d.ffn_down.weight"
	nameOutScale     = "blk.%d.layer_output_scale.weight"
)

// LayerWeights は 1 層分の重みです。
type LayerWeights struct {
	Config LayerConfig

	AttnNorm     []float32 // [hidden]
	PostAttnNorm []float32
	FFNNorm      []float32
	PostFFNNorm  []float32
	QNorm        []float32 // [head_dim]
	KNorm        []float32

	Q *pq4.Matrix // [num_heads × head_dim, hidden]
	K *pq4.Matrix // [kv_heads × head_dim, hidden]
	V *pq4.Matrix // K と同じ形。nil なら、v に k の出力（正規化の前）を使う
	O *pq4.Matrix // [hidden, num_heads × head_dim]

	Gate *pq4.Matrix // [ffn, hidden]
	Up   *pq4.Matrix // [ffn, hidden]
	Down *pq4.Matrix // [hidden, ffn]

	OutScale float32 // 層の出力に掛ける係数。例: 0.0544

	// RopeInverse は RoPE の逆周波数 [head_dim / 2] です。角度は「位置 × RopeInverse[i]」で、
	// 次元 i と i + head_dim / 2 を組にして回転します。
	RopeInverse []float32
}

// Weights はモデル全体の重みです。行列は .pq4m を mmap した領域を指します（File を閉じるまで有効）。
type Weights struct {
	Config     *Config
	File       *pq4.File
	Embedding  *pq4.Matrix // [vocab, hidden]。埋め込みの参照と語彙出力の両方に使う
	OutputNorm []float32   // [hidden]
	Layers     []LayerWeights
}

// loader は、重みを読む間の最初のエラーだけを覚えておく道具です。エラーの後は何も読みません。
type loader struct {
	file *pq4.File
	err  error
}

// vector は F32 のテンソルを読み、長さを確かめます。例: l.vector("output_norm.weight", 3840)
func (l *loader) vector(name string, length int) []float32 {
	if l.err != nil {
		return nil
	}
	values, shape, err := l.file.Float32(name)
	if err == nil && (len(shape) != 1 || shape[0] != length) {
		err = fmt.Errorf("gemma4: tensor %s has shape %v, want [%d]", name, shape, length)
	}
	l.err = err
	return values
}

// matrix は PQ4 の行列を読みます。例: l.matrix("blk.0.attn_q.weight", 4096, 3840)
func (l *loader) matrix(name string, rows, cols int) *pq4.Matrix {
	if l.err != nil {
		return nil
	}
	matrix, err := l.file.Matrix(name, rows, cols)
	l.err = err
	return matrix
}

// has は、名前のテンソルがファイルにあるかを返します。
func (l *loader) has(name string) bool {
	_, err := l.file.Record(name)
	return err == nil
}

// ropeInverseFrequencies は、RoPE の逆周波数（長さ headDim / 2）を llama.cpp（ggml の rope_neox）と同じ式で求めます。
//
//	逆周波数[i] = (base^(−2 / headDim))^i ÷ factors[i]
//
// factors が nil なら割りません。Gemma 4 の全文脈の層は、factors の後ろ 3/4 が 1e30 で、その次元はほぼ回転しません。
// 例: headDim 256、base 10000 -> [1, 0.9306, 0.8660, …]
func ropeInverseFrequencies(base float64, headDim int, factors []float32) []float32 {
	half := headDim / 2
	// llama.cpp は theta_scale を float32 で持ち、powf で i 乗します。同じ精度にそろえます。
	thetaScale := float32(math.Pow(float64(float32(base)), -2/float64(headDim)))
	inverse := make([]float32, half)
	for index := range inverse {
		value := float32(math.Pow(float64(thetaScale), float64(index)))
		if factors != nil {
			value /= factors[index]
		}
		inverse[index] = value
	}
	return inverse
}

// OpenWeights は .pq4m を開き、Gemma 4 の重みを読みます。戻り値の Close で mmap を解除します。
// 例: OpenWeights("./data/gemma4-12b-qat.pq4m")
func OpenWeights(path string) (*Weights, error) {
	file, err := pq4.OpenFile(path)
	if err != nil {
		return nil, err
	}
	weights, err := loadWeights(file)
	if err != nil {
		file.Close()
		return nil, err
	}
	return weights, nil
}

// Close は .pq4m の mmap を解除します。Close の後は、重みを使ってはいけません。
func (w *Weights) Close() error {
	return w.File.Close()
}

// loadWeights は、開いた .pq4m から構成と全テンソルを読みます。
func loadWeights(file *pq4.File) (*Weights, error) {
	config, err := ParseConfig(file.Header.ModelConfig)
	if err != nil {
		return nil, err
	}
	hidden := config.HiddenSize
	l := &loader{file: file}
	weights := &Weights{
		Config:     config,
		File:       file,
		Embedding:  l.matrix(NameEmbedding, config.VocabSize, hidden),
		OutputNorm: l.vector(nameOutputNorm, hidden),
		Layers:     make([]LayerWeights, config.NumLayers()),
	}
	// 全文脈の層の head_dim は層によらず同じなので、周波数の係数は 1 本です。
	var fullFactors []float32
	for index, layerConfig := range config.Layers {
		named := func(format string) string { return fmt.Sprintf(format, index) }
		queryWidth := config.QueryWidth(index)
		kvWidth := config.KVWidth(index)
		layer := LayerWeights{
			Config:       layerConfig,
			AttnNorm:     l.vector(named(nameAttnNorm), hidden),
			PostAttnNorm: l.vector(named(namePostAttnNorm), hidden),
			FFNNorm:      l.vector(named(nameFFNNorm), hidden),
			PostFFNNorm:  l.vector(named(namePostFFNNorm), hidden),
			QNorm:        l.vector(named(nameAttnQNorm), layerConfig.HeadDim),
			KNorm:        l.vector(named(nameAttnKNorm), layerConfig.HeadDim),
			Q:            l.matrix(named(nameAttnQ), queryWidth, hidden),
			K:            l.matrix(named(nameAttnK), kvWidth, hidden),
			O:            l.matrix(named(nameAttnO), hidden, queryWidth),
			Gate:         l.matrix(named(nameFFNGate), config.FeedForwardSize, hidden),
			Up:           l.matrix(named(nameFFNUp), config.FeedForwardSize, hidden),
			Down:         l.matrix(named(nameFFNDown), hidden, config.FeedForwardSize),
		}
		if l.has(named(nameAttnV)) {
			layer.V = l.matrix(named(nameAttnV), kvWidth, hidden)
		}
		if scale := l.vector(named(nameOutScale), 1); l.err == nil {
			layer.OutScale = scale[0]
		}
		if layerConfig.Sliding {
			layer.RopeInverse = ropeInverseFrequencies(config.RopeBaseSliding, layerConfig.HeadDim, nil)
		} else {
			if fullFactors == nil {
				fullFactors = l.vector(nameRopeFreqs, layerConfig.HeadDim/2)
			}
			if l.err == nil {
				layer.RopeInverse = ropeInverseFrequencies(config.RopeBaseFull, layerConfig.HeadDim, fullFactors)
			}
		}
		if l.err != nil {
			return nil, fmt.Errorf("gemma4: layer %d: %w", index, l.err)
		}
		weights.Layers[index] = layer
	}
	if l.err != nil {
		return nil, l.err
	}
	return weights, nil
}
