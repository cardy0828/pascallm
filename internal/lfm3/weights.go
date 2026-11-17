package lfm3

import (
	"fmt"
	"path/filepath"

	"github.com/cardy0828/pascallm/internal/linear"
	"github.com/cardy0828/pascallm/internal/safetensors"
)

// EmbeddingTable は、トークン ID から埋め込みベクトルを取り出せる表です。
type EmbeddingTable interface {
	// Lookup は token 番目の行を dst（長さ hidden_size）に書き込みます。
	Lookup(token int, dst []float32)
}

// DenseEmbedding は FP32 の埋め込み表です。
type DenseEmbedding struct {
	Table *linear.Dense
}

// Lookup は token 番目の行を dst にコピーします。
func (e DenseEmbedding) Lookup(token int, dst []float32) {
	copy(dst, e.Table.Row(token))
}

// ConvWeights は short convolution 層の重みです。
type ConvWeights struct {
	InProj  linear.Matrix // [3 × hidden, hidden]。出力は B・C・x の順に hidden ずつ並ぶ
	OutProj linear.Matrix // [hidden, hidden]
	Kernel  []float32     // [hidden][kernel_size]。Kernel[c*kernel_size+k] が、k = kernel_size-1 で最新の入力に掛かる
}

// AttentionWeights は attention 層の重みです。
type AttentionWeights struct {
	Q, K, V, O linear.Matrix
	Gate       linear.Matrix // [num_heads, hidden]。ヘッドごとの sigmoid ゲート
	QNorm      []float32     // [head_dim]
	KNorm      []float32     // [head_dim]
}

// FeedForwardWeights は FFN の重みです。y = W2(silu(W1·x) * (W3·x))
type FeedForwardWeights struct {
	W1, W2, W3 linear.Matrix
}

// LayerWeights は 1 層分の重みです。Conv と Attention は、層の種類に応じてどちらか一方だけが入ります。
type LayerWeights struct {
	Kind             LayerKind
	OperatorNorm     []float32
	PostOperatorNorm []float32
	FFNNorm          []float32
	PostFFNNorm      []float32
	Conv             *ConvWeights
	Attention        *AttentionWeights
	FeedForward      FeedForwardWeights
}

// Weights はモデル全体の重みです。
type Weights struct {
	Config    *Config
	Embedding EmbeddingTable
	Layers    []LayerWeights
	FinalNorm []float32     // embedding_norm（名前に反して全層の後に掛かる最終正規化）
	LMHead    linear.Matrix // [vocab, hidden]。埋め込み表と共有（tie_word_embeddings）
}

// Hugging Face 形式でのテンソル名です。%d には層の番号（0 始まり）が入ります。
const (
	nameEmbedding        = "model.embed_tokens.weight"
	nameFinalNorm        = "model.embedding_norm.weight"
	nameOperatorNorm     = "model.layers.%d.operator_norm.weight"
	namePostOperatorNorm = "model.layers.%d.post_operator_norm.weight"
	nameFFNNorm          = "model.layers.%d.ffn_norm.weight"
	namePostFFNNorm      = "model.layers.%d.post_ffn_norm.weight"
	nameConvInProj       = "model.layers.%d.conv.in_proj.weight"
	nameConvOutProj      = "model.layers.%d.conv.out_proj.weight"
	nameConvKernel       = "model.layers.%d.conv.conv.weight"
	nameAttnQ            = "model.layers.%d.self_attn.q_proj.weight"
	nameAttnK            = "model.layers.%d.self_attn.k_proj.weight"
	nameAttnV            = "model.layers.%d.self_attn.v_proj.weight"
	nameAttnO            = "model.layers.%d.self_attn.o_proj.weight"
	nameAttnGate         = "model.layers.%d.self_attn.gate_proj.weight"
	nameAttnQNorm        = "model.layers.%d.self_attn.q_norm.weight"
	nameAttnKNorm        = "model.layers.%d.self_attn.k_norm.weight"
	nameFFNW1            = "model.layers.%d.feed_forward.w1.weight"
	nameFFNW2            = "model.layers.%d.feed_forward.w2.weight"
	nameFFNW3            = "model.layers.%d.feed_forward.w3.weight"
)

// TensorSpec は、モデルが必要とする 1 テンソルの名前・形状と、行列（Linear）かどうかです。
// 変換ツールは、この一覧に従って量子化の対象を決めます。
type TensorSpec struct {
	Name   string
	Shape  []int
	Linear bool // nn.Linear の重み（量子化の候補）なら true
}

// TensorSpecs は、config に従って必要なテンソルを、変換・読み込みの順に並べて返します。
// 埋め込み表は Linear=false（参照表）として返し、語彙出力への流用は呼び出し側が決めます。
func TensorSpecs(config *Config) ([]TensorSpec, error) {
	hidden := config.HiddenSize
	attentionWidth := config.NumHeads * config.HeadDim
	kvWidth := config.NumKVHeads * config.HeadDim
	specs := []TensorSpec{
		{Name: nameEmbedding, Shape: []int{config.VocabSize, hidden}},
		{Name: nameFinalNorm, Shape: []int{hidden}},
	}
	for layer := 0; layer < config.NumLayers; layer++ {
		kind, err := config.LayerKind(layer)
		if err != nil {
			return nil, err
		}
		named := func(format string, shape []int, isLinear bool) TensorSpec {
			return TensorSpec{Name: fmt.Sprintf(format, layer), Shape: shape, Linear: isLinear}
		}
		specs = append(specs,
			named(nameOperatorNorm, []int{hidden}, false),
			named(namePostOperatorNorm, []int{hidden}, false),
			named(nameFFNNorm, []int{hidden}, false),
			named(namePostFFNNorm, []int{hidden}, false),
		)
		if kind.IsAttention() {
			specs = append(specs,
				named(nameAttnQ, []int{attentionWidth, hidden}, true),
				named(nameAttnK, []int{kvWidth, hidden}, true),
				named(nameAttnV, []int{kvWidth, hidden}, true),
				named(nameAttnO, []int{hidden, attentionWidth}, true),
				named(nameAttnGate, []int{config.NumHeads, hidden}, true),
				named(nameAttnQNorm, []int{config.HeadDim}, false),
				named(nameAttnKNorm, []int{config.HeadDim}, false),
			)
		} else {
			specs = append(specs,
				named(nameConvInProj, []int{3 * hidden, hidden}, true),
				named(nameConvOutProj, []int{hidden, hidden}, true),
				named(nameConvKernel, []int{hidden, 1, config.ConvKernelSize}, false),
			)
		}
		specs = append(specs,
			named(nameFFNW1, []int{config.IntermediateSize, hidden}, true),
			named(nameFFNW2, []int{hidden, config.IntermediateSize}, true),
			named(nameFFNW3, []int{config.IntermediateSize, hidden}, true),
		)
	}
	return specs, nil
}

// TensorSource は、名前を指定して重みを取り出せる入力です（safetensors・PQ4 ファイルのどちらも実装する）。
type TensorSource interface {
	// Vector は 1 次元（または [hidden, 1, k] のような小さな）テンソルを FP32 で返します。
	Vector(name string, shape []int) ([]float32, error)
	// Matrix は 2 次元の重みを行列として返します。
	Matrix(name string, rows, cols int) (linear.Matrix, error)
	// Embedding は埋め込み表を返します。
	Embedding(name string, vocab, hidden int) (EmbeddingTable, error)
	// LMHead は語彙出力の行列を返します（LFM3 は埋め込み表と共有）。
	LMHead(embeddingName string, vocab, hidden int) (linear.Matrix, error)
}

// LoadWeights は source から全テンソルを読み、Weights を組み立てます。
func LoadWeights(config *Config, source TensorSource) (*Weights, error) {
	weights := &Weights{Config: config, Layers: make([]LayerWeights, config.NumLayers)}
	var err error
	if weights.Embedding, err = source.Embedding(nameEmbedding, config.VocabSize, config.HiddenSize); err != nil {
		return nil, err
	}
	if weights.LMHead, err = source.LMHead(nameEmbedding, config.VocabSize, config.HiddenSize); err != nil {
		return nil, err
	}
	if weights.FinalNorm, err = source.Vector(nameFinalNorm, []int{config.HiddenSize}); err != nil {
		return nil, err
	}
	for layer := range weights.Layers {
		if err := loadLayer(config, source, layer, &weights.Layers[layer]); err != nil {
			return nil, err
		}
	}
	return weights, nil
}

// layerLoader は 1 層分の読み込みで、最初に起きたエラーだけを覚えておく小さな道具です。
// エラーの後の読み込みは何もせず nil を返すので、呼び出し側は最後に err を 1 回確かめれば済みます。
type layerLoader struct {
	source TensorSource
	layer  int
	err    error
}

// vector は層内の 1 次元テンソルを読みます。例: loader.vector(nameFFNNorm, 2048)
func (l *layerLoader) vector(format string, shape ...int) []float32 {
	if l.err != nil {
		return nil
	}
	values, err := l.source.Vector(fmt.Sprintf(format, l.layer), shape)
	l.err = err
	return values
}

// matrix は層内の 2 次元の重みを読みます。例: loader.matrix(nameAttnQ, 2048, 2048)
func (l *layerLoader) matrix(format string, rows, cols int) linear.Matrix {
	if l.err != nil {
		return nil
	}
	matrix, err := l.source.Matrix(fmt.Sprintf(format, l.layer), rows, cols)
	l.err = err
	return matrix
}

// loadLayer は layer 番目の層の重みを読み、out に入れます。
func loadLayer(config *Config, source TensorSource, layer int, out *LayerWeights) error {
	kind, err := config.LayerKind(layer)
	if err != nil {
		return err
	}
	hidden := config.HiddenSize
	loader := &layerLoader{source: source, layer: layer}
	out.Kind = kind
	out.OperatorNorm = loader.vector(nameOperatorNorm, hidden)
	out.PostOperatorNorm = loader.vector(namePostOperatorNorm, hidden)
	out.FFNNorm = loader.vector(nameFFNNorm, hidden)
	out.PostFFNNorm = loader.vector(namePostFFNNorm, hidden)
	if kind.IsAttention() {
		attentionWidth := config.NumHeads * config.HeadDim
		kvWidth := config.NumKVHeads * config.HeadDim
		out.Attention = &AttentionWeights{
			Q:     loader.matrix(nameAttnQ, attentionWidth, hidden),
			K:     loader.matrix(nameAttnK, kvWidth, hidden),
			V:     loader.matrix(nameAttnV, kvWidth, hidden),
			O:     loader.matrix(nameAttnO, hidden, attentionWidth),
			Gate:  loader.matrix(nameAttnGate, config.NumHeads, hidden),
			QNorm: loader.vector(nameAttnQNorm, config.HeadDim),
			KNorm: loader.vector(nameAttnKNorm, config.HeadDim),
		}
	} else {
		out.Conv = &ConvWeights{
			InProj:  loader.matrix(nameConvInProj, 3*hidden, hidden),
			OutProj: loader.matrix(nameConvOutProj, hidden, hidden),
			Kernel:  loader.vector(nameConvKernel, hidden, 1, config.ConvKernelSize),
		}
	}
	out.FeedForward = FeedForwardWeights{
		W1: loader.matrix(nameFFNW1, config.IntermediateSize, hidden),
		W2: loader.matrix(nameFFNW2, hidden, config.IntermediateSize),
		W3: loader.matrix(nameFFNW3, config.IntermediateSize, hidden),
	}
	if loader.err != nil {
		return fmt.Errorf("lfm3: layer %d: %w", layer, loader.err)
	}
	return nil
}

// SafetensorsSource は、Hugging Face 形式のモデルを FP32 に変換して読む TensorSource です。
type SafetensorsSource struct {
	File   *safetensors.File
	tables map[string]*linear.Dense // 変換済みの埋め込み表（名前ごと）
}

// OpenSafetensorsModel は、モデルのディレクトリから config.json と model.safetensors を開きます。
// 例: OpenSafetensorsModel("./models/lfm3-1.7b-training")
func OpenSafetensorsModel(directory string) (*Config, *SafetensorsSource, error) {
	config, err := LoadConfig(filepath.Join(directory, "config.json"))
	if err != nil {
		return nil, nil, err
	}
	file, err := safetensors.Open(filepath.Join(directory, "model.safetensors"))
	if err != nil {
		return nil, nil, err
	}
	return config, &SafetensorsSource{File: file}, nil
}

// Vector は 1 次元のテンソルを FP32 で返します。
func (s *SafetensorsSource) Vector(name string, shape []int) ([]float32, error) {
	return s.File.Float32(name, shape)
}

// Matrix は 2 次元の重みを FP32 の Dense として返します。
func (s *SafetensorsSource) Matrix(name string, rows, cols int) (linear.Matrix, error) {
	values, err := s.File.Float32(name, []int{rows, cols})
	if err != nil {
		return nil, err
	}
	return linear.NewDense(rows, cols, values)
}

// Embedding は埋め込み表を FP32 で返します。LMHead と同じ配列を共有するため、2 回目は変換しません。
func (s *SafetensorsSource) Embedding(name string, vocab, hidden int) (EmbeddingTable, error) {
	table, err := s.denseTable(name, vocab, hidden)
	if err != nil {
		return nil, err
	}
	return DenseEmbedding{Table: table}, nil
}

// LMHead は埋め込み表と同じ FP32 の行列を返します（tie_word_embeddings）。
func (s *SafetensorsSource) LMHead(embeddingName string, vocab, hidden int) (linear.Matrix, error) {
	return s.denseTable(embeddingName, vocab, hidden)
}

// denseTable は埋め込み表を FP32 の Dense として返し、2 回目以降は同じものを返します
// （埋め込みと語彙出力で 1 GB の配列を二重に持たないため）。
func (s *SafetensorsSource) denseTable(name string, vocab, hidden int) (*linear.Dense, error) {
	if table, ok := s.tables[name]; ok {
		return table, nil
	}
	values, err := s.File.Float32(name, []int{vocab, hidden})
	if err != nil {
		return nil, err
	}
	table, err := linear.NewDense(vocab, hidden, values)
	if err != nil {
		return nil, err
	}
	if s.tables == nil {
		s.tables = map[string]*linear.Dense{}
	}
	s.tables[name] = table
	return table, nil
}
