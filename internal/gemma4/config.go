// Package gemma4 は、Google Gemma 4 のテキストモデル（12B など、層ごとの埋め込みと MoE のない構成）を
// CPU 上で 1 トークンずつ計算する参照実装です（M7）。
//
// 目的は速度ではなく、GPU 実装（internal/gemma4gpu）の正しさを確かめる基準を作ることです。
// 計算の順序と式は、llama.cpp の src/models/gemma4.cpp（GGUF を動かす実装）に合わせています。
// 重みは、GGUF の Q4_0 を詰め替えた .pq4m（cmd/gguf-to-pq4）から読みます。テンソル名は GGUF のままです。
//
// 1 層の計算:
//
//	x  = RMSNorm(h) × attn_norm
//	q, k, v = Wq·x, Wk·x, Wv·x（全文脈の層は Wv がなく、v に k の出力を使う）
//	q = RMSNorm(q) × q_norm に RoPE、k = RMSNorm(k) × k_norm に RoPE、v = RMSNorm(v)（ヘッドごと、v は重みなし）
//	a  = Wo·attention(q, k, v)（スコアに 1/sqrt(head_dim) は掛けない）
//	h1 = h + RMSNorm(a) × post_attention_norm
//	y  = RMSNorm(h1) × ffn_norm
//	f  = Wdown·(gelu(Wgate·y) × (Wup·y))
//	h  = (h1 + RMSNorm(f) × post_ffw_norm) × layer_output_scale
//
// 全体: h = 埋め込み × sqrt(hidden)、全層の後に logits = lm_head(RMSNorm(h) × output_norm)、
// 最後に logits = cap × tanh(logits / cap)（cap = final_logit_softcapping）。語彙出力は埋め込み表と共有です。
package gemma4

import (
	"encoding/json"
	"fmt"
	"math"

	"github.com/cardy0828/pascallm/internal/gguf"
)

// Architecture は、この実装が対応する GGUF の general.architecture です。
const Architecture = "gemma4"

// LayerConfig は 1 層分の attention の構成です。
// 例: 局所の層 {Sliding: true, KVHeads: 8, HeadDim: 256}、全文脈の層 {Sliding: false, KVHeads: 1, HeadDim: 512}
type LayerConfig struct {
	Sliding bool `json:"sliding"`  // 直近 SlidingWindow 個だけを見る層なら true
	KVHeads int  `json:"kv_heads"` // k・v のヘッド数
	HeadDim int  `json:"head_dim"` // 1 ヘッドの次元数
}

// Config はモデルの構成です。.pq4m のヘッダー（model_config）に JSON で保存します。
type Config struct {
	Architecture      string        `json:"architecture"`            // "gemma4"
	HiddenSize        int           `json:"hidden_size"`             // 例: 3840
	FeedForwardSize   int           `json:"feed_forward_size"`       // 例: 15360
	NumHeads          int           `json:"num_heads"`               // query のヘッド数。例: 16
	VocabSize         int           `json:"vocab_size"`              // 例: 262144
	NormEps           float64       `json:"norm_eps"`                // 例: 1e-6
	SlidingWindow     int           `json:"sliding_window"`          // 例: 1024
	RopeBaseFull      float64       `json:"rope_base_full"`          // 全文脈の層の RoPE の底。例: 1e6
	RopeBaseSliding   float64       `json:"rope_base_sliding"`       // 局所の層の RoPE の底。例: 10000
	FinalLogitSoftcap float64       `json:"final_logit_softcapping"` // 0 なら使わない。例: 30
	Layers            []LayerConfig `json:"layers"`
	BOSTokenID        int           `json:"bos_token_id"` // 例: 2
	EOSTokenID        int           `json:"eos_token_id"` // 例: 1
	EOTTokenID        int           `json:"eot_token_id"` // ターンの終わり。例: 106
}

// NumLayers は層の数を返します。例: 48
func (c *Config) NumLayers() int { return len(c.Layers) }

// EmbeddingScale は、埋め込みに掛ける sqrt(hidden_size) を返します（llama.cpp と同じく float32 で計算する）。
// 例: hidden_size 3840 -> 61.967735
func (c *Config) EmbeddingScale() float32 {
	return float32(math.Sqrt(float64(c.HiddenSize)))
}

// QueryWidth は、層 layer の q の出力の次元数（ヘッド数 × ヘッドの次元数）を返します。例: 局所 4096、全文脈 8192
func (c *Config) QueryWidth(layer int) int {
	return c.NumHeads * c.Layers[layer].HeadDim
}

// KVWidth は、層 layer の k（v も同じ）の出力の次元数を返します。例: 局所 2048、全文脈 512
func (c *Config) KVWidth(layer int) int {
	return c.Layers[layer].KVHeads * c.Layers[layer].HeadDim
}

// Validate は、構成がこの実装の前提を満たすかを確かめます。
func (c *Config) Validate() error {
	switch {
	case c.Architecture != Architecture:
		return fmt.Errorf("gemma4: architecture %q is not supported (only %q)", c.Architecture, Architecture)
	case c.HiddenSize <= 0 || c.FeedForwardSize <= 0 || c.NumHeads <= 0 || c.VocabSize <= 0:
		return fmt.Errorf("gemma4: sizes must be positive (hidden %d, ffn %d, heads %d, vocab %d)",
			c.HiddenSize, c.FeedForwardSize, c.NumHeads, c.VocabSize)
	case len(c.Layers) == 0:
		return fmt.Errorf("gemma4: the model has no layers")
	case c.SlidingWindow <= 0:
		return fmt.Errorf("gemma4: sliding_window %d must be positive", c.SlidingWindow)
	}
	for index, layer := range c.Layers {
		if layer.KVHeads <= 0 || c.NumHeads%layer.KVHeads != 0 {
			return fmt.Errorf("gemma4: layer %d has %d kv heads, which does not divide %d heads", index, layer.KVHeads, c.NumHeads)
		}
		if layer.HeadDim <= 0 || layer.HeadDim%2 != 0 {
			return fmt.Errorf("gemma4: layer %d has head_dim %d (must be a positive even number)", index, layer.HeadDim)
		}
	}
	return nil
}

// ParseConfig は、.pq4m のヘッダーに保存した JSON から Config を読みます。
func ParseConfig(raw []byte) (*Config, error) {
	var config Config
	if err := json.Unmarshal(raw, &config); err != nil {
		return nil, fmt.Errorf("gemma4: model_config: %w", err)
	}
	if err := config.Validate(); err != nil {
		return nil, err
	}
	return &config, nil
}

// ggufReader は、GGUF のメタデータを読む間の最初のエラーだけを覚えておく道具です。
// キーは "gemma4." を前に付けて読みます。
type ggufReader struct {
	file *gguf.File
	err  error
}

// integer は整数のメタデータを読みます。例: r.integer("gemma4.block_count") -> 48
func (r *ggufReader) integer(key string) int {
	if r.err != nil {
		return 0
	}
	value, err := r.file.Int(key)
	r.err = err
	return int(value)
}

// float は小数のメタデータを読みます。例: r.float("gemma4.rope.freq_base") -> 1e6
func (r *ggufReader) float(key string) float64 {
	if r.err != nil {
		return 0
	}
	value, err := r.file.Float(key)
	r.err = err
	return value
}

// ConfigFromGGUF は、GGUF のメタデータから Config を作ります（変換ツールが使う）。
// 層ごとの埋め込み（E2B・E4B）、KV を共有する層、MoE（26B）を持つモデルは、この実装の対象外なのでエラーにします。
func ConfigFromGGUF(file *gguf.File) (*Config, error) {
	architecture, err := file.String("general.architecture")
	if err != nil {
		return nil, err
	}
	if architecture != Architecture {
		return nil, fmt.Errorf("gemma4: architecture %q is not supported (only %q)", architecture, Architecture)
	}
	const prefix = Architecture + "."
	r := &ggufReader{file: file}
	layerCount := r.integer(prefix + "block_count")
	config := &Config{
		Architecture:      architecture,
		HiddenSize:        r.integer(prefix + "embedding_length"),
		FeedForwardSize:   r.integer(prefix + "feed_forward_length"),
		NumHeads:          r.integer(prefix + "attention.head_count"),
		NormEps:           r.float(prefix + "attention.layer_norm_rms_epsilon"),
		SlidingWindow:     r.integer(prefix + "attention.sliding_window"),
		RopeBaseFull:      r.float(prefix + "rope.freq_base"),
		RopeBaseSliding:   r.float(prefix + "rope.freq_base_swa"),
		FinalLogitSoftcap: r.float(prefix + "final_logit_softcapping"),
		BOSTokenID:        r.integer("tokenizer.ggml.bos_token_id"),
		EOSTokenID:        r.integer("tokenizer.ggml.eos_token_id"),
		EOTTokenID:        r.integer("tokenizer.ggml.eot_token_id"),
	}
	headDimFull := r.integer(prefix + "attention.key_length")
	headDimSliding := r.integer(prefix + "attention.key_length_swa")
	perLayerEmbedding := r.integer(prefix + "embedding_length_per_layer_input")
	sharedKVLayers := r.integer(prefix + "attention.shared_kv_layers")
	if r.err != nil {
		return nil, r.err
	}
	if perLayerEmbedding != 0 || sharedKVLayers != 0 {
		return nil, fmt.Errorf("gemma4: per-layer embeddings (%d) and shared KV layers (%d) are not supported", perLayerEmbedding, sharedKVLayers)
	}
	kvHeads, err := file.IntArray(prefix + "attention.head_count_kv")
	if err != nil {
		return nil, err
	}
	sliding, err := file.BoolArray(prefix + "attention.sliding_window_pattern")
	if err != nil {
		return nil, err
	}
	if len(kvHeads) != layerCount || len(sliding) != layerCount {
		return nil, fmt.Errorf("gemma4: %d layers, but %d kv head counts and %d sliding flags", layerCount, len(kvHeads), len(sliding))
	}
	for index := 0; index < layerCount; index++ {
		headDim := headDimFull
		if sliding[index] {
			headDim = headDimSliding
		}
		config.Layers = append(config.Layers, LayerConfig{Sliding: sliding[index], KVHeads: int(kvHeads[index]), HeadDim: headDim})
	}
	embedding, err := file.Tensor(NameEmbedding)
	if err != nil {
		return nil, err
	}
	if len(embedding.Dims) != 2 || embedding.Dims[0] != config.HiddenSize {
		return nil, fmt.Errorf("gemma4: %s has shape %v, want [%d, vocab]", NameEmbedding, embedding.Dims, config.HiddenSize)
	}
	config.VocabSize = embedding.Dims[1]
	if err := config.Validate(); err != nil {
		return nil, err
	}
	return config, nil
}
