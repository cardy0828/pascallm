// Package lfm3 は、LiquidAI LFM3 のテキストモデルを CPU 上の FP32 で 1 トークンずつ計算する参照実装です。
//
// 目的は速度ではなく、GPU 実装（M2 以降）と量子化形式（PQ4）の正しさを確かめる基準を作ることです。
// 計算の順序と式は、モデルに付属する modeling_lfm3.py（Hugging Face 版）に合わせています。
package lfm3

import (
	"encoding/json"
	"fmt"
	"os"
)

// LayerKind は層の種類です。
type LayerKind int

const (
	// LayerConv は gated short convolution（幅 7 の因果畳み込み）の層です。
	LayerConv LayerKind = iota
	// LayerFullAttention は全文脈を見る attention の層です（RoPE は head_dim の半分だけ回転）。
	LayerFullAttention
	// LayerSlidingAttention は直近 SlidingWindow トークンだけを見る attention の層です。
	LayerSlidingAttention
)

// String は層の種類を config.json の表記で返します。例: LayerConv -> "conv"
func (k LayerKind) String() string {
	switch k {
	case LayerConv:
		return "conv"
	case LayerFullAttention:
		return "full_attention"
	case LayerSlidingAttention:
		return "sliding_attention"
	default:
		return fmt.Sprintf("unknown(%d)", int(k))
	}
}

// IsAttention は attention 層（全文脈・窓付きのどちらか）なら true を返します。
func (k LayerKind) IsAttention() bool {
	return k == LayerFullAttention || k == LayerSlidingAttention
}

// RopeParameters は、層の種類ごとの RoPE の設定です。
// 例: 全文脈層は {Theta: 1e6, PartialRotaryFactor: 0.5}、窓付き層は {Theta: 10000, PartialRotaryFactor: 1}
type RopeParameters struct {
	Theta               float64 `json:"rope_theta"`
	PartialRotaryFactor float64 `json:"partial_rotary_factor"`
	Type                string  `json:"rope_type"`
}

// Config は、config.json のうち推論に必要な項目です。
type Config struct {
	HiddenSize        int                       `json:"hidden_size"`         // 例: 2048
	IntermediateSize  int                       `json:"intermediate_size"`   // 例: 7680
	NumLayers         int                       `json:"num_hidden_layers"`   // 例: 24
	NumHeads          int                       `json:"num_attention_heads"` // 例: 16
	NumKVHeads        int                       `json:"num_key_value_heads"` // 例: 4
	HeadDim           int                       `json:"head_dim"`            // 例: 128
	VocabSize         int                       `json:"vocab_size"`          // 例: 128000
	NormEps           float64                   `json:"norm_eps"`            // 例: 1e-5
	ConvKernelSize    int                       `json:"conv_L_cache"`        // 例: 7
	ConvBias          bool                      `json:"conv_bias"`           // LFM3 1.7B は false
	SlidingWindow     int                       `json:"sliding_window"`      // 例: 256
	TieWordEmbeddings bool                      `json:"tie_word_embeddings"`
	UseSandwichNorm   bool                      `json:"use_sandwich_norm"`
	LayerTypes        []string                  `json:"layer_types"`
	RopeParameters    map[string]RopeParameters `json:"rope_parameters"`
	BOSTokenID        int                       `json:"bos_token_id"`
	EOSTokenID        int                       `json:"eos_token_id"`
}

// LoadConfig は config.json を読み、この実装が対応する構成かを確かめます。
// 例: LoadConfig("./models/lfm3-1.7b-training/config.json")
func LoadConfig(path string) (*Config, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return ParseConfig(raw)
}

// ParseConfig は config.json の内容を解釈し、Validate で対応範囲を確かめます。
func ParseConfig(raw []byte) (*Config, error) {
	var config Config
	if err := json.Unmarshal(raw, &config); err != nil {
		return nil, fmt.Errorf("lfm3: config: %w", err)
	}
	if err := config.Validate(); err != nil {
		return nil, err
	}
	return &config, nil
}

// Validate は、この参照実装が扱える構成かを確かめます。
// 対応しない機能（畳み込みのバイアス、sandwich norm なし、default 以外の RoPE など）があればエラーにします。
func (c *Config) Validate() error {
	switch {
	case c.HiddenSize <= 0 || c.IntermediateSize <= 0 || c.NumHeads <= 0 || c.NumKVHeads <= 0 || c.HeadDim <= 0:
		return fmt.Errorf("lfm3: config has non-positive sizes")
	case c.NumHeads%c.NumKVHeads != 0:
		return fmt.Errorf("lfm3: num_attention_heads %d is not a multiple of num_key_value_heads %d", c.NumHeads, c.NumKVHeads)
	case len(c.LayerTypes) != c.NumLayers:
		return fmt.Errorf("lfm3: layer_types has %d entries, want %d", len(c.LayerTypes), c.NumLayers)
	case c.ConvBias:
		return fmt.Errorf("lfm3: conv_bias=true is not supported")
	case !c.UseSandwichNorm:
		return fmt.Errorf("lfm3: use_sandwich_norm=false is not supported")
	case !c.TieWordEmbeddings:
		return fmt.Errorf("lfm3: untied word embeddings are not supported")
	case c.HeadDim%2 != 0:
		return fmt.Errorf("lfm3: head_dim %d must be even for RoPE", c.HeadDim)
	}
	for index := range c.LayerTypes {
		kind, err := c.LayerKind(index)
		if err != nil {
			return err
		}
		if kind.IsAttention() {
			rope, ok := c.RopeParameters[kind.String()]
			if !ok {
				return fmt.Errorf("lfm3: rope_parameters has no entry for %s", kind)
			}
			if rope.Type != "default" {
				return fmt.Errorf("lfm3: rope_type %q is not supported", rope.Type)
			}
		}
	}
	return nil
}

// LayerKind は index 番目（0 始まり）の層の種類を返します。
func (c *Config) LayerKind(index int) (LayerKind, error) {
	switch c.LayerTypes[index] {
	case "conv":
		return LayerConv, nil
	case "full_attention":
		return LayerFullAttention, nil
	case "sliding_attention":
		return LayerSlidingAttention, nil
	default:
		return 0, fmt.Errorf("lfm3: layer %d has unknown type %q", index, c.LayerTypes[index])
	}
}

// partialRotaryFactor は RoPE を掛ける割合を返します。未指定（0）の場合は 1（全次元）です。
func (r RopeParameters) partialRotaryFactor() float64 {
	if r.PartialRotaryFactor == 0 {
		return 1
	}
	return r.PartialRotaryFactor
}
