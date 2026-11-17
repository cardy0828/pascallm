package lfm3

import "fmt"

// EmbeddingTensorName は、埋め込み表のテンソル名です（GPU 実装が F16 のまま読むために公開しています）。
const EmbeddingTensorName = nameEmbedding

// RopeInverseFrequencies は、kind の層で使う RoPE の逆周波数（長さ head_dim/2）を返します。
// CPU の参照実装と同じ値を GPU 実装に渡すために使います。
// 例: 全文脈層（head_dim 128、factor 0.5）なら、先頭 32 個が正の値、後ろ 32 個が 0
func (c *Config) RopeInverseFrequencies(kind LayerKind) ([]float32, error) {
	rope, ok := c.RopeParameters[kind.String()]
	if !ok {
		return nil, fmt.Errorf("lfm3: rope_parameters has no entry for %s", kind)
	}
	return ropeInverseFrequencies(rope, c.HeadDim), nil
}
