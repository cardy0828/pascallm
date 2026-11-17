package gemma4gpu

import (
	"fmt"
	"github.com/cardy0828/pascallm/internal/kernels"
)

// kvStorage はメモリ確保の要素幅と、その型でビルドしたカーネルを一緒に管理します。
type kvStorage struct {
	name                 string
	keyBytes, valueBytes int
	cubin                []byte
}

// resolveKVStorage は起動指定を検査します。例: k-fp32 -> K 4 byte、V 2 byte。
func resolveKVStorage(name string) (kvStorage, error) {
	switch name {
	case "", "fp16":
		return kvStorage{"fp16", float16Bytes, float16Bytes, kernels.Gemma4SM61}, nil
	case "k-fp32":
		return kvStorage{"k-fp32", float32Bytes, float16Bytes, kernels.Gemma4K32SM61}, nil
	case "fp32":
		return kvStorage{"fp32", float32Bytes, float32Bytes, kernels.Gemma4KV32SM61}, nil
	default:
		return kvStorage{}, fmt.Errorf("gemma4gpu: invalid KV cache %q (fp16, k-fp32, fp32)", name)
	}
}

// KVCacheFormat は実際の KV 格納形式を返します。例: k-fp32。
func (m *Model) KVCacheFormat() string { return m.kvCache }
