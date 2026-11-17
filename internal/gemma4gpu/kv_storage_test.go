package gemma4gpu

import "testing"

// TestKVStorage はメモリ幅とカーネルの組合せ、無効指定を検査します。GPU は不要です。
func TestKVStorage(t *testing.T) {
	for _, c := range []struct {
		name string
		k, v int
	}{{"", 2, 2}, {"fp16", 2, 2}, {"k-fp32", 4, 2}, {"fp32", 4, 4}} {
		s, err := resolveKVStorage(c.name)
		if err != nil || s.keyBytes != c.k || s.valueBytes != c.v || len(s.cubin) == 0 {
			t.Fatalf("%q: %+v %v", c.name, s, err)
		}
	}
	if _, err := resolveKVStorage("fp8"); err == nil {
		t.Fatal("invalid format accepted")
	}
}
