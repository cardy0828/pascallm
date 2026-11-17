package gguf

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

// builder は、テスト用の GGUF を組み立てる道具です。
type builder struct{ bytes.Buffer }

// put は値をリトルエンディアンで書きます。
func (b *builder) put(value any) { binary.Write(&b.Buffer, binary.LittleEndian, value) }

// text は「長さ + バイト列」の文字列を書きます。
func (b *builder) text(value string) {
	b.put(uint64(len(value)))
	b.WriteString(value)
}

// buildFile は、メタデータ 4 個とテンソル 2 個（F32 の 2 要素、Q4_0 の 32 × 2）の GGUF を作ります。
func buildFile() []byte {
	b := &builder{}
	b.WriteString(magic)
	b.put(uint32(3))
	b.put(uint64(2)) // テンソル数
	b.put(uint64(4)) // メタデータ数
	b.text("general.architecture")
	b.put(uint32(valueString))
	b.text("gemma4")
	b.text("gemma4.block_count")
	b.put(uint32(valueUint32))
	b.put(uint32(48))
	b.text("gemma4.rope.freq_base")
	b.put(uint32(valueFloat32))
	b.put(float32(1e6))
	b.text("flags")
	b.put(uint32(valueArray))
	b.put(uint32(valueBool))
	b.put(uint64(3))
	b.Write([]byte{1, 0, 1})

	b.text("norm.weight")
	b.put(uint32(1))
	b.put(uint64(2))
	b.put(uint32(TypeF32))
	b.put(uint64(0))
	b.text("proj.weight")
	b.put(uint32(2))
	b.put(uint64(32))
	b.put(uint64(2))
	b.put(uint32(TypeQ4_0))
	b.put(uint64(32)) // 2 番目のテンソルは、データ領域の 32 バイト目から

	for b.Len()%defaultAlignment != 0 {
		b.WriteByte(0)
	}
	b.put([]float32{1.5, -2})
	b.Write(make([]byte, 24)) // 32 バイト境界まで
	for index := 0; index < 2*Q4BlockBytes; index++ {
		b.WriteByte(byte(index))
	}
	return b.Bytes()
}

// TestOpenReadsMetadataAndTensors は、メタデータの型ごとの読み出しと、テンソルの範囲を確かめます。
func TestOpenReadsMetadataAndTensors(t *testing.T) {
	path := filepath.Join(t.TempDir(), "test.gguf")
	if err := os.WriteFile(path, buildFile(), 0o600); err != nil {
		t.Fatal(err)
	}
	file, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	if name, err := file.String("general.architecture"); err != nil || name != "gemma4" {
		t.Errorf("architecture = %q, %v", name, err)
	}
	if count, err := file.Int("gemma4.block_count"); err != nil || count != 48 {
		t.Errorf("block_count = %d, %v", count, err)
	}
	if base, err := file.Float("gemma4.rope.freq_base"); err != nil || base != 1e6 {
		t.Errorf("freq_base = %g, %v", base, err)
	}
	if flags, err := file.BoolArray("flags"); err != nil || len(flags) != 3 || !flags[0] || flags[1] || !flags[2] {
		t.Errorf("flags = %v, %v", flags, err)
	}
	if _, err := file.Int("general.architecture"); err == nil {
		t.Error("Int accepted a string value")
	}
	matrix, err := file.Tensor("proj.weight")
	if err != nil {
		t.Fatal(err)
	}
	data, err := file.Bytes(matrix)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) != 2*Q4BlockBytes || data[0] != 0 || data[len(data)-1] != byte(2*Q4BlockBytes-1) {
		t.Errorf("proj.weight data has %d bytes, first %d, last %d", len(data), data[0], data[len(data)-1])
	}
	if _, err := file.Tensor("missing"); err == nil {
		t.Error("Tensor found a missing name")
	}
}

// TestOpenRejectsTruncatedFile は、途中で切れたファイルを断ることを確かめます。
func TestOpenRejectsTruncatedFile(t *testing.T) {
	content := buildFile()
	for _, length := range []int{3, 20, len(content) - 1} {
		path := filepath.Join(t.TempDir(), "broken.gguf")
		if err := os.WriteFile(path, content[:length], 0o600); err != nil {
			t.Fatal(err)
		}
		if file, err := Open(path); err == nil {
			file.Close()
			t.Errorf("Open accepted a file truncated to %d bytes", length)
		}
	}
}
