package llamakld

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

// TestOpenReadsTokensAndLogProbs は、n_ctx 4・語彙 3・区切り 2 のファイルから、トークン列と対数確率を読めることを確かめます。
func TestOpenReadsTokensAndLogProbs(t *testing.T) {
	const context, vocab, chunks = 4, 3, 2
	var content bytes.Buffer
	content.WriteString(magic)
	binary.Write(&content, binary.LittleEndian, []int32{context, vocab, chunks})
	binary.Write(&content, binary.LittleEndian, []int32{10, 11, 12, 13, 20, 21, 22, 23})
	// 1 区切りにつき 1 レコード（位置 2）。レコードは uint16 × (2 × ceil(3 / 2) + 4) = 8 個。
	for chunk := 0; chunk < chunks; chunk++ {
		binary.Write(&content, binary.LittleEndian, []float32{0.5, -4}) // scale、min_log_prob
		binary.Write(&content, binary.LittleEndian, []uint16{0, 2, uint16(4 + chunk), 0})
	}
	path := filepath.Join(t.TempDir(), "base.kld")
	if err := os.WriteFile(path, content.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	file, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	if file.ContextSize != context || file.VocabSize != vocab || file.Chunks != chunks || file.FirstScored() != 2 ||
		file.ScoredPerChunk() != 1 {
		t.Fatalf("header: context %d vocab %d chunks %d", file.ContextSize, file.VocabSize, file.Chunks)
	}
	if tokens := file.ChunkTokens(1); tokens[0] != 20 || tokens[3] != 23 {
		t.Errorf("chunk 1 tokens = %v", tokens)
	}
	values := make([]float32, vocab)
	floor, err := file.LogProbs(1, 2, values)
	if err != nil {
		t.Fatal(err)
	}
	// 区切り 1 の q は [0, 2, 5] なので、対数確率は −4 + 0.5 × q。
	if floor != -4 || values[0] != -4 || values[1] != -3 || values[2] != -1.5 {
		t.Errorf("log probs = %v, floor %g", values, floor)
	}
	if _, err := file.LogProbs(0, 1, values); err == nil {
		t.Error("LogProbs accepted a position that is not stored")
	}
}
