// Package llamakld は、llama.cpp の llama-perplexity --kl-divergence-base が書くファイルを読みます。
//
// このファイルには、評価したトークン列と、各区切り（n_ctx トークン）の後半の位置での次トークンの対数確率が
// 入っています。llama.cpp を基準にして、自作の実装の logits を突き合わせるために使います（M7）。
//
// ファイルの配置（リトルエンディアン）:
//
//	"_logits_"（8 バイト）、n_ctx（int32）、n_vocab（int32）、n_chunk（int32）
//	トークン列: int32 × n_chunk × n_ctx
//	対数確率: 区切りごとに、位置 n_ctx/2 〜 n_ctx−2 の順で 1 位置 1 レコード
//
// 1 レコードは uint16 × (2 × ceil(n_vocab / 2) + 4) です。先頭の 8 バイトは float32 の scale と min_log_prob で、
// その後に語彙の数だけ uint16 の q が並びます。対数確率は min_log_prob + scale × q です。
// 最大値より 16 以上小さい logits は、最小値に切り上げられています（q = 0）。
//
// llama-perplexity は、各区切りの最初のトークンを BOS に置き換えて評価します（語彙が BOS を付ける設定のとき）。
// ファイルのトークン列は置き換える前のものなので、同じ計算をするには区切りの先頭を BOS にしてください。
package llamakld

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"syscall"
)

// ファイルの形式に関する定数です。
const (
	magic            = "_logits_"
	headerBytes      = len(magic) + 3*int32Bytes // マジック、n_ctx、n_vocab、n_chunk
	int32Bytes       = 4
	uint16Bytes      = 2
	recordHeaderSize = 4 // レコードの先頭の uint16 の数（float32 の scale と min_log_prob）
)

// File は、読み取り用に開いた基準ファイルです。Close で mmap を解除します。
type File struct {
	ContextSize int // 1 区切りのトークン数（n_ctx）
	VocabSize   int
	Chunks      int // 区切りの数

	mapping []byte
	tokens  []byte // int32 × Chunks × ContextSize
	records []byte
}

// Open は基準ファイルを mmap して開き、大きさを確かめます。
// 例: Open("./data/m7/gemma12b-base.kld")
func Open(path string) (*File, error) {
	handle, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer handle.Close()
	stat, err := handle.Stat()
	if err != nil {
		return nil, err
	}
	if stat.Size() < int64(headerBytes) {
		return nil, fmt.Errorf("llamakld: %s is too small", path)
	}
	mapping, err := syscall.Mmap(int(handle.Fd()), 0, int(stat.Size()), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("llamakld: mmap %s: %w", path, err)
	}
	file, err := parse(path, mapping)
	if err != nil {
		syscall.Munmap(mapping)
		return nil, err
	}
	return file, nil
}

// parse はヘッダーを読み、トークン列と対数確率の範囲を決めます。
func parse(path string, mapping []byte) (*File, error) {
	if string(mapping[:len(magic)]) != magic {
		return nil, fmt.Errorf("llamakld: %s is not a llama.cpp logits file", path)
	}
	field := func(index int) int {
		offset := len(magic) + index*int32Bytes
		return int(int32(binary.LittleEndian.Uint32(mapping[offset : offset+int32Bytes])))
	}
	file := &File{ContextSize: field(0), VocabSize: field(1), Chunks: field(2), mapping: mapping}
	if file.ContextSize < 2 || file.VocabSize <= 0 || file.Chunks <= 0 {
		return nil, fmt.Errorf("llamakld: %s has n_ctx %d, n_vocab %d, n_chunk %d", path, file.ContextSize, file.VocabSize, file.Chunks)
	}
	tokenBytes := int64(file.Chunks) * int64(file.ContextSize) * int32Bytes
	recordBytes := int64(file.Chunks) * int64(file.ScoredPerChunk()) * int64(file.recordSize())
	if int64(headerBytes)+tokenBytes+recordBytes > int64(len(mapping)) {
		return nil, fmt.Errorf("llamakld: %s is truncated (need %d bytes, have %d)", path, int64(headerBytes)+tokenBytes+recordBytes, len(mapping))
	}
	file.tokens = mapping[headerBytes : int64(headerBytes)+tokenBytes]
	file.records = mapping[int64(headerBytes)+tokenBytes:]
	return file, nil
}

// Close は mmap を解除します。
func (f *File) Close() error {
	return syscall.Munmap(f.mapping)
}

// FirstScored は、各区切りで対数確率が保存されている最初の位置（n_ctx / 2）を返します。例: n_ctx 128 -> 64
func (f *File) FirstScored() int { return f.ContextSize / 2 }

// ScoredPerChunk は、1 区切りで保存されている位置の数（n_ctx − 1 − n_ctx/2）を返します。例: n_ctx 128 -> 63
func (f *File) ScoredPerChunk() int { return f.ContextSize - 1 - f.FirstScored() }

// recordSize は 1 レコードのバイト数です。例: n_vocab 262144 -> (262144 + 4) × 2
func (f *File) recordSize() int {
	return (2*((f.VocabSize+1)/2) + recordHeaderSize) * uint16Bytes
}

// ChunkTokens は、区切り chunk（0 始まり）のトークン列（長さ ContextSize）を返します（BOS に置き換える前のもの）。
func (f *File) ChunkTokens(chunk int) []int {
	tokens := make([]int, f.ContextSize)
	base := chunk * f.ContextSize * int32Bytes
	for index := range tokens {
		offset := base + index*int32Bytes
		tokens[index] = int(int32(binary.LittleEndian.Uint32(f.tokens[offset : offset+int32Bytes])))
	}
	return tokens
}

// LogProbs は、区切り chunk の位置 position（FirstScored() 〜 ContextSize − 2）の入力に対する、
// 次トークンの対数確率を dst（長さ VocabSize）に書き、切り上げの下限（min_log_prob）を返します。
// 下限と同じ値のトークンは、本当の対数確率がそれ以下で、値がわかりません（比べるときは除く）。
// 例: LogProbs(0, 64, dst) は、区切り 0 のトークン 0〜64 を入れた後の分布
func (f *File) LogProbs(chunk, position int, dst []float32) (floor float32, err error) {
	index := position - f.FirstScored()
	if chunk < 0 || chunk >= f.Chunks || index < 0 || index >= f.ScoredPerChunk() || len(dst) != f.VocabSize {
		return 0, fmt.Errorf("llamakld: no record for chunk %d position %d (dst %d)", chunk, position, len(dst))
	}
	size := f.recordSize()
	start := (chunk*f.ScoredPerChunk() + index) * size
	record := f.records[start : start+size]
	scale := math.Float32frombits(binary.LittleEndian.Uint32(record[0:4]))
	minimum := math.Float32frombits(binary.LittleEndian.Uint32(record[4:8]))
	values := record[recordHeaderSize*uint16Bytes:]
	for token := range dst {
		quantized := binary.LittleEndian.Uint16(values[token*uint16Bytes : (token+1)*uint16Bytes])
		dst[token] = minimum + scale*float32(quantized)
	}
	return minimum, nil
}
