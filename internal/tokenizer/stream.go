package tokenizer

import (
	"fmt"
	"strings"
	"unicode/utf8"
)

// StreamDecoder は、生成中のトークンを 1 個ずつ文字列に戻します。
// 1 文字が複数のトークンに分かれることがある（日本語の 1 文字は 3 バイト）ため、
// 文字の途中で終わるバイトは次のトークンまで持ち越します。
type StreamDecoder struct {
	tokenizer *Tokenizer
	pending   []byte // まだ返していないバイト（文字の途中）
}

// NewStreamDecoder は StreamDecoder を作ります。例: decoder := tok.NewStreamDecoder()
func (t *Tokenizer) NewStreamDecoder() *StreamDecoder {
	return &StreamDecoder{tokenizer: t}
}

// Push はトークンを 1 個加え、文字として完成した部分を返します（なければ ""）。
// 特殊トークンも、その文字列（例: "<|im_end|>"）として加えます。取り除くかどうかは呼び出し側で決めます。
// 例: 「日」が 2 トークン（2 バイト + 1 バイト）なら、1 回目は ""、2 回目は "日" を返す
func (d *StreamDecoder) Push(id int) (string, error) {
	if !d.tokenizer.Known(id) {
		return "", fmt.Errorf("tokenizer: unknown token %d", id)
	}
	d.pending = append(d.pending, d.tokenizer.tokenBytes[id]...)
	complete := completeUTF8Length(d.pending)
	text := strings.ToValidUTF8(string(d.pending[:complete]), "�")
	d.pending = append(d.pending[:0], d.pending[complete:]...)
	return text, nil
}

// Flush は持ち越していたバイトを返し、空にします。文字の途中のバイトは U+FFFD になります。
// 例: 生成が文字の途中で終わったとき、最後に呼ぶ
func (d *StreamDecoder) Flush() string {
	text := strings.ToValidUTF8(string(d.pending), "�")
	d.pending = d.pending[:0]
	return text
}

// completeUTF8Length は、バイト列のうち、最後の文字が途中で切れていない部分の長さを返します。
// 末尾が「正しい先頭バイトで始まり、まだ足りない」文字なら、その先頭の手前までを返します。
// 例: "あ" の 3 バイトのうち 2 バイト → 0、"aあ"（4 バイト）→ 4、不正なバイト 0xFF → 1（持ち越さない）
func completeUTF8Length(data []byte) int {
	for start := len(data) - 1; start >= 0 && start >= len(data)-utf8.UTFMax; start-- {
		if !utf8.RuneStart(data[start]) {
			continue
		}
		if utf8.FullRune(data[start:]) {
			return len(data)
		}
		return start
	}
	return len(data)
}
