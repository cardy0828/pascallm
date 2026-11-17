package tokenizer

import (
	"slices"
	"testing"
)

// TestSplitPieces は、事前分割の 7 個の選択肢の境目を例で確かめます
// （tokenizer.json 全体との一致は cmd/tokenizer-check で確かめる）。
func TestSplitPieces(t *testing.T) {
	cases := []struct {
		text string
		want []string
	}{
		{"Hello world!\n\n", []string{"Hello", " world", "!\n\n"}},
		{"東京 2024年", []string{"東京", " ", "202", "4", "年"}},
		{"I'm OK'S 'x", []string{"I", "'m", " OK", "'S", " '", "x"}},
		{"   x", []string{"  ", " x"}},
		{"a  \n \n  b", []string{"a", "  \n \n", " ", " b"}},
		{"end  ", []string{"end", "  "}},
		{"「東京」", []string{"「東京", "」"}},
		{"x　y", []string{"x", "　y"}},
		{" !!\n\nx", []string{" !!\n\n", "x"}},
	}
	for _, c := range cases {
		if got := splitPieces(c.text); !slices.Equal(got, c.want) {
			t.Errorf("splitPieces(%q) = %q, want %q", c.text, got, c.want)
		}
	}
}

// TestByteLevelRoundTrip は、byte レベルの対応が 256 通りすべてで元に戻るかを確かめます。
func TestByteLevelRoundTrip(t *testing.T) {
	all := make([]byte, byteCount)
	for value := range all {
		all[value] = byte(value)
	}
	decoded, err := byteLevelDecode(byteLevelEncode(string(all)))
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(decoded, all) {
		t.Fatal("byte-level round trip changed the bytes")
	}
	if byteToRune[' '] != 'Ġ' || byteToRune['\n'] != 'Ċ' {
		t.Fatalf("space → %q, newline → %q", byteToRune[' '], byteToRune['\n'])
	}
}

// TestCompleteUTF8Length は、文字の途中のバイトを持ち越す長さを確かめます。
func TestCompleteUTF8Length(t *testing.T) {
	hiragana := []byte("あ") // 3 バイト
	cases := []struct {
		data []byte
		want int
	}{
		{hiragana[:2], 0},
		{append([]byte("a"), hiragana...), 4},
		{append([]byte("a"), hiragana[:1]...), 1},
		{[]byte{0xFF}, 1},
		{nil, 0},
	}
	for _, c := range cases {
		if got := completeUTF8Length(c.data); got != c.want {
			t.Errorf("completeUTF8Length(%v) = %d, want %d", c.data, got, c.want)
		}
	}
}
