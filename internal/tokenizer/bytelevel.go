package tokenizer

import (
	"fmt"
	"strings"
)

// byteCount は 1 バイトがとりうる値の数です。
const byteCount = 256

// byte レベルの対応表で「そのままの文字」になるバイトの範囲です（GPT-2 の bytes_to_unicode と同じ）。
// これ以外のバイト（制御文字・空白など）は、U+0100 から順に割り当てた文字にします。
const (
	printableASCIIFirst    = '!' // 0x21
	printableASCIILast     = '~' // 0x7E
	latin1FirstBlockFirst  = '¡' // 0xA1
	latin1FirstBlockLast   = '¬' // 0xAC
	latin1SecondBlockFirst = '®' // 0xAE
	latin1SecondBlockLast  = 'ÿ' // 0xFF
	// shiftedRuneBase は、そのままにしないバイトに割り当てる文字の最初の値です（U+0100 = 'Ā'）。
	shiftedRuneBase = 0x100
)

// byteToRune はバイト → 文字、runeToByte は文字 → バイトの対応です。例: 0x20（空白）→ 'Ġ'（U+0120）
var (
	byteToRune [byteCount]rune
	runeToByte = make(map[rune]byte, byteCount)
)

// init は byteToRune と runeToByte を作ります。そのままにしないバイトには、バイトの小さい順に
// U+0100 から文字を割り当てます。例: 0x00 → 'Ā'（U+0100）、0x20 → 'Ġ'（U+0120）、0x0A → 'Ċ'（U+010A）
func init() {
	shifted := 0
	for value := 0; value < byteCount; value++ {
		if keepsOwnCharacter(value) {
			byteToRune[value] = rune(value)
		} else {
			byteToRune[value] = rune(shiftedRuneBase + shifted)
			shifted++
		}
		runeToByte[byteToRune[value]] = byte(value)
	}
}

// keepsOwnCharacter は、そのバイトが byte レベルの対応表で同じ値の文字になるかを返します。
// 例: keepsOwnCharacter('A') → true、keepsOwnCharacter(' ') → false、keepsOwnCharacter(0xAD) → false
func keepsOwnCharacter(value int) bool {
	inPrintableASCII := value >= printableASCIIFirst && value <= printableASCIILast
	inLatin1First := value >= latin1FirstBlockFirst && value <= latin1FirstBlockLast
	inLatin1Second := value >= latin1SecondBlockFirst && value <= latin1SecondBlockLast
	return inPrintableASCII || inLatin1First || inLatin1Second
}

// byteLevelEncode は、文字列の各バイトを対応表の文字に置き換えます。例: " 日" → "Ġæĺ¥"
func byteLevelEncode(text string) string {
	var builder strings.Builder
	builder.Grow(2 * len(text))
	for index := 0; index < len(text); index++ {
		builder.WriteRune(byteToRune[text[index]])
	}
	return builder.String()
}

// byteLevelDecode は byteLevelEncode の逆で、対応表の文字の並びをバイト列に戻します。
// 例: "Ġhello" → []byte(" hello")。表にない文字があればエラーです。
func byteLevelDecode(text string) ([]byte, error) {
	decoded := make([]byte, 0, len(text))
	for _, character := range text {
		value, ok := runeToByte[character]
		if !ok {
			return nil, fmt.Errorf("character %U is not a byte-level character", character)
		}
		decoded = append(decoded, value)
	}
	return decoded, nil
}
