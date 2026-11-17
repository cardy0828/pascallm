package tokenizer

import "unicode"

// この file は、tokenizer.json の事前分割の正規表現（expectedSplitPattern）を手で再現します。
// Go の regexp は先読み（(?!\S)）を書けないため、7 個の選択肢を、正規表現と同じ順に試します。
//
//	1. (?i:'s|'t|'re|'ve|'m|'ll|'d)    英語の短縮形
//	2. [^\r\n\p{L}\p{N}]?\p{L}+       文字の並び（前に記号や空白を 1 個まで付けてよい）
//	3. \p{N}{1,3}                      数字 1〜3 個
//	4. ' '?[^\s\p{L}\p{N}]+[\r\n]*      記号の並び（前に空白 1 個、後ろに改行を付けてよい）
//	5. \s*[\r\n]+                      改行を含む空白の並び（最後の改行まで）
//	6. \s+(?!\S)                       空白の並び（後ろに空白以外が続くなら最後の 1 個を残す）
//	7. \s+                             空白の並び
//
// 正規表現の検索と同じく、ある位置で最初に一致した選択肢で 1 片を切り出し、その終わりから続けます。
// どの文字も 2・3・4・7 のどれかに一致するため、一致しない位置はありません。

// splitPieces は、文字列を事前分割の片に分けます。
// 例: "Hello world!\n\n" → ["Hello", " world", "!\n\n"]、"東京 2024年" → ["東京", " ", "202", "4", "年"]
func splitPieces(text string) []string {
	runes := []rune(text)
	var pieces []string
	for start := 0; start < len(runes); {
		end := matchPieceAt(runes, start)
		pieces = append(pieces, string(runes[start:end]))
		start = end
	}
	return pieces
}

// matchPieceAt は、位置 start から切り出す 1 片の終わり（含まない）を返します。
// 例: runes = "Hi there", start = 2 → 8（" there"）
func matchPieceAt(runes []rune, start int) int {
	if end, ok := matchContraction(runes, start); ok {
		return end
	}
	if end, ok := matchLetters(runes, start); ok {
		return end
	}
	if end, ok := matchDigits(runes, start); ok {
		return end
	}
	if end, ok := matchSymbols(runes, start); ok {
		return end
	}
	if end, ok := matchSpacesThroughLastNewline(runes, start); ok {
		return end
	}
	if end, ok := matchSpacesBeforeNonSpace(runes, start); ok {
		return end
	}
	if end, ok := matchSpaces(runes, start); ok {
		return end
	}
	// ここには来ません（どの文字も上のどれかに一致する）。念のため 1 文字を 1 片にします。
	return start + 1
}

// isLetter は \p{L}、isNumber は \p{N}、isSpace は \s（Unicode の White_Space。全角空白 U+3000 を含む）です。
// 例: isLetter('あ') → true、isNumber('３') → true、isSpace('　') → true
func isLetter(r rune) bool { return unicode.IsLetter(r) }

// isNumber は \p{N} です（isLetter の説明を参照）。
func isNumber(r rune) bool { return unicode.IsNumber(r) }

// isSpace は \s です（isLetter の説明を参照）。
func isSpace(r rune) bool { return unicode.IsSpace(r) }

// isLineBreak は [\r\n] です。例: isLineBreak('\n') → true、isLineBreak(' ') → false
func isLineBreak(r rune) bool { return r == '\r' || r == '\n' }

// isSymbol は [^\s\p{L}\p{N}]（空白・文字・数字以外）です。例: isSymbol('!') → true、isSymbol('、') → true
func isSymbol(r rune) bool { return !isSpace(r) && !isLetter(r) && !isNumber(r) }

// isLetterPrefix は [^\r\n\p{L}\p{N}]（選択肢 2 で文字の前に付けてよい 1 文字）です。
// 例: isLetterPrefix(' ') → true、isLetterPrefix('「') → true、isLetterPrefix('\n') → false
func isLetterPrefix(r rune) bool { return !isLineBreak(r) && !isLetter(r) && !isNumber(r) }

// runEnd は、位置 start から条件を満たす文字が続く終わり（含まない）を返します。
// 例: runEnd("abc1", 0, isLetter) → 3
func runEnd(runes []rune, start int, condition func(rune) bool) int {
	end := start
	for end < len(runes) && condition(runes[end]) {
		end++
	}
	return end
}

// matchContraction は選択肢 1（'s・'t・'re・'ve・'m・'ll・'d、大文字小文字を区別しない）です。
// 例: "'S good" の位置 0 → 2、"'ll" → 3、"'x" → 一致しない
func matchContraction(runes []rune, start int) (int, bool) {
	if runes[start] != '\'' {
		return 0, false
	}
	// 2 文字の候補（re・ve・ll）は 1 文字目が s・t・m・d と重ならないため、順番による違いはありません。
	singleLetters := []rune{'s', 't', 'm', 'd'}
	pairs := [][2]rune{{'r', 'e'}, {'v', 'e'}, {'l', 'l'}}
	next := start + 1
	for _, pair := range pairs {
		if next+1 < len(runes) && equalFoldASCII(runes[next], pair[0]) && equalFoldASCII(runes[next+1], pair[1]) {
			return next + 2, true
		}
	}
	for _, letter := range singleLetters {
		if next < len(runes) && equalFoldASCII(runes[next], letter) {
			return next + 1, true
		}
	}
	return 0, false
}

// equalFoldASCII は、r が小文字の英字 lower と大文字小文字を区別せずに等しいかを返します。
// Unicode の大文字小文字の対応で 's' に一致する 'ſ'（U+017F）も等しいとします。
// 例: equalFoldASCII('S', 's') → true、equalFoldASCII('ſ', 's') → true
func equalFoldASCII(r rune, lower rune) bool {
	if unicode.ToLower(r) == lower {
		return true
	}
	const longS = 'ſ'
	return lower == 's' && r == longS
}

// matchLetters は選択肢 2（[^\r\n\p{L}\p{N}]?\p{L}+）です。
// 例: " world" → 6、"「東京" → 3、"abc" → 3、"!" → 一致しない
func matchLetters(runes []rune, start int) (int, bool) {
	if isLetterPrefix(runes[start]) && start+1 < len(runes) && isLetter(runes[start+1]) {
		return runEnd(runes, start+1, isLetter), true
	}
	if isLetter(runes[start]) {
		return runEnd(runes, start, isLetter), true
	}
	return 0, false
}

// maxDigitsPerPiece は、選択肢 3 で 1 片にまとめる数字の最大数です（\p{N}{1,3}）。
const maxDigitsPerPiece = 3

// matchDigits は選択肢 3（\p{N}{1,3}）です。例: "12345" → 3、"7a" → 1
func matchDigits(runes []rune, start int) (int, bool) {
	if !isNumber(runes[start]) {
		return 0, false
	}
	end := start
	for end < len(runes) && end-start < maxDigitsPerPiece && isNumber(runes[end]) {
		end++
	}
	return end, true
}

// matchSymbols は選択肢 4（' '?[^\s\p{L}\p{N}]+[\r\n]*）です。
// 例: " !!\n\nx" → 5（" !!\n\n"）、"。" → 1、" a" → 一致しない
func matchSymbols(runes []rune, start int) (int, bool) {
	symbolStart := start
	if runes[start] == ' ' && start+1 < len(runes) && isSymbol(runes[start+1]) {
		symbolStart = start + 1
	}
	if !isSymbol(runes[symbolStart]) {
		return 0, false
	}
	end := runEnd(runes, symbolStart, isSymbol)
	return runEnd(runes, end, isLineBreak), true
}

// matchSpacesThroughLastNewline は選択肢 5（\s*[\r\n]+）です。空白の並びのうち、最後の改行までを 1 片にします。
// 例: "  \n \n  x" → 5（"  \n \n"）、"   x" → 一致しない（改行がない）
func matchSpacesThroughLastNewline(runes []rune, start int) (int, bool) {
	spacesEnd := runEnd(runes, start, isSpace)
	for end := spacesEnd; end > start; end-- {
		if isLineBreak(runes[end-1]) {
			return end, true
		}
	}
	return 0, false
}

// matchSpacesBeforeNonSpace は選択肢 6（\s+(?!\S)）です。空白の並びが文字列の終わりまで続けば全部、
// 後ろに空白以外が続くなら最後の 1 個を残した部分を 1 片にします（残した 1 個は次の単語の前に付く）。
// 例: "   x" → 2（"  "）、"  " で終わる → 2、" x" → 一致しない（残すと空になる）
func matchSpacesBeforeNonSpace(runes []rune, start int) (int, bool) {
	spacesEnd := runEnd(runes, start, isSpace)
	if spacesEnd == start {
		return 0, false
	}
	if spacesEnd == len(runes) {
		return spacesEnd, true
	}
	if spacesEnd-start >= 2 {
		return spacesEnd - 1, true
	}
	return 0, false
}

// matchSpaces は選択肢 7（\s+）です。例: " x" → 1
func matchSpaces(runes []rune, start int) (int, bool) {
	spacesEnd := runEnd(runes, start, isSpace)
	return spacesEnd, spacesEnd > start
}
