// Package tokenizer は、Hugging Face の tokenizer.json（byte レベルの BPE）を Go だけで扱います（M3）。
//
// LFM3 と Gemma 4 の確認済みの構成に対応します。LFM3 の構成は次のとおりです。
//   - normalizer なし
//   - 事前分割: 決まった正規表現での分割（pretokenize.go）→ ByteLevel（add_prefix_space なし）
//   - モデル: BPE（ignore_merges あり、byte_fallback なし）
//   - 追加トークン: lstrip・rstrip・single_word・normalized がすべて false
//
// Gemma 4 は空白の ▁ への置換、Unicode BPE（ignore_merges なし）、byte fallback です（gemma.go）。
// 構成が違う tokenizer.json は Load でエラーにします（黙って違う分割をしないため）。
//
// 特殊トークン（"<|im_start|>"・"<|im_end|>" など special が true の追加トークン）は、Segment.Control が true の
// 区間でだけ 1 トークンとして認識します。利用者の本文は Control を false にして渡し、本文中の "<|im_end|>" などで
// 会話の区切りを偽装できないようにします。特殊ではない追加トークン（"<think>"・"<tool_call>" など）は、
// Hugging Face と同じく、どの区間でも認識します（split_special_tokens=True でも認識されるため）。
package tokenizer

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sort"
	"strings"
	"sync"
)

// expectedSplitPattern は、pretokenize.go が手で再現している事前分割の正規表現です。
// tokenizer.json の値がこれと違えば、Load はエラーにします。
const expectedSplitPattern = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`

// pieceCacheLimit は、事前分割の 1 片 → トークン列 のキャッシュに置く最大の件数です。
// 超えたらキャッシュを空にします（会話の文章では同じ片が何度も出るため、それで十分です）。
const pieceCacheLimit = 200000

// Segment は Encode に渡す文字列の 1 区間です。
type Segment struct {
	Text string // 文字列。例: "<|im_start|>user\n" や、利用者の本文
	// Control が true の区間では、追加トークンの文字列（"<|im_start|>" など）を 1 トークンとして認識します。
	// チャットテンプレートが書いた部分だけを true にし、利用者の本文は false にします。
	Control bool
}

// addedToken は追加トークン 1 個です。例: {id: 124899, content: "<|im_start|>", special: true}
type addedToken struct {
	id      int
	content string
	special bool // true なら、復元（Decode の skipSpecial）で取り除く対象
}

// mergeKey は BPE のマージ規則 1 個の左右の組です。例: {"Ġ", "t"}
type mergeKey struct {
	left, right string
}

// Tokenizer は読み込み済みのトークナイザーです。Encode・Decode は複数のゴルーチンから同時に呼べます。
type Tokenizer struct {
	gemma      bool             // 空白置換・Unicode BPE・byte fallback を使う Gemma 4 か
	vocab      map[string]int   // byte レベルの文字列 → 番号。例: "Ġthe" → 279
	mergeRanks map[mergeKey]int // マージ規則 → 優先順位（小さいほど先にマージする）
	added      []addedToken     // 追加トークン（長い順。最長一致で探すため）
	// addedFirstBytes[b] は、b で始まる追加トークンがあるかです（findAdded で、関係ない位置を飛ばすため）。
	addedFirstBytes [byteCount]bool
	tokenBytes      [][]byte // 番号 → 復元したときのバイト列
	special         []bool   // 番号 → 特殊トークンか
	isAdded         []bool   // 番号 → 追加トークン（特殊でないものを含む）か

	cacheMutex sync.Mutex
	pieceCache map[string][]int // 事前分割の 1 片 → トークン列
}

// fileAddedToken は tokenizer.json の added_tokens の 1 要素です。
type fileAddedToken struct {
	ID         int    `json:"id"`
	Content    string `json:"content"`
	Special    bool   `json:"special"`
	SingleWord bool   `json:"single_word"`
	Lstrip     bool   `json:"lstrip"`
	Rstrip     bool   `json:"rstrip"`
	Normalized bool   `json:"normalized"`
}

// fileFormat は tokenizer.json のうち、読む部分です。
type fileFormat struct {
	AddedTokens  []fileAddedToken `json:"added_tokens"`
	Normalizer   json.RawMessage  `json:"normalizer"`
	PreTokenizer struct {
		Type          string `json:"type"`
		Pretokenizers []struct {
			Type    string `json:"type"`
			Pattern struct {
				Regex string `json:"Regex"`
			} `json:"pattern"`
			Behavior       string `json:"behavior"`
			Invert         bool   `json:"invert"`
			AddPrefixSpace bool   `json:"add_prefix_space"`
			UseRegex       bool   `json:"use_regex"`
		} `json:"pretokenizers"`
	} `json:"pre_tokenizer"`
	Model struct {
		Type         string          `json:"type"`
		IgnoreMerges bool            `json:"ignore_merges"`
		ByteFallback bool            `json:"byte_fallback"`
		Vocab        map[string]int  `json:"vocab"`
		Merges       json.RawMessage `json:"merges"`
	} `json:"model"`
}

// Load は tokenizer.json を読みます。例: Load("./models/lfm3-1.7b-training/tokenizer.json")
func Load(path string) (*Tokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var file fileFormat
	if err := json.Unmarshal(data, &file); err != nil {
		return nil, fmt.Errorf("tokenizer: %s: %w", path, err)
	}
	if err := checkTokenizerFormat(&file, data); err != nil {
		return nil, fmt.Errorf("tokenizer: %s: %w", path, err)
	}
	merges, err := parseMerges(file.Model.Merges)
	if err != nil {
		return nil, fmt.Errorf("tokenizer: %s: %w", path, err)
	}
	return build(&file, merges)
}

// checkSupported は、tokenizer.json の構成がこの実装の対応する形かを確かめます。
func checkSupported(file *fileFormat) error {
	if normalizer := strings.TrimSpace(string(file.Normalizer)); normalizer != "" && normalizer != "null" {
		return errors.New("normalizer is not supported")
	}
	model := file.Model
	if model.Type != "BPE" || !model.IgnoreMerges || model.ByteFallback {
		return fmt.Errorf("model must be BPE with ignore_merges and without byte_fallback (type=%s ignore_merges=%v byte_fallback=%v)",
			model.Type, model.IgnoreMerges, model.ByteFallback)
	}
	steps := file.PreTokenizer.Pretokenizers
	if file.PreTokenizer.Type != "Sequence" || len(steps) != 2 {
		return errors.New("pre_tokenizer must be Sequence[Split, ByteLevel]")
	}
	split, byteLevel := steps[0], steps[1]
	if split.Type != "Split" || split.Pattern.Regex != expectedSplitPattern || split.Behavior != "Isolated" || split.Invert {
		return errors.New("pre_tokenizer Split differs from the supported pattern")
	}
	if byteLevel.Type != "ByteLevel" || byteLevel.AddPrefixSpace || byteLevel.UseRegex {
		return errors.New("pre_tokenizer ByteLevel must have add_prefix_space=false and use_regex=false")
	}
	for _, token := range file.AddedTokens {
		if token.SingleWord || token.Lstrip || token.Rstrip || token.Normalized {
			return fmt.Errorf("added token %q uses unsupported options", token.Content)
		}
		if token.Content == "" {
			return fmt.Errorf("added token %d is empty", token.ID)
		}
	}
	return nil
}

// parseMerges は merges を読みます。新しい形式（[["a", "b"], ...]）と古い形式（["a b", ...]）の両方に対応します。
// 例: [["Ġ", "t"], ["Ġ", "a"]] → [{Ġ t} {Ġ a}]
func parseMerges(raw json.RawMessage) ([]mergeKey, error) {
	var pairs [][]string
	if err := json.Unmarshal(raw, &pairs); err == nil {
		merges := make([]mergeKey, len(pairs))
		for i, pair := range pairs {
			if len(pair) != 2 {
				return nil, fmt.Errorf("merge %d has %d parts", i, len(pair))
			}
			merges[i] = mergeKey{left: pair[0], right: pair[1]}
		}
		return merges, nil
	}
	var lines []string
	if err := json.Unmarshal(raw, &lines); err != nil {
		return nil, fmt.Errorf("merges: %w", err)
	}
	merges := make([]mergeKey, len(lines))
	for i, line := range lines {
		left, right, ok := strings.Cut(line, " ")
		if !ok {
			return nil, fmt.Errorf("merge %d %q has no space", i, line)
		}
		merges[i] = mergeKey{left: left, right: right}
	}
	return merges, nil
}

// build は、読んだ内容から番号の表・マージの優先順位・追加トークンの一覧を作ります。
func build(file *fileFormat, merges []mergeKey) (*Tokenizer, error) {
	size := 0
	for _, id := range file.Model.Vocab {
		size = max(size, id+1)
	}
	for _, token := range file.AddedTokens {
		size = max(size, token.ID+1)
	}
	t := &Tokenizer{
		gemma:      file.Model.ByteFallback,
		vocab:      file.Model.Vocab,
		mergeRanks: make(map[mergeKey]int, len(merges)),
		tokenBytes: make([][]byte, size),
		special:    make([]bool, size),
		isAdded:    make([]bool, size),
		pieceCache: make(map[string][]int),
	}
	for rank, merge := range merges {
		if _, exists := t.mergeRanks[merge]; !exists {
			t.mergeRanks[merge] = rank
		}
	}
	for text, id := range file.Model.Vocab {
		var decoded []byte
		var err error
		if t.gemma {
			decoded, err = gemmaTokenBytes(text)
		} else {
			decoded, err = byteLevelDecode(text)
		}
		if err != nil {
			return nil, fmt.Errorf("tokenizer: vocab %d %q: %w", id, text, err)
		}
		t.tokenBytes[id] = decoded
	}
	for _, token := range file.AddedTokens {
		t.added = append(t.added, addedToken{id: token.ID, content: token.Content, special: token.Special})
		t.tokenBytes[token.ID] = []byte(token.Content)
		t.special[token.ID] = token.Special
		t.isAdded[token.ID] = true
	}
	// 最長一致で探すため、長い順に並べます（同じ長さなら番号順）。
	sort.Slice(t.added, func(i, j int) bool {
		if len(t.added[i].content) != len(t.added[j].content) {
			return len(t.added[i].content) > len(t.added[j].content)
		}
		return t.added[i].id < t.added[j].id
	})
	for _, token := range t.added {
		t.addedFirstBytes[token.content[0]] = true
	}
	// byte_fallback がないため、どの 1 バイトも語彙にある必要があります（ないと分割できない文字が出る）。
	for value := 0; value < byteCount; value++ {
		key := string(byteToRune[value])
		if t.gemma {
			key = fmt.Sprintf("<0x%02X>", value)
		}
		if _, ok := t.vocab[key]; !ok {
			return nil, fmt.Errorf("tokenizer: vocab lacks the single byte %d", value)
		}
	}
	return t, nil
}

// Size は番号の数（最大の番号 + 1）を返します。例: LFM3 では 125033
// モデルの語彙（128,000）より小さく、Size 以上の番号はトークナイザーにありません。
func (t *Tokenizer) Size() int { return len(t.tokenBytes) }

// TokenID は、追加トークンまたは語彙の文字列の番号を返します。例: TokenID("<|im_end|>") → 124900, true
func (t *Tokenizer) TokenID(content string) (int, bool) {
	for _, token := range t.added {
		if token.content == content {
			return token.id, true
		}
	}
	id, ok := t.vocab[content]
	return id, ok
}

// IsSpecial は、番号が特殊トークン（復元のときに取り除く対象）かを返します。例: IsSpecial(124900) → true
func (t *Tokenizer) IsSpecial(id int) bool {
	return id >= 0 && id < len(t.special) && t.special[id]
}

// IsAdded は、番号が追加トークン（<|im_end|> などの特殊トークンと、<think>・<tool_call> などの特殊でないもの）かを
// 返します。例: IsAdded(125019) → true（<tool_call>）、IsAdded(279) → false
func (t *Tokenizer) IsAdded(id int) bool {
	return id >= 0 && id < len(t.isAdded) && t.isAdded[id]
}

// TokenBytes は、番号のトークンを復元したときのバイト列を返します（追加トークンはその文字列）。
// ない番号なら nil です。返した値は書き換えないでください。例: TokenBytes(279) → []byte(" the")
func (t *Tokenizer) TokenBytes(id int) []byte {
	if !t.Known(id) {
		return nil
	}
	return t.tokenBytes[id]
}

// Known は、番号がトークナイザーにあるかを返します。例: Known(127999) → false（モデルの語彙の余り）
func (t *Tokenizer) Known(id int) bool {
	return id >= 0 && id < len(t.tokenBytes) && t.tokenBytes[id] != nil
}

// EncodeText は、特殊トークンを認識せずに（特殊ではない追加トークンは認識して）文字列をトークン列にします。
// Hugging Face の tokenizer(text, add_special_tokens=False, split_special_tokens=True) と同じ結果です。
// 例: EncodeText("こんにちは") → [ ... ]
func (t *Tokenizer) EncodeText(text string) ([]int, error) {
	return t.Encode(Segment{Text: text})
}

// Encode は、区間の並びをトークン列にします。特殊トークンは Control の区間でだけ、特殊ではない追加トークンは
// どの区間でも認識します。追加トークン以外の部分は、区間の境目をまたいでつなげてから分割します
// （全体を 1 つの文字列として Hugging Face で分割した結果と同じになるようにするため）。
// 例: Encode(Segment{"<|im_start|>user\n", true}, Segment{"こんにちは", false}) → [124899, ...]
func (t *Tokenizer) Encode(segments ...Segment) ([]int, error) {
	var ids []int
	var plain strings.Builder
	flush := func() error {
		if plain.Len() == 0 {
			return nil
		}
		encoded, err := t.encodePlain(plain.String())
		plain.Reset()
		ids = append(ids, encoded...)
		return err
	}
	for _, segment := range segments {
		rest := segment.Text
		for rest != "" {
			index, token, found := t.findAdded(rest, segment.Control)
			if !found {
				plain.WriteString(rest)
				break
			}
			plain.WriteString(rest[:index])
			if err := flush(); err != nil {
				return nil, err
			}
			ids = append(ids, token.id)
			rest = rest[index+len(token.content):]
		}
	}
	if err := flush(); err != nil {
		return nil, err
	}
	return ids, nil
}

// findAdded は、text の中で最初に現れる追加トークンを探します（同じ位置では最も長いもの）。
// includeSpecial が false なら特殊トークンは探しません（利用者の本文の区間）。
// 例: findAdded("ab<|im_end|>\n", true) → 2, <|im_end|>, true、findAdded("ab<|im_end|>", false) → 見つからない
func (t *Tokenizer) findAdded(text string, includeSpecial bool) (int, addedToken, bool) {
	for index := 0; index < len(text); index++ {
		if !t.addedFirstBytes[text[index]] {
			continue
		}
		for _, token := range t.added {
			if token.special && !includeSpecial {
				continue
			}
			if strings.HasPrefix(text[index:], token.content) {
				return index, token, true
			}
		}
	}
	return 0, addedToken{}, false
}

// encodePlain は、追加トークンを含まない文字列を、事前分割 → byte レベル → BPE の順にトークン列にします。
func (t *Tokenizer) encodePlain(text string) ([]int, error) {
	if t.gemma {
		return t.encodePiece(strings.ReplaceAll(text, " ", "▁"))
	}
	var ids []int
	for _, piece := range splitPieces(text) {
		encoded, err := t.encodePiece(piece)
		if err != nil {
			return nil, err
		}
		ids = append(ids, encoded...)
	}
	return ids, nil
}

// encodePiece は、事前分割の 1 片をトークン列にします（キャッシュを使います）。例: " world" → [1917]
func (t *Tokenizer) encodePiece(piece string) ([]int, error) {
	t.cacheMutex.Lock()
	cached, ok := t.pieceCache[piece]
	t.cacheMutex.Unlock()
	if ok {
		return cached, nil
	}
	word := piece
	if !t.gemma {
		word = byteLevelEncode(piece)
	}
	ids, err := t.bpe(word)
	if err != nil {
		return nil, err
	}
	t.cacheMutex.Lock()
	if len(t.pieceCache) >= pieceCacheLimit {
		t.pieceCache = make(map[string][]int)
	}
	t.pieceCache[piece] = ids
	t.cacheMutex.Unlock()
	return ids, nil
}

// Decode は、トークン列を文字列に戻します。skipSpecial が true なら特殊トークンを取り除きます。
// 不正な UTF-8 は U+FFFD に置き換えます（Hugging Face の decode と同じ）。
// 例: Decode([]int{124899, 882}, false) → "<|im_start|>user"
func (t *Tokenizer) Decode(ids []int, skipSpecial bool) (string, error) {
	var buffer []byte
	for _, id := range ids {
		if !t.Known(id) {
			return "", fmt.Errorf("tokenizer: unknown token %d", id)
		}
		if skipSpecial && t.special[id] {
			continue
		}
		buffer = append(buffer, t.tokenBytes[id]...)
	}
	return strings.ToValidUTF8(string(buffer), "�"), nil
}
