package tokenizer

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
)

// IsGemma は Gemma 4 の Unicode BPE かを返します。テンプレートの選択と設定検査で使います。
func (t *Tokenizer) IsGemma() bool { return t.gemma }

// checkTokenizerFormat は LFM3 または確認済みの Gemma 4 の構成だけを許します。
// normalizer や decoder が異なる入力を黙って処理しないため、Gemma は各段階も比較します。
func checkTokenizerFormat(file *fileFormat, data []byte) error {
	if !file.Model.ByteFallback {
		return checkSupported(file)
	}
	if file.Model.Type != "BPE" || file.Model.IgnoreMerges {
		return fmt.Errorf("tokenizer: unsupported Gemma BPE options")
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	expected := map[string]string{
		"normalizer":    `{"type":"Replace","pattern":{"String":" "},"content":"▁"}`,
		"pre_tokenizer": `{"type":"Split","pattern":{"String":" "},"behavior":"MergedWithPrevious","invert":false}`,
		"decoder":       `{"type":"Sequence","decoders":[{"type":"Replace","pattern":{"String":"▁"},"content":" "},{"type":"ByteFallback"},{"type":"Fuse"}]}`,
	}
	for name, want := range expected {
		var a, b any
		if err := json.Unmarshal(raw[name], &a); err != nil {
			return err
		}
		if err := json.Unmarshal([]byte(want), &b); err != nil {
			return err
		}
		encodedA, _ := json.Marshal(a)
		encodedB, _ := json.Marshal(b)
		if string(encodedA) != string(encodedB) {
			return fmt.Errorf("tokenizer: unsupported Gemma %s", name)
		}
	}
	for _, token := range file.AddedTokens {
		if token.Content == "" || token.SingleWord || token.Lstrip || token.Rstrip || token.Normalized {
			return fmt.Errorf("tokenizer: unsupported added token %q", token.Content)
		}
	}
	return nil
}

// gemmaTokenBytes は語彙の表記を復元時のバイト列にします。例: ▁東京 → 空白+東京、<0xE3> → {0xe3}。
func gemmaTokenBytes(text string) ([]byte, error) {
	if len(text) == 6 && strings.HasPrefix(text, "<0x") && strings.HasSuffix(text, ">") {
		value, err := strconv.ParseUint(text[3:5], 16, 8)
		if err != nil {
			return nil, err
		}
		return []byte{byte(value)}, nil
	}
	return []byte(strings.ReplaceAll(text, "▁", " ")), nil
}
