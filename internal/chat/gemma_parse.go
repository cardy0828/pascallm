package chat

import (
	"fmt"
	"github.com/cardy0828/pascallm/internal/pyjson"
	"strings"
)

// ParseGemmaCall は call:関数名{引数:値} を OpenAI で返せる値にします。
// 文字列は <|"|> で囲まれ、内部の引用符・改行・バックスラッシュはそのまま保持します。
func ParseGemmaCall(text string, tools *ToolSet) (ParsedToolCall, bool) {
	rest, ok := strings.CutPrefix(strings.TrimSpace(text), "call:")
	if !ok {
		return ParsedToolCall{}, false
	}
	brace := strings.IndexByte(rest, '{')
	if brace < 1 {
		return ParsedToolCall{}, false
	}
	name := strings.TrimSpace(rest[:brace])
	if !GemmaIdentifier(name) {
		return ParsedToolCall{}, false
	}
	if _, ok := tools.Function(name); !ok {
		return ParsedToolCall{}, false
	}
	p := gemmaValueParser{rest: rest[brace:]}
	value, err := p.value(0)
	if err != nil || strings.TrimSpace(p.rest) != "" || value.Kind != pyjson.Object {
		return ParsedToolCall{}, false
	}
	return ParsedToolCall{Name: name, Arguments: value}, true
}

// maxGemmaValueDepth は生成した DSL の入れ子を読む上限です。
const maxGemmaValueDepth = 64

// gemmaValueParser はツール DSL の値だけを読む小さな再帰下降パーサーです。
type gemmaValueParser struct{ rest string }

// value は現在位置の値を読みます。再帰を64段までに制限し、壊れた生成でスタックを使い切らないようにします。
func (p *gemmaValueParser) value(depth int) (pyjson.Value, error) {
	if depth > maxGemmaValueDepth {
		return pyjson.Value{}, fmt.Errorf("Gemma value is nested too deeply")
	}
	p.rest = strings.TrimLeft(p.rest, " \t\r\n")
	if strings.HasPrefix(p.rest, GemmaQuote) {
		text, after, ok := strings.Cut(p.rest[len(GemmaQuote):], GemmaQuote)
		if !ok {
			return pyjson.Value{}, fmt.Errorf("unclosed Gemma string")
		}
		p.rest = after
		return pyjson.NewString(text), nil
	}
	if p.rest == "" {
		return pyjson.Value{}, fmt.Errorf("missing value")
	}
	switch p.rest[0] {
	case '{':
		p.rest = p.rest[1:]
		result := pyjson.Value{Kind: pyjson.Object}
		seen := map[string]bool{}
		for {
			p.rest = strings.TrimLeft(p.rest, " \t\r\n")
			if strings.HasPrefix(p.rest, "}") {
				p.rest = p.rest[1:]
				return result, nil
			}
			var key string
			if strings.HasPrefix(p.rest, GemmaQuote) {
				v, err := p.value(depth + 1)
				if err != nil {
					return result, err
				}
				key = v.String
				p.rest = strings.TrimLeft(p.rest, " \t\r\n")
				if !strings.HasPrefix(p.rest, ":") {
					return result, fmt.Errorf("missing colon")
				}
				p.rest = p.rest[1:]
			} else {
				before, after, ok := strings.Cut(p.rest, ":")
				if !ok || !GemmaIdentifier(strings.TrimSpace(before)) {
					return result, fmt.Errorf("invalid key")
				}
				key = strings.TrimSpace(before)
				p.rest = after
			}
			if seen[key] {
				return result, fmt.Errorf("duplicate key")
			}
			seen[key] = true
			v, err := p.value(depth + 1)
			if err != nil {
				return result, err
			}
			result.Members = append(result.Members, pyjson.Member{Key: key, Value: v})
			if err := p.separator('}'); err != nil {
				return result, err
			}
		}
	case '[':
		p.rest = p.rest[1:]
		result := pyjson.Value{Kind: pyjson.Array}
		for {
			p.rest = strings.TrimLeft(p.rest, " \t\r\n")
			if strings.HasPrefix(p.rest, "]") {
				p.rest = p.rest[1:]
				return result, nil
			}
			v, err := p.value(depth + 1)
			if err != nil {
				return result, err
			}
			result.Items = append(result.Items, v)
			if err := p.separator(']'); err != nil {
				return result, err
			}
		}
	default:
		end := strings.IndexAny(p.rest, ",}] \t\r\n")
		if end < 0 {
			end = len(p.rest)
		}
		v, err := pyjson.Parse([]byte(p.rest[:end]))
		p.rest = p.rest[end:]
		if err != nil || (v.Kind != pyjson.Null && v.Kind != pyjson.Bool && v.Kind != pyjson.Number) {
			return pyjson.Value{}, fmt.Errorf("invalid scalar")
		}
		return v, nil
	}
}

// separator は次の要素の前のコンマを読みます。閉じ括弧は次の value のループに残します。
func (p *gemmaValueParser) separator(close byte) error {
	p.rest = strings.TrimLeft(p.rest, " \t\r\n")
	if len(p.rest) > 0 && p.rest[0] == close {
		return nil
	}
	if !strings.HasPrefix(p.rest, ",") {
		return fmt.Errorf("missing separator")
	}
	p.rest = strings.TrimLeft(p.rest[1:], " \t\r\n")
	if p.rest == "" || p.rest[0] == close {
		return fmt.Errorf("trailing comma")
	}
	return nil
}
