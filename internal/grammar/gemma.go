package grammar

import (
	"fmt"
	"github.com/cardy0828/pascallm/internal/chat"
	"github.com/cardy0828/pascallm/internal/pyjson"
)

// GemmaToolCalls は <|tool_call>call:f{x:<|"|>値<|"|>}<tool_call|> の文法を作ります。
// 既知の引数は Schema の順で、required・型・enum・配列などの制約を JSON と同じ規則で扱います。
func GemmaToolCalls(functions []pyjson.Value, start, end, quote int, multiple bool) (*Grammar, error) {
	if len(functions) == 0 {
		return nil, fmt.Errorf("grammar: no Gemma functions")
	}
	b := &Builder{}
	var alternatives [][]Element
	for _, function := range functions {
		name, _ := function.GetString("name")
		if !chat.GemmaIdentifier(name) {
			return nil, fmt.Errorf("grammar: invalid Gemma function name")
		}
		params, ok := function.Get("parameters")
		if !ok || params.Kind == pyjson.Null {
			params = pyjson.Value{Kind: pyjson.Object, Members: []pyjson.Member{{Key: "type", Value: pyjson.NewString("object")}, {Key: "additionalProperties", Value: pyjson.Value{Kind: pyjson.Bool}}}}
		}
		if err := checkGemmaSchemaKeys(params); err != nil {
			return nil, err
		}
		c := &jsonCompiler{builder: b, root: params, refs: map[string]int{}, gemma: true, quoteToken: quote}
		c.whitespace = buildWhitespace(b)
		var plain ByteSet
		plain.AddRange(0, 0xff)
		chars := b.Repeat("gemma-string-chars", Bytes(plain), 0, -1)
		c.stringRule = b.Rule("gemma-string", Seq(One(Token(quote)), One(Ref(chars)), One(Token(quote))))
		c.integerRule = c.buildInteger()
		c.numberRule = c.buildNumber()
		c.anyValue = c.buildAnyValue()
		arguments, err := c.compile(params)
		if err != nil {
			return nil, err
		}
		alternatives = append(alternatives, Seq(One(Token(start)), Literal("call:"+name), One(Ref(arguments)), One(Token(end))))
	}
	call := b.Rule("gemma-call", alternatives...)
	calls := call
	if multiple {
		calls = b.NewRule("gemma-calls")
		b.SetAlternatives(calls, One(Ref(call)), Seq(One(Ref(call)), One(Ref(calls))))
	}
	return b.Build(calls), nil
}

// checkGemmaSchemaKeys は裸のキーにできない名前を拒否します。$defs なども再帰して確認します。
func checkGemmaSchemaKeys(value pyjson.Value) error {
	if value.Kind == pyjson.Object {
		for _, m := range value.Members {
			if m.Key == "properties" {
				for _, prop := range m.Value.Members {
					if !chat.GemmaIdentifier(prop.Key) {
						return fmt.Errorf("grammar: unsupported Gemma key %q", prop.Key)
					}
				}
			}
			if err := checkGemmaSchemaKeys(m.Value); err != nil {
				return err
			}
		}
	} else if value.Kind == pyjson.Array {
		for _, item := range value.Items {
			if err := checkGemmaSchemaKeys(item); err != nil {
				return err
			}
		}
	}
	return nil
}

// gemmaLiteral は const・enum を DSL の要素にします。文字列内の特殊タグは通常のバイトとして扱います。
func gemmaLiteral(value pyjson.Value, quote int) []Element {
	switch value.Kind {
	case pyjson.String:
		return Seq(One(Token(quote)), Literal(value.String), One(Token(quote)))
	case pyjson.Array:
		result := Literal("[")
		for i, v := range value.Items {
			if i > 0 {
				result = append(result, Byte(','))
			}
			result = append(result, gemmaLiteral(v, quote)...)
		}
		return append(result, Byte(']'))
	case pyjson.Object:
		result := Literal("{")
		for i, m := range value.Members {
			if i > 0 {
				result = append(result, Byte(','))
			}
			result = append(result, Literal(m.Key+":")...)
			result = append(result, gemmaLiteral(m.Value, quote)...)
		}
		return append(result, Byte('}'))
	default:
		return Literal(value.Dumps())
	}
}
