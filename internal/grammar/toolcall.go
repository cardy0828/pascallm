package grammar

import (
	"fmt"

	"github.com/cardy0828/pascallm/internal/pyjson"
)

// この file は、ツールの呼び出し（chat_template.jinja の render_tool_calls の形）の文法を作ります。
//
//	<tool_call>\n<function=名前>\n<parameter=引数名>\n値\n</parameter>\n…</function>\n</tool_call>
//
// <tool_call> などのタグは追加トークンで、トークンの要素として照合します。
//   - 関数の名前は tools にあるものだけ、引数名は parameters.properties にあるものだけです。
//   - 引数は properties の順に並べ、required の引数は省けません。
//   - 値は、Schema が文字列を許す（または型がわからない）なら任意の文字列、enum・const ならその値、
//     それ以外は Schema に合う JSON です（テンプレートと同じく、文字列は引用符を付けずに書く）。

// ToolTokens は、ツールの呼び出しのタグのトークン番号です。
type ToolTokens struct {
	CallStart      int // <tool_call>
	CallEnd        int // </tool_call>
	FunctionStart  int // <function=
	FunctionEnd    int // </function>
	ParameterStart int // <parameter=
	ParameterEnd   int // </parameter>
}

// ToolCalls は、functions（関数の定義。{"name": ..., "parameters": {...}}）のどれかの呼び出しを
// 1 件（multiple が true なら 1 件以上）だけ受け付ける文法を作ります。前後と呼び出しの間には改行を
// maxLeadingNewlines 個まで許します（テンプレートは呼び出しの間を "\n\n" にする）。
// 例: ToolCalls([{"name": "get_weather", "parameters": {...}}], tokens, true)
func ToolCalls(functions []pyjson.Value, tokens ToolTokens, multiple bool) (*Grammar, error) {
	if len(functions) == 0 {
		return nil, fmt.Errorf("grammar: no functions to call")
	}
	builder := &Builder{}
	newlines := leadingNewlines(builder)
	alternatives := make([][]Element, 0, len(functions))
	for _, function := range functions {
		call, err := functionCall(builder, function, tokens)
		if err != nil {
			return nil, err
		}
		alternatives = append(alternatives, One(Ref(call)))
	}
	call := builder.Rule("call", alternatives...)
	calls := call
	if multiple {
		calls = builder.NewRule("calls")
		builder.SetAlternatives(calls, One(Ref(call)), Seq(One(Ref(call)), One(Ref(newlines)), One(Ref(calls))))
	}
	root := builder.Rule("root", Seq(One(Ref(newlines)), One(Ref(calls)), One(Ref(newlines))))
	return builder.Build(root), nil
}

// functionCall は関数 1 個の呼び出しの規則を作ります。
func functionCall(builder *Builder, function pyjson.Value, tokens ToolTokens) (int, error) {
	name, ok := function.GetString("name")
	if !ok || name == "" {
		return 0, fmt.Errorf("grammar: function name must be a non-empty string")
	}
	parameters, _ := function.Get("parameters")
	compiler := newJSONCompiler(builder, parameters)
	var props []property
	if properties, ok := parameters.Get("properties"); ok && properties.Kind == pyjson.Object {
		required := make(map[string]bool)
		if list, ok := parameters.Get("required"); ok && list.Kind == pyjson.Array {
			for _, item := range list.Items {
				if item.Kind == pyjson.String {
					required[item.String] = true
				}
			}
		}
		for _, member := range properties.Members {
			value, err := parameterValue(compiler, member.Value)
			if err != nil {
				return 0, fmt.Errorf("function %q parameter %q: %w", name, member.Key, err)
			}
			parameter := builder.Rule("parameter-"+member.Key, Seq(
				One(Token(tokens.ParameterStart)), Literal(member.Key+">\n"),
				One(Ref(value)), Literal("\n"), One(Token(tokens.ParameterEnd)), Literal("\n")))
			props = append(props, property{key: member.Key, value: parameter, required: required[member.Key]})
		}
	}
	// 引数を後ろから順につなぎます。optional の引数は省けます。
	rest := builder.Rule("parameters-end", nil)
	for index := len(props) - 1; index >= 0; index-- {
		take := Seq(One(Ref(props[index].value)), One(Ref(rest)))
		if props[index].required {
			rest = builder.Rule("parameters", take)
		} else {
			rest = builder.Rule("parameters", take, One(Ref(rest)))
		}
	}
	return builder.Rule("function-"+name, Seq(
		One(Token(tokens.CallStart)), Literal("\n"),
		One(Token(tokens.FunctionStart)), Literal(name+">\n"),
		One(Ref(rest)),
		One(Token(tokens.FunctionEnd)), Literal("\n"), One(Token(tokens.CallEnd)))), nil
}

// parameterValue は引数の値の規則を作ります（この file の先頭の説明を参照）。
func parameterValue(compiler *jsonCompiler, schema pyjson.Value) (int, error) {
	builder := compiler.builder
	if value, ok := schema.Get("const"); ok {
		return builder.Rule("const", Literal(value.PythonStr())), nil
	}
	if values, ok := schema.Get("enum"); ok && values.Kind == pyjson.Array && len(values.Items) > 0 {
		alternatives := make([][]Element, 0, len(values.Items))
		for _, value := range values.Items {
			alternatives = append(alternatives, Literal(value.PythonStr()))
		}
		return builder.Rule("enum", alternatives...), nil
	}
	allowsText, err := allowsString(compiler.root, schema, 0)
	if err != nil {
		return 0, err
	}
	if allowsText {
		var anyByte ByteSet
		anyByte.AddRange(0, 0xff)
		return builder.Repeat("text", Bytes(anyByte), 0, -1), nil
	}
	return compiler.compile(schema)
}

// maxRefDepth は、allowsString が $ref をたどる深さの上限です（循環する参照で止まらなくなるのを防ぐ）。
const maxRefDepth = 32

// allowsString は、Schema が文字列を許すか（または型がわからないか）を返します。
// 例: {"type": "string"} → true、{"type": ["integer", "null"]} → false、{} → true
func allowsString(root, schema pyjson.Value, depth int) (bool, error) {
	if depth > maxRefDepth {
		return false, fmt.Errorf("grammar: $ref is nested too deeply")
	}
	if schema.Kind != pyjson.Object {
		return true, nil
	}
	if ref, ok := schema.GetString("$ref"); ok {
		target, err := resolveRef(root, ref)
		if err != nil {
			return false, err
		}
		return allowsString(root, target, depth+1)
	}
	for _, key := range []string{"anyOf", "oneOf"} {
		if alternatives, ok := schema.Get(key); ok && alternatives.Kind == pyjson.Array {
			for _, alternative := range alternatives.Items {
				allowed, err := allowsString(root, alternative, depth+1)
				if err != nil || allowed {
					return allowed, err
				}
			}
			return false, nil
		}
	}
	types, err := schemaTypeList(schema)
	if err != nil {
		return false, err
	}
	return len(types) == 0 || contains(types, "string"), nil
}
