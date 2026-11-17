package chat

import (
	"math"
	"strconv"
	"strings"

	"github.com/cardy0828/pascallm/internal/pyjson"
)

// この file は、出力のツールの呼び出し（XML の形）を読み、引数の値を JSON Schema の型に合わせる処理です。
//
// 呼び出しの形（chat_template.jinja の render_tool_calls と同じ）:
//
//	<tool_call>
//	<function=get_weather>
//	<parameter=city>
//	Paris
//	</parameter>
//	</function>
//	</tool_call>
//
// 値はすべて文字列で出るため、定義の型が string 以外を含むときだけ JSON として読みます。

// ToolSet は、要求の tools を関数の名前で引けるようにしたものです。
type ToolSet struct {
	functions map[string]pyjson.Value // 名前 → 関数の定義（{"name": ..., "parameters": {...}}）
	order     []string                // 定義の順の名前
}

// NewToolSet は tools から ToolSet を作ります。遅延読み込み（defer_loading）の関数も含めます。
// 名前が文字列でない定義は飛ばします。例: NewToolSet([{"type": "function", "function": {"name": "f"}}])
func NewToolSet(tools []pyjson.Value) *ToolSet {
	set := &ToolSet{functions: make(map[string]pyjson.Value)}
	for _, tool := range tools {
		function := functionOf(tool)
		name, ok := function.GetString("name")
		if !ok {
			continue
		}
		if _, seen := set.functions[name]; !seen {
			set.order = append(set.order, name)
		}
		set.functions[name] = function
	}
	return set
}

// Names は関数の名前を定義の順で返します。例: ["get_weather", "search"]
func (s *ToolSet) Names() []string { return s.order }

// Function は名前の関数の定義を返します。例: Function("get_weather") → {...}, true
func (s *ToolSet) Function(name string) (pyjson.Value, bool) {
	function, ok := s.functions[name]
	return function, ok
}

// ParameterSchema は、関数の引数の JSON Schema（parameters.properties の値）を返します。
// 例: ParameterSchema("get_weather", "city") → {"type": "string"}, true
func (s *ToolSet) ParameterSchema(function, parameter string) (pyjson.Value, bool) {
	definition, ok := s.functions[function]
	if !ok {
		return pyjson.Value{}, false
	}
	parameters, ok := definition.Get("parameters")
	if !ok {
		return pyjson.Value{}, false
	}
	properties, ok := parameters.Get("properties")
	if !ok {
		return pyjson.Value{}, false
	}
	return properties.Get(parameter)
}

// ParseToolCall は、"<tool_call>" と "</tool_call>" の間の文字列を呼び出しにします。形が違えば ok = false です。
// 空白の入り方には寛容で、"</function>" がなくても引数を読み終えていれば受け付けます。
// 例: "\n<function=f>\n<parameter=x>\n1\n</parameter>\n</function>\n" → {f {"x": 1}}（x が integer のとき）
func ParseToolCall(text string, tools *ToolSet) (ParsedToolCall, bool) {
	rest := strings.TrimSpace(text)
	rest, ok := strings.CutPrefix(rest, FunctionStart)
	if !ok {
		return ParsedToolCall{}, false
	}
	name, rest, ok := strings.Cut(rest, ">")
	name = strings.TrimSpace(name)
	if !ok || name == "" || strings.ContainsAny(name, "\n<") {
		return ParsedToolCall{}, false
	}
	call := ParsedToolCall{Name: name, Arguments: pyjson.NewObject()}
	for {
		rest = strings.TrimLeft(rest, " \t\r\n")
		switch {
		case rest == "":
			return call, true
		case strings.HasPrefix(rest, FunctionEnd):
			return call, strings.TrimSpace(rest[len(FunctionEnd):]) == ""
		case strings.HasPrefix(rest, ParameterStart):
			var key, value string
			key, rest, ok = strings.Cut(rest[len(ParameterStart):], ">")
			key = strings.TrimSpace(key)
			if !ok || key == "" || strings.ContainsAny(key, "\n<") {
				return ParsedToolCall{}, false
			}
			value, rest, ok = strings.Cut(rest, ParameterEnd)
			if !ok {
				return ParsedToolCall{}, false
			}
			call.Arguments.Set(key, convertArgument(trimOneNewline(value), schemaFor(tools, name, key)))
		default:
			return ParsedToolCall{}, false
		}
	}
}

// schemaFor は、引数の JSON Schema を返します（tools がないか、定義にない引数なら null の値）。
func schemaFor(tools *ToolSet, function, parameter string) pyjson.Value {
	if tools == nil {
		return pyjson.Value{}
	}
	schema, _ := tools.ParameterSchema(function, parameter)
	return schema
}

// trimOneNewline は、値の前後に 1 個ずつ付く改行を外します（テンプレートが "\n" + 値 + "\n" と書くため）。
// 例: "\nParis\n" → "Paris"、"\n\nA\n\n" → "\nA\n"
func trimOneNewline(value string) string {
	value = strings.TrimPrefix(value, "\n")
	return strings.TrimSuffix(value, "\n")
}

// convertArgument は、文字列の値を Schema の型に合わせます。
//   - 型がわからない、または string だけなら文字列のまま
//   - string 以外の型を含むなら JSON として読み、読めた値の種類が型のどれかに合えばその値、合わなければ文字列
//
// 例: ("3", {"type": "integer"}) → 3、("abc", {"type": "integer"}) → "abc"、("3", {"type": "string"}) → "3"
func convertArgument(value string, schema pyjson.Value) pyjson.Value {
	types := SchemaTypes(schema)
	if len(types) == 0 || (len(types) == 1 && types["string"]) {
		return pyjson.NewString(value)
	}
	parsed, err := pyjson.Parse([]byte(value))
	if err != nil || !MatchesType(parsed, types) {
		return pyjson.NewString(value)
	}
	return parsed
}

// SchemaTypes は、JSON Schema が許す型の名前の集合を返します。
// type（文字列または配列）、anyOf・oneOf の和、enum・const の値の種類から求めます。わからなければ空です。
// 例: {"type": ["integer", "null"]} → {integer, null}、{"enum": ["a", 1]} → {string, integer}
func SchemaTypes(schema pyjson.Value) map[string]bool {
	types := make(map[string]bool)
	if typeValue, ok := schema.Get("type"); ok {
		switch typeValue.Kind {
		case pyjson.String:
			types[typeValue.String] = true
		case pyjson.Array:
			for _, item := range typeValue.Items {
				if item.Kind == pyjson.String {
					types[item.String] = true
				}
			}
		}
		if nullable, ok := schema.Get("nullable"); ok && nullable.Truthy() {
			types["null"] = true
		}
		return types
	}
	for _, key := range []string{"anyOf", "oneOf"} {
		if alternatives, ok := schema.Get(key); ok && alternatives.Kind == pyjson.Array {
			for _, alternative := range alternatives.Items {
				for name := range SchemaTypes(alternative) {
					types[name] = true
				}
			}
		}
	}
	if values, ok := schema.Get("enum"); ok && values.Kind == pyjson.Array {
		for _, value := range values.Items {
			types[valueTypeName(value)] = true
		}
	}
	if value, ok := schema.Get("const"); ok {
		types[valueTypeName(value)] = true
	}
	return types
}

// valueTypeName は、値の JSON Schema の型の名前を返します。例: 3 → "integer"、3.5 → "number"、"a" → "string"
func valueTypeName(value pyjson.Value) string {
	switch value.Kind {
	case pyjson.Null:
		return "null"
	case pyjson.Bool:
		return "boolean"
	case pyjson.Number:
		if value.IsInteger() {
			return "integer"
		}
		return "number"
	case pyjson.String:
		return "string"
	case pyjson.Array:
		return "array"
	default:
		return "object"
	}
}

// MatchesType は、値の種類が型の集合のどれかに合うかを返します。
// integer は小数部が 0 の数（例: 2.0）も含み、number は整数も含みます（JSON Schema と同じ）。
// 例: (3, {number}) → true、(2.5, {integer}) → false
func MatchesType(value pyjson.Value, types map[string]bool) bool {
	name := valueTypeName(value)
	switch {
	case types[name]:
		return true
	case value.Kind == pyjson.Number && types["number"]:
		return true
	case value.Kind == pyjson.Number && types["integer"]:
		return isIntegralNumber(value.Number)
	default:
		return false
	}
}

// isIntegralNumber は、数の表記の値が整数か（小数部が 0 か）を返します。例: "2.0" → true、"2.5" → false
func isIntegralNumber(text string) bool {
	number, err := strconv.ParseFloat(text, 64)
	return err == nil && number == math.Trunc(number)
}
