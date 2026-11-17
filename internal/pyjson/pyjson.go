// Package pyjson は、キーの順を保って JSON を読み、Python の json.dumps(ensure_ascii=False) と同じ形で
// 書き出します（M6）。
//
// チャットテンプレート（chat_template.jinja）は、ツールの定義や引数を Jinja の tojson で書きます。
// transformers の tojson は json.dumps(x, ensure_ascii=False) で、次の形になります。
//   - 区切りは ", " と ": "（例: {"a": 1, "b": [1, 2]}）
//   - キーは要求の順のまま（並べ替えない）
//   - 日本語などは \u にせず、そのまま書く。制御文字・"・\ だけを \ で書く
//   - 小数は Python の repr の形（例: 1.0、1e-05、1e+16）
//
// Go の encoding/json はキーの順を保たず、< > & を < などにするため、この形は作れません。
package pyjson

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
)

// Kind は JSON の値の種類です。
type Kind int

// 値の種類です。
const (
	Null Kind = iota
	Bool
	Number
	String
	Array
	Object
)

// Value は JSON の値 1 個です。Kind に応じて、使う項目が決まります。
type Value struct {
	Kind    Kind
	Bool    bool     // Kind が Bool のときの値
	Number  string   // Kind が Number のときの、元の JSON の表記。例: "1.50"、"-3"、"1e2"
	String  string   // Kind が String のときの値
	Items   []Value  // Kind が Array のときの要素
	Members []Member // Kind が Object のときのキーと値（元の順）
}

// Member はオブジェクトのキーと値 1 組です。
type Member struct {
	Key   string
	Value Value
}

// Parse は JSON を読みます。キーの順と数の表記を保ちます。同じキーが 2 回あれば、Python の dict と同じく
// 最初の位置に後の値を入れます。例: Parse([]byte(`{"b": 1, "a": [true, null]}`))
func Parse(data []byte) (Value, error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	value, err := parseValue(decoder)
	if err != nil {
		return Value{}, err
	}
	if _, err := decoder.Token(); err != io.EOF {
		return Value{}, errors.New("pyjson: unexpected data after the JSON value")
	}
	return value, nil
}

// parseValue は decoder から値 1 個を読みます。
func parseValue(decoder *json.Decoder) (Value, error) {
	token, err := decoder.Token()
	if err != nil {
		return Value{}, err
	}
	switch typed := token.(type) {
	case nil:
		return Value{Kind: Null}, nil
	case bool:
		return Value{Kind: Bool, Bool: typed}, nil
	case json.Number:
		return Value{Kind: Number, Number: typed.String()}, nil
	case string:
		return Value{Kind: String, String: typed}, nil
	case json.Delim:
		switch typed {
		case '[':
			return parseArray(decoder)
		case '{':
			return parseObject(decoder)
		}
	}
	return Value{}, fmt.Errorf("pyjson: unexpected token %v", token)
}

// parseArray は "[" の後から "]" までを読みます。
func parseArray(decoder *json.Decoder) (Value, error) {
	array := Value{Kind: Array, Items: []Value{}}
	for decoder.More() {
		item, err := parseValue(decoder)
		if err != nil {
			return Value{}, err
		}
		array.Items = append(array.Items, item)
	}
	if _, err := decoder.Token(); err != nil {
		return Value{}, err
	}
	return array, nil
}

// parseObject は "{" の後から "}" までを読みます。
func parseObject(decoder *json.Decoder) (Value, error) {
	object := Value{Kind: Object, Members: []Member{}}
	for decoder.More() {
		token, err := decoder.Token()
		if err != nil {
			return Value{}, err
		}
		key, ok := token.(string)
		if !ok {
			return Value{}, errors.New("pyjson: object key must be a string")
		}
		value, err := parseValue(decoder)
		if err != nil {
			return Value{}, err
		}
		object.Set(key, value)
	}
	if _, err := decoder.Token(); err != nil {
		return Value{}, err
	}
	return object, nil
}

// NewString は文字列の値を作ります。例: NewString("Paris")
func NewString(text string) Value { return Value{Kind: String, String: text} }

// NewObject は空のオブジェクトを作ります。
func NewObject() Value { return Value{Kind: Object, Members: []Member{}} }

// Set は、オブジェクトのキーに値を入れます。キーがすでにあれば、その位置のまま値を置き換えます（Python の dict と同じ）。
// 例: v.Set("city", NewString("Paris"))
func (v *Value) Set(key string, value Value) {
	for index := range v.Members {
		if v.Members[index].Key == key {
			v.Members[index].Value = value
			return
		}
	}
	v.Members = append(v.Members, Member{Key: key, Value: value})
}

// Get は、オブジェクトのキーの値を返します。オブジェクトでないか、キーがなければ ok = false です。
// 例: Get("function") → {...}, true
func (v Value) Get(key string) (Value, bool) {
	if v.Kind != Object {
		return Value{}, false
	}
	for _, member := range v.Members {
		if member.Key == key {
			return member.Value, true
		}
	}
	return Value{}, false
}

// GetString は、キーの値が文字列ならその値を返します。例: {"name": "f"} の GetString("name") → "f", true
func (v Value) GetString(key string) (string, bool) {
	value, ok := v.Get(key)
	if !ok || value.Kind != String {
		return "", false
	}
	return value.String, true
}

// Truthy は、Python（Jinja）で真とみなされるかを返します。
// null・false・0・空文字列・空配列・空オブジェクトは偽です。例: Value{Kind: String, String: ""}.Truthy() → false
func (v Value) Truthy() bool {
	switch v.Kind {
	case Bool:
		return v.Bool
	case Number:
		number, err := strconv.ParseFloat(v.Number, 64)
		return err != nil || number != 0
	case String:
		return v.String != ""
	case Array:
		return len(v.Items) > 0
	case Object:
		return len(v.Members) > 0
	default:
		return false
	}
}

// IsInteger は、数の表記が整数（小数点も指数もない）かを返します。Python の json.loads で int になるものです。
// 例: "3" → true、"3.0" → false、"1e2" → false
func (v Value) IsInteger() bool {
	return v.Kind == Number && !strings.ContainsAny(v.Number, ".eE")
}

// Dumps は、Python の json.dumps(v, ensure_ascii=False) と同じ文字列を返します。
// 例: {"a": [1, 2.50], "b": "日本"} → `{"a": [1, 2.5], "b": "日本"}`
func (v Value) Dumps() string {
	var builder strings.Builder
	v.writeDumps(&builder)
	return builder.String()
}

// writeDumps は Dumps の中身を builder に書きます。
func (v Value) writeDumps(builder *strings.Builder) {
	switch v.Kind {
	case Null:
		builder.WriteString("null")
	case Bool:
		builder.WriteString(boolText(v.Bool))
	case Number:
		builder.WriteString(v.PythonNumber())
	case String:
		writeQuoted(builder, v.String)
	case Array:
		builder.WriteByte('[')
		for index, item := range v.Items {
			if index > 0 {
				builder.WriteString(", ")
			}
			item.writeDumps(builder)
		}
		builder.WriteByte(']')
	case Object:
		builder.WriteByte('{')
		for index, member := range v.Members {
			if index > 0 {
				builder.WriteString(", ")
			}
			writeQuoted(builder, member.Key)
			builder.WriteString(": ")
			member.Value.writeDumps(builder)
		}
		builder.WriteByte('}')
	}
}

// boolText は真偽値を JSON の表記にします。例: true → "true"
func boolText(value bool) string {
	if value {
		return "true"
	}
	return "false"
}

// writeQuoted は、文字列を json.dumps(ensure_ascii=False) の形で引用符付きで書きます。
// " と \ と制御文字（U+0000〜U+001F）だけを \ で書き、ほかはそのまま書きます。
// 例: "a\"b\n日本" → `"a\"b\n日本"`
func writeQuoted(builder *strings.Builder, text string) {
	builder.WriteByte('"')
	for _, r := range text {
		switch r {
		case '"':
			builder.WriteString(`\"`)
		case '\\':
			builder.WriteString(`\\`)
		case '\n':
			builder.WriteString(`\n`)
		case '\r':
			builder.WriteString(`\r`)
		case '\t':
			builder.WriteString(`\t`)
		case '\b':
			builder.WriteString(`\b`)
		case '\f':
			builder.WriteString(`\f`)
		default:
			if isControl(r) {
				fmt.Fprintf(builder, `\u%04x`, r)
			} else {
				builder.WriteRune(r)
			}
		}
	}
	builder.WriteByte('"')
}

// isControl は、JSON の文字列で \ を使って書く必要がある制御文字（U+0000〜U+001F）かを返します。
// 例: isControl('\x01') → true、isControl(' ') → false
func isControl(r rune) bool {
	const lastControl = 0x1f
	return r >= 0 && r <= lastControl
}

// Python の float の repr が指数の形になる範囲です。10 進の指数（1.23e+05 の 5）がこの範囲の外なら指数の形です。
const (
	minFixedExponent = -4
	maxFixedExponent = 15
)

// PythonNumber は、数を Python の json.loads で読んで str()（json.dumps も同じ）にした表記を返します。
//   - 整数の表記なら int として書く。例: "-0" → "0"、"007" は JSON にない
//   - それ以外は float の repr。例: "1.50" → "1.5"、"1e2" → "100.0"、"0.00001" → "1e-05"、"1e400" → "Infinity"
func (v Value) PythonNumber() string {
	if v.IsInteger() {
		return pythonInteger(v.Number)
	}
	number, err := strconv.ParseFloat(v.Number, 64)
	if err != nil && !math.IsInf(number, 0) {
		return v.Number
	}
	return PythonFloat(number)
}

// pythonInteger は整数の表記を Python の int の str() にします。JSON の整数は先頭に 0 を付けないため、
// 変わるのは "-0" だけです。例: "-0" → "0"、"-12" → "-12"
func pythonInteger(text string) string {
	if strings.TrimLeft(strings.TrimPrefix(text, "-"), "0") == "" {
		return "0"
	}
	return text
}

// PythonFloat は、float64 を Python の repr と同じ表記にします（最短で元に戻せる桁数）。
// 例: 1 → "1.0"、0.1 → "0.1"、1e16 → "1e+16"、1e-05 → "1e-05"、-0 → "-0.0"、+Inf → "Infinity"
func PythonFloat(number float64) string {
	switch {
	case math.IsInf(number, 1):
		return "Infinity"
	case math.IsInf(number, -1):
		return "-Infinity"
	case math.IsNaN(number):
		return "NaN"
	}
	// 'e' の最短表記は "-d.ddde±XX" の形です。
	formatted := strconv.FormatFloat(number, 'e', -1, 64)
	sign := ""
	if strings.HasPrefix(formatted, "-") {
		sign, formatted = "-", formatted[1:]
	}
	mantissa, exponentText, _ := strings.Cut(formatted, "e")
	digits := strings.Replace(mantissa, ".", "", 1)
	exponent, _ := strconv.Atoi(exponentText)
	if exponent < minFixedExponent || exponent > maxFixedExponent {
		return sign + scientific(digits, exponent)
	}
	return sign + fixed(digits, exponent)
}

// scientific は、有効数字 digits と 10 進の指数を、Python の指数の表記にします（指数は 2 桁以上、符号付き）。
// 例: ("15", 20) → "1.5e+20"、("1", -5) → "1e-05"
func scientific(digits string, exponent int) string {
	mantissa := digits[:1]
	if len(digits) > 1 {
		mantissa += "." + digits[1:]
	}
	sign := "+"
	if exponent < 0 {
		sign, exponent = "-", -exponent
	}
	return fmt.Sprintf("%se%s%02d", mantissa, sign, exponent)
}

// fixed は、有効数字 digits と 10 進の指数を、小数点の表記にします（整数なら ".0" を付ける）。
// 例: ("15", 0) → "1.5"、("1", 2) → "100.0"、("25", -3) → "0.0025"
func fixed(digits string, exponent int) string {
	point := exponent + 1 // 小数点の前の桁数
	switch {
	case point <= 0:
		return "0." + strings.Repeat("0", -point) + digits
	case point >= len(digits):
		return digits + strings.Repeat("0", point-len(digits)) + ".0"
	default:
		return digits[:point] + "." + digits[point:]
	}
}

// PythonStr は、テンプレートの render_value と同じく、値を文字列にします。
// 文字列はそのまま、真偽値は true/false、null は null、オブジェクト・配列は Dumps、数は Python の str() です。
// 例: NewString("Paris") → "Paris"、{"a": 1} → `{"a": 1}`
func (v Value) PythonStr() string {
	if v.Kind == String {
		return v.String
	}
	return v.Dumps()
}
