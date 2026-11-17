package grammar

import (
	"testing"

	"github.com/cardy0828/pascallm/internal/pyjson"
)

// accepts は、文字列を最後まで照合して、文法の終わりに着けるかを返します。
func accepts(g *Grammar, text string) bool {
	state := g.Initial().AdvanceBytes(g, []byte(text))
	return state.Accepting()
}

// mustSchema は JSON の文字列から文法を作ります（失敗したらテストを止める）。
func mustSchema(t *testing.T, schema string) *Grammar {
	t.Helper()
	value, err := pyjson.Parse([]byte(schema))
	if err != nil {
		t.Fatal(err)
	}
	g, err := JSONSchema(value)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

// TestJSONSchemaObject は、プロパティの型・必須・順・追加のキーの扱いを確かめます。
func TestJSONSchemaObject(t *testing.T) {
	g := mustSchema(t, `{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"},
		"tags": {"type": "array", "items": {"type": "string"}, "maxItems": 2}, "kind": {"enum": ["a", "b", 3]}},
		"required": ["name", "age"]}`)
	cases := []struct {
		text string
		want bool
	}{
		{`{"name": "太郎", "age": 3}`, true},
		{"\n{\n  \"name\": \"a\\n\\u00e9\",\n  \"age\": -12,\n  \"tags\": [\"x\", \"y\"]\n}", true},
		{"{\n\t\"name\": \"a\",\n\t\"age\": 1\n}", true},
		{`{"name": "a", "age": 1}` + "\n", false}, // JSON の後の空白
		{`{"name":  "a", "age": 1}`, false},       // スペース 2 個
		{`{"name":"a","age":0,"kind":3}`, true},
		{`{"name":"a","age":0,"kind":"b"}`, true},
		{`{"name": "a"}`, false},                                  // age がない
		{`{"age": 3, "name": "a"}`, false},                        // 順が違う
		{`{"name": "a", "age": 1.5}`, false},                      // integer ではない
		{`{"name": "a", "age": 01}`, false},                       // 先頭の 0
		{`{"name": "a", "age": 1, "x": 1}`, false},                // 追加のキー
		{`{"name": "a", "age": 1, "tags": [1]}`, false},           // 要素の型
		{`{"name": "a", "age": 1, "tags": ["a","b","c"]}`, false}, // maxItems
		{`{"name": "a", "age": 1, "kind": "c"}`, false},           // enum にない
		{`{"name": "a", "age": 1,}`, false},                       // 最後の ,
	}
	for _, c := range cases {
		if got := accepts(g, c.text); got != c.want {
			t.Errorf("accepts(%q) = %v, want %v", c.text, got, c.want)
		}
	}
}

// TestJSONSchemaUnionRefAndNumbers は、anyOf・$ref の再帰・数の形を確かめます。
func TestJSONSchemaUnionRefAndNumbers(t *testing.T) {
	g := mustSchema(t, `{"$defs": {"node": {"type": "object", "properties": {"v": {"type": "number"},
		"next": {"anyOf": [{"$ref": "#/$defs/node"}, {"type": "null"}]}}, "required": ["v", "next"]}},
		"$ref": "#/$defs/node"}`)
	cases := []struct {
		text string
		want bool
	}{
		{`{"v": 1, "next": null}`, true},
		{`{"v": -1.25e+3, "next": {"v": 0.5, "next": {"v": 2E-2, "next": null}}}`, true},
		{`{"v": 1., "next": null}`, false},
		{`{"v": .5, "next": null}`, false},
		{`{"v": 1, "next": {"v": 2}}`, false},
	}
	for _, c := range cases {
		if got := accepts(g, c.text); got != c.want {
			t.Errorf("accepts(%q) = %v, want %v", c.text, got, c.want)
		}
	}
}

// TestJSONObjectAndLimits は、任意の object と、空白の上限を確かめます。
func TestJSONObjectAndLimits(t *testing.T) {
	g := JSONObject()
	for _, text := range []string{`{}`, `{"a": [1, {"b": null}], "c": "d"}`, "\n\n{ \"x\" : true }"} {
		if !accepts(g, text) {
			t.Errorf("JSONObject rejected %q", text)
		}
	}
	for _, text := range []string{`[]`, `"a"`, `{"a" 1}`, `{"a": 1}}`} {
		if accepts(g, text) {
			t.Errorf("JSONObject accepted %q", text)
		}
	}
	if accepts(g, "{\n"+spaces(maxIndentSpaces+1)+"}") {
		t.Error("indentation beyond the limit was accepted")
	}
	if !accepts(g, "{\n"+spaces(maxIndentSpaces)+"}") {
		t.Error("indentation at the limit was rejected")
	}
	if accepts(g, "{\n  \t}") {
		t.Error("indentation mixing spaces and tabs was accepted")
	}
}

// spaces は n 個の空白を返します。
func spaces(n int) string {
	buffer := make([]byte, n)
	for index := range buffer {
		buffer[index] = ' '
	}
	return string(buffer)
}

// テスト用のツールのタグのトークン番号です（実際の番号とは関係ない）。
var testTokens = ToolTokens{CallStart: 1, CallEnd: 2, FunctionStart: 3, FunctionEnd: 4, ParameterStart: 5, ParameterEnd: 6}

// step はトークン（番号が正）かバイト列（文字列）を照合する手順 1 個です。
type step struct {
	token int
	text  string
}

// run は手順を順に照合し、最後に終われる状態かを返します。途中で合わなくなれば false です。
func run(g *Grammar, steps []step) bool {
	state := g.Initial()
	for _, s := range steps {
		if s.token > 0 {
			state = state.AdvanceToken(g, s.token)
		} else {
			state = state.AdvanceBytes(g, []byte(s.text))
		}
		if state.Dead() {
			return false
		}
	}
	return state.Accepting()
}

// callSteps は、関数の呼び出し 1 件の手順を作ります。arguments は [名前, 値, 名前, 値, ...] です。
func callSteps(name string, arguments ...string) []step {
	steps := []step{{token: testTokens.CallStart}, {text: "\n"}, {token: testTokens.FunctionStart}, {text: name + ">\n"}}
	for index := 0; index+1 < len(arguments); index += 2 {
		steps = append(steps, step{token: testTokens.ParameterStart}, step{text: arguments[index] + ">\n" + arguments[index+1] + "\n"},
			step{token: testTokens.ParameterEnd}, step{text: "\n"})
	}
	return append(steps, step{token: testTokens.FunctionEnd}, step{text: "\n"}, step{token: testTokens.CallEnd})
}

// TestToolCalls は、ツールの呼び出しの文法で、名前・必須の引数・値の型・件数を確かめます。
func TestToolCalls(t *testing.T) {
	functions, err := pyjson.Parse([]byte(`[{"name": "get_weather", "parameters": {"type": "object", "properties": {
		"city": {"type": "string"}, "days": {"type": "integer"}, "unit": {"enum": ["c", "f"]}}, "required": ["city"]}},
		{"name": "now"}]`))
	if err != nil {
		t.Fatal(err)
	}
	single, err := ToolCalls(functions.Items, testTokens, false)
	if err != nil {
		t.Fatal(err)
	}
	multiple, err := ToolCalls(functions.Items, testTokens, true)
	if err != nil {
		t.Fatal(err)
	}
	two := append(callSteps("now"), append([]step{{text: "\n\n"}}, callSteps("get_weather", "city", "東京")...)...)
	cases := []struct {
		name    string
		grammar *Grammar
		steps   []step
		want    bool
	}{
		{"city only", single, callSteps("get_weather", "city", "東京\n(港区)"), true},
		{"all arguments", single, callSteps("get_weather", "city", "a", "days", "3", "unit", "f"), true},
		{"no arguments", single, callSteps("now"), true},
		{"missing required", single, callSteps("get_weather", "days", "3"), false},
		{"wrong type", single, callSteps("get_weather", "city", "a", "days", "three"), false},
		{"enum value", single, callSteps("get_weather", "city", "a", "unit", "k"), false},
		{"unknown function", single, callSteps("delete_all"), false},
		{"two calls in single", single, two, false},
		{"two calls in multiple", multiple, two, true},
	}
	for _, c := range cases {
		if got := run(c.grammar, c.steps); got != c.want {
			t.Errorf("%s: accepted = %v, want %v", c.name, got, c.want)
		}
	}
}

// TestEither は、2 つの文法のどちらにも合う文字列を受け付けることを確かめます。
func TestEither(t *testing.T) {
	g := Either(mustSchema(t, `{"type": "integer"}`), mustSchema(t, `{"type": "boolean"}`))
	for _, text := range []string{"12", "true"} {
		if !accepts(g, text) {
			t.Errorf("Either rejected %q", text)
		}
	}
	if accepts(g, `"a"`) {
		t.Error(`Either accepted "a"`)
	}
}

// fakeVocabulary はテスト用の語彙です。番号 0〜9 は 1 バイトの数字、10 は終わり、11 は追加トークンです。
type fakeVocabulary struct{}

func (fakeVocabulary) Size() int { return 12 }
func (fakeVocabulary) TokenBytes(id int) []byte {
	switch {
	case id < 10:
		return []byte{byte('0' + id)}
	case id == 10:
		return []byte("<end>")
	default:
		return []byte("<think>")
	}
}
func (fakeVocabulary) IsAdded(id int) bool { return id >= 10 }

// TestConstraint は、終わりのトークンと追加トークンの扱い、思考の間は縛らないことを確かめます。
func TestConstraint(t *testing.T) {
	g := mustSchema(t, `{"type": "integer"}`)
	const thinkEnd, end, think = 11, 10, 11
	c := NewConstraint(g, fakeVocabulary{}, false, thinkEnd, []int{end})
	if c.Allows(end) {
		t.Error("end token was allowed before any digit")
	}
	if c.Allows(think) {
		t.Error("an added token was allowed by the grammar")
	}
	if !c.Allows(5) {
		t.Fatal("digit 5 was rejected")
	}
	if err := c.Accept(5); err != nil {
		t.Fatal(err)
	}
	if !c.Allows(end) {
		t.Error("end token was rejected after a complete integer")
	}
	reasoning := NewConstraint(g, fakeVocabulary{}, true, thinkEnd, []int{end})
	if reasoning.Allows(end) {
		t.Error("end token was allowed while reasoning")
	}
	if err := reasoning.Accept(thinkEnd); err != nil {
		t.Fatal(err)
	}
	if reasoning.Allows(end) || !reasoning.Allows(0) {
		t.Error("constraint did not start after </think>")
	}
}
