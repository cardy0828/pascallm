package chat

import (
	"testing"

	"github.com/cardy0828/pascallm/internal/pyjson"
)

// testToolSet は、型の変換を確かめるためのツールの定義です。
func testToolSet(t *testing.T) *ToolSet {
	t.Helper()
	tools, err := pyjson.Parse([]byte(`[{"type": "function", "function": {"name": "f", "parameters": {"type": "object",
		"properties": {"s": {"type": "string"}, "i": {"type": "integer"}, "n": {"type": ["number", "null"]},
		"b": {"type": "boolean"}, "o": {"type": "object"}, "a": {"type": "array"}, "e": {"enum": [1, 2]},
		"u": {"anyOf": [{"type": "integer"}, {"type": "string"}]}}}}}]`))
	if err != nil {
		t.Fatal(err)
	}
	return NewToolSet(tools.Items)
}

// TestParseToolCallConvertsTypes は、引数の値が Schema の型に合わせて変換されることを確かめます。
func TestParseToolCallConvertsTypes(t *testing.T) {
	text := "\n<function=f>\n" +
		"<parameter=s>\n123\n</parameter>\n" +
		"<parameter=i>\n42\n</parameter>\n" +
		"<parameter=n>\nnull\n</parameter>\n" +
		"<parameter=b>\ntrue\n</parameter>\n" +
		"<parameter=o>\n{\"k\": [1, 2.50]}\n</parameter>\n" +
		"<parameter=a>\nnot json\n</parameter>\n" +
		"<parameter=e>\n2\n</parameter>\n" +
		"<parameter=u>\n7\n</parameter>\n" +
		"<parameter=extra>\n\n2 行\n目\n\n</parameter>\n" +
		"</function>\n"
	call, ok := ParseToolCall(text, testToolSet(t))
	if !ok {
		t.Fatal("ParseToolCall failed")
	}
	want := `{"s": "123", "i": 42, "n": null, "b": true, "o": {"k": [1, 2.5]}, "a": "not json", "e": 2, "u": 7, "extra": "\n2 行\n目\n"}`
	if call.Name != "f" || call.Arguments.Dumps() != want {
		t.Fatalf("call = %s %s, want f %s", call.Name, call.Arguments.Dumps(), want)
	}
}

// TestParseToolCallRejectsBrokenText は、形の崩れた呼び出しを断ることを確かめます。
func TestParseToolCallRejectsBrokenText(t *testing.T) {
	for _, text := range []string{
		"",
		"<function=>\n</function>",
		"\n<function=f>\n<parameter=x>\n1\n", // </parameter> がない
		"\n<function=f>\n</function>\n余分な文字",            // </function> の後
		"\n<function=f\n<parameter=x>\n1\n</parameter>", // 名前の後の > がない
	} {
		if call, ok := ParseToolCall(text, nil); ok {
			t.Errorf("ParseToolCall(%q) = %+v, want failure", text, call)
		}
	}
	if call, ok := ParseToolCall("\n<function=g>\n<parameter=x>\n1\n</parameter>\n", nil); !ok || call.Arguments.Dumps() != `{"x": "1"}` {
		t.Errorf("call without </function> = %+v, %v", call, ok)
	}
}

// TestRenderToolCallsRoundTrip は、読み取った呼び出しをテンプレートで書くと、元の文字列に戻ることを確かめます。
func TestRenderToolCallsRoundTrip(t *testing.T) {
	text := "\n<function=f>\n<parameter=s>\nabc\n</parameter>\n<parameter=o>\n{\"k\": [1, 2.5]}\n</parameter>\n</function>\n"
	call, ok := ParseToolCall(text, testToolSet(t))
	if !ok {
		t.Fatal("ParseToolCall failed")
	}
	w := &writer{}
	renderToolCalls(w, []ToolCall{{Name: call.Name, Arguments: call.Arguments}})
	var rendered string
	for _, segment := range w.segments {
		rendered += segment.Text
	}
	if want := ToolCallStart + text + ToolCallEnd; rendered != want {
		t.Fatalf("rendered = %q, want %q", rendered, want)
	}
}
