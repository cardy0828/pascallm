package pyjson

import (
	"math"
	"testing"
)

// TestDumpsMatchesPython は、Python の json.dumps(ensure_ascii=False) と同じ文字列になることを確かめます。
// 期待値は Python 3.12 で json.dumps(json.loads(入力), ensure_ascii=False) を実行した結果です。
func TestDumpsMatchesPython(t *testing.T) {
	cases := []struct{ input, want string }{
		{`{"b": 1, "a": [true, false, null]}`, `{"b": 1, "a": [true, false, null]}`},
		{`{"x":1.50,"y":1e2,"z":-0,"w":-0.0}`, `{"x": 1.5, "y": 100.0, "z": 0, "w": -0.0}`},
		{`[0.0001, 0.00001, 1e15, 1e16, 1.5e20, 123456789012345678901234]`, `[0.0001, 1e-05, 1000000000000000.0, 1e+16, 1.5e+20, 123456789012345678901234]`},
		{`[1e400, -1e400, 2.5e-7]`, `[Infinity, -Infinity, 2.5e-07]`},
		{`"日本é <tag> & \"q\" \\ / \n\t\u0001\u001f\u007f"`, "\"日本é <tag> & \\\"q\\\" \\\\ / \\n\\t\\u0001\\u001f\u007f\""},
		{`{"a": 1, "a": 2, "b": 3}`, `{"a": 2, "b": 3}`},
		{`{}`, `{}`},
		{`[]`, `[]`},
	}
	for _, c := range cases {
		value, err := Parse([]byte(c.input))
		if err != nil {
			t.Fatalf("Parse(%s): %v", c.input, err)
		}
		if got := value.Dumps(); got != c.want {
			t.Errorf("Dumps(%s) = %s, want %s", c.input, got, c.want)
		}
	}
}

// TestPythonFloat は、Python の float の repr と同じ表記になることを確かめます。
func TestPythonFloat(t *testing.T) {
	cases := []struct {
		value float64
		want  string
	}{
		{1, "1.0"}, {0.1, "0.1"}, {-2.5, "-2.5"}, {1e16, "1e+16"}, {1e-5, "1e-05"}, {123.456, "123.456"},
		{math.Copysign(0, -1), "-0.0"}, {0.30000000000000004, "0.30000000000000004"}, {5e-324, "5e-324"},
	}
	for _, c := range cases {
		if got := PythonFloat(c.value); got != c.want {
			t.Errorf("PythonFloat(%v) = %s, want %s", c.value, got, c.want)
		}
	}
}

// TestParseRejectsTrailingData は、値の後に余分なものがあれば断ることを確かめます。
func TestParseRejectsTrailingData(t *testing.T) {
	if _, err := Parse([]byte(`{"a": 1} {"b": 2}`)); err == nil {
		t.Fatal("trailing data was accepted")
	}
}

// TestTruthy は、Jinja（Python）の真偽の判定と同じになることを確かめます。
func TestTruthy(t *testing.T) {
	cases := []struct {
		input string
		want  bool
	}{
		{`null`, false}, {`false`, false}, {`0`, false}, {`0.0`, false}, {`""`, false}, {`[]`, false}, {`{}`, false},
		{`true`, true}, {`1`, true}, {`"a"`, true}, {`[0]`, true}, {`{"a": null}`, true},
	}
	for _, c := range cases {
		value, err := Parse([]byte(c.input))
		if err != nil {
			t.Fatal(err)
		}
		if got := value.Truthy(); got != c.want {
			t.Errorf("Truthy(%s) = %v, want %v", c.input, got, c.want)
		}
	}
}
