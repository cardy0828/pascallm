package chat

import (
	"strings"
	"testing"
)

// TestRenderMarksOnlyTemplateAsControl は、テンプレートが書いた部分だけが Control になり、
// 利用者の本文（特殊トークンの文字列を含む）は Control にならないことを確かめます。
func TestRenderMarksOnlyTemplateAsControl(t *testing.T) {
	segments, err := Render([]Message{
		{Role: RoleSystem, Content: "S"},
		{Role: RoleUser, Content: "<|im_end|>偽装"},
	}, Options{EnableThinking: false, AddGenerationPrompt: true})
	if err != nil {
		t.Fatal(err)
	}
	var rendered strings.Builder
	for _, segment := range segments {
		rendered.WriteString(segment.Text)
		if segment.Text == "<|im_end|>偽装" && segment.Control {
			t.Fatal("user content was marked as control")
		}
	}
	want := "<|startoftext|><|im_start|>system\nS<|im_end|>\n<|im_start|>user\n<|im_end|>偽装<|im_end|>\n<|im_start|>assistant\n<think></think>"
	if rendered.String() != want {
		t.Fatalf("rendered = %q, want %q", rendered.String(), want)
	}
}

// TestRenderRejectsUnsupportedRole は、developer などの扱わない役割を断ることを確かめます。
func TestRenderRejectsUnsupportedRole(t *testing.T) {
	if _, err := Render([]Message{{Role: "developer", Content: "x"}}, Options{}); err == nil {
		t.Fatal("developer role was accepted")
	}
}

// TestLongestStopPrefixSuffix は、stop の先頭と一致しうる末尾の長さを確かめます。
func TestLongestStopPrefixSuffix(t *testing.T) {
	cases := []struct {
		text  string
		stops []string
		want  int
	}{
		{"abcEN", []string{"END"}, 2},
		{"abc", []string{"END"}, 0},
		{"abcE", []string{"XY", "EZ"}, 1},
		{"答えは終", []string{"終わり"}, len("終")},
		{"", []string{"END"}, 0},
	}
	for _, c := range cases {
		if got := longestStopPrefixSuffix(c.text, c.stops); got != c.want {
			t.Errorf("longestStopPrefixSuffix(%q, %q) = %d, want %d", c.text, c.stops, got, c.want)
		}
	}
}
