package grammar

import (
	"github.com/cardy0828/pascallm/internal/pyjson"
	"strings"
	"testing"
)

// TestGemmaToolGrammar は特殊引用符と通常バイトを分けて照合し、必須項目・型・enum・余分な呼び出しを検証します。
func TestGemmaToolGrammar(t *testing.T) {
	f, err := pyjson.Parse([]byte(`{"name":"f","parameters":{"type":"object","properties":{"city":{"type":"string"},"n":{"type":"integer"},"unit":{"enum":["c","f"]}},"required":["city","n","unit"],"additionalProperties":false}}`))
	if err != nil {
		t.Fatal(err)
	}
	const start, end, quote = 48, 49, 52
	g, err := GemmaToolCalls([]pyjson.Value{f}, start, end, quote, false)
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		text string
		ok   bool
	}{
		{`call:f{city:§東京§,n:2,unit:§c§}`, true},
		{`call:f{city:§東京§,unit:§c§}`, false},
		{`call:f{city:§東京§,n:2.5,unit:§c§}`, false},
		{`call:f{city:§東京§,n:2,unit:§x§}`, false},
		{`call:f{city:§a",}\b§,n:2,unit:§f§}`, true},
	}
	for _, item := range cases {
		state := g.Initial().AdvanceToken(g, start)
		for i, part := range strings.Split(item.text, "§") {
			if i > 0 {
				state = state.AdvanceToken(g, quote)
			}
			state = state.AdvanceBytes(g, []byte(part))
		}
		state = state.AdvanceToken(g, end)
		if state.Accepting() != item.ok {
			t.Errorf("accepts %s = %v", item.text, state.Accepting())
		}
	}
}
