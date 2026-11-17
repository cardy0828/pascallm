// Package chat は、LFM3 のチャットテンプレート（chat_template.jinja）のうち、文字の会話とツールの部分を
// Go で書き直したものです（M3、ツールは M6）。Jinja の実行系は持ちません。
//
// 対応する役割は system・user・assistant・tool です。画像・音声は扱いません（呼び出し側で断る）。
// 結果は tokenizer.Segment の並びで、テンプレートが書いた部分だけを Control にします。
// こうすると、利用者の本文やツールの結果にある "<|im_end|>" などの文字列は特殊トークンになりません。
package chat

import (
	"fmt"
	"strings"

	"github.com/cardy0828/pascallm/internal/pyjson"
	"github.com/cardy0828/pascallm/internal/tokenizer"
)

// テンプレートが書く文字列です（chat_template.jinja と同じ）。
const (
	BeginOfText    = "<|startoftext|>"
	MessageStart   = "<|im_start|>"
	MessageEnd     = "<|im_end|>"
	ThinkStart     = "<think>"
	ThinkEnd       = "</think>"
	EmptyThinking  = ThinkStart + ThinkEnd
	messageEndLine = MessageEnd + "\n"
)

// ツールのためにテンプレートが書く文字列です（chat_template.jinja と同じ）。
const (
	ToolsStart           = "<tools>\n"
	ToolsEnd             = "\n</tools>"
	ToolCallStart        = "<tool_call>"
	ToolCallEnd          = "</tool_call>"
	FunctionStart        = "<function="
	FunctionEnd          = "</function>"
	ParameterStart       = "<parameter="
	ParameterEnd         = "</parameter>"
	ToolResponseStart    = "<tool_response>"
	ToolResponseEnd      = "</tool_response>"
	AdditionalToolsStart = "<additional_tools>"
	AdditionalToolsEnd   = "</additional_tools>"
	toolCallSeparator    = "\n\n"
)

// 扱う役割です。
const (
	RoleSystem    = "system"
	RoleUser      = "user"
	RoleAssistant = "assistant"
	RoleTool      = "tool"
)

// Message は会話の 1 発言です。
type Message struct {
	Name         string   // tool 応答の関数名（tool_call_id を優先）。
	ContentParts []string // text 部品ごとの境界（Gemma の trim 用）。nil は文字列の本文。
	Role         string   // "system"・"user"・"assistant"・"tool"
	Content      string   // system・user・assistant の本文。例: "こんにちは"
	// Reasoning は assistant の思考です（API の reasoning_content）。nil は「指定なし」で、空文字列とは区別します
	// （テンプレートが「定義されているか」で分岐するため）。
	Reasoning *string
	// ToolCalls は assistant が行ったツールの呼び出しです。
	ToolCalls []ToolCall
	// ToolCallID は tool の発言が答える呼び出しの id です。同じ id の tool が続くと 1 つにまとめます。
	ToolCallID string
	// ToolContent は tool の発言の content です（文字列・null・部品の配列）。
	// 部品は {"type": "text", "text": ...} と {"type": "tool_reference", "name": ...} などです。
	ToolContent pyjson.Value
	// AdditionalTools は tool の発言の additional_tools です（nil なら指定なし）。
	AdditionalTools *pyjson.Value
}

// ToolCall は assistant のツールの呼び出し 1 件です。
type ToolCall struct {
	ID   string // OpenAI の呼び出し ID。tool 応答の名前解決に使う。
	Name string // 関数の名前。例: "get_weather"
	// Arguments は引数のオブジェクトです（キーの順を保つ）。オブジェクトでなければ引数なしとして書きます。
	Arguments pyjson.Value
}

// Options はテンプレートの設定です。
type Options struct {
	// EnableThinking が true なら生成の入口を "<think>"、false なら "<think></think>" にします。
	// テンプレートの既定は true です。
	EnableThinking bool
	// AddGenerationPrompt が true なら、最後に assistant の発言の入口を付けます（生成するときは true）。
	AddGenerationPrompt bool
	// Tools は要求の tools です（1 個は {"type": "function", "function": {...}} の形。キーの順を保つ）。
	Tools []pyjson.Value
}

// writer は、テンプレートが書いた部分（Control）と本文を区別して区間を集めます。
type writer struct {
	segments []tokenizer.Segment
}

// control はテンプレートが書いた文字列を足します。例: w.control("<|im_start|>user\n")
func (w *writer) control(text string) {
	if text != "" {
		w.segments = append(w.segments, tokenizer.Segment{Text: text, Control: true})
	}
}

// content は利用者やツールから来た文字列を足します。例: w.content("こんにちは")
func (w *writer) content(text string) {
	if text != "" {
		w.segments = append(w.segments, tokenizer.Segment{Text: text})
	}
}

// Render は会話をテンプレートに当てはめ、トークナイザーに渡す区間の並びにします。
// 例: [{user "こんにちは"}]、思考あり →
//
//	"<|startoftext|>" "<|im_start|>user\n" "こんにちは" "<|im_end|>\n" "<|im_start|>assistant\n<think>"
//
// （本文の「こんにちは」だけが Control=false）
func Render(messages []Message, options Options) ([]tokenizer.Segment, error) {
	w := &writer{}
	w.control(BeginOfText)

	systemText := ""
	if len(messages) > 0 && messages[0].Role == RoleSystem {
		systemText = messages[0].Content
		messages = messages[1:]
	}
	toolLines, err := toolDefinitionLines(options.Tools)
	if err != nil {
		return nil, err
	}
	writeSystem(w, systemText, toolLines)

	lastUser := lastUserIndex(messages)
	var pending []pyjson.Value // まとめる途中の tool の応答の部品
	for index, message := range messages {
		if err := checkRole(message.Role); err != nil {
			return nil, err
		}
		if message.Role == RoleTool && message.AdditionalTools == nil {
			items := toolItems(message.ToolContent)
			if mergesWithNext(messages, index) {
				pending = append(pending, items...)
				continue
			}
			w.control(MessageStart + RoleTool + "\n")
			if len(pending) > 0 {
				err = renderToolResponse(w, pyjson.Value{Kind: pyjson.Array, Items: append(pending, items...)}, options.Tools)
				pending = nil
			} else {
				err = renderToolResponse(w, noneAsEmpty(message.ToolContent), options.Tools)
			}
			if err != nil {
				return nil, err
			}
			w.control(messageEndLine)
			continue
		}
		w.control(MessageStart + message.Role + "\n")
		switch message.Role {
		case RoleAssistant:
			renderAssistant(w, message, index > lastUser)
		case RoleTool:
			if err := renderAdditionalTools(w, message); err != nil {
				return nil, err
			}
		default:
			w.content(message.Content)
		}
		w.control(messageEndLine)
	}

	if options.AddGenerationPrompt {
		if options.EnableThinking {
			w.control(MessageStart + RoleAssistant + "\n" + ThinkStart)
		} else {
			w.control(MessageStart + RoleAssistant + "\n" + EmptyThinking)
		}
	}
	return w.segments, nil
}

// writeSystem は system の発言を書きます。本文とツールの定義の両方が空なら何も書きません。
// ツールの定義は、本文があれば空行を挟んで後ろに付けます。
// 例: ("あなたは助手です。", [`{"name": "f"}`]) → "<|im_start|>system\nあなたは助手です。\n\n<tools>\n{...}\n</tools><|im_end|>\n"
func writeSystem(w *writer, systemText string, toolLines []string) {
	if systemText == "" && len(toolLines) == 0 {
		return
	}
	w.control(MessageStart + RoleSystem + "\n")
	w.content(systemText)
	if len(toolLines) > 0 {
		if systemText != "" {
			w.control("\n\n")
		}
		w.control(ToolsStart)
		w.content(strings.Join(toolLines, "\n"))
		w.control(ToolsEnd)
	}
	w.control(messageEndLine)
}

// toolDefinitionLines は、tools のうち遅延読み込み（defer_loading）でない関数の定義を、1 行 1 個の JSON にします。
// 例: [{"type": "function", "function": {"name": "f"}}] → [`{"name": "f"}`]
func toolDefinitionLines(tools []pyjson.Value) ([]string, error) {
	var lines []string
	for _, tool := range tools {
		function := functionOf(tool)
		if isDeferred(tool, function) {
			continue
		}
		line, err := toolToJSON(tool)
		if err != nil {
			return nil, err
		}
		lines = append(lines, line)
	}
	return lines, nil
}

// functionOf は、ツールの定義のうち関数の部分を返します（テンプレートの tool.function if defined else tool）。
// 例: {"type": "function", "function": {"name": "f"}} → {"name": "f"}
func functionOf(tool pyjson.Value) pyjson.Value {
	if function, ok := tool.Get("function"); ok {
		return function
	}
	return tool
}

// isDeferred は、関数かツールの defer_loading が真かを返します。
func isDeferred(tool, function pyjson.Value) bool {
	if deferred, ok := function.Get("defer_loading"); ok && deferred.Truthy() {
		return true
	}
	deferred, ok := tool.Get("defer_loading")
	return ok && deferred.Truthy()
}

// toolToJSON は、ツールの関数の定義を、defer_loading を除いた JSON にします（テンプレートの tool_to_json）。
// 例: {"name": "f", "defer_loading": true, "parameters": {}} → `{"name": "f", "parameters": {}}`
func toolToJSON(tool pyjson.Value) (string, error) {
	function := functionOf(tool)
	if function.Kind != pyjson.Object {
		return "", fmt.Errorf("chat: tool definition must be an object, got %s", function.Dumps())
	}
	trimmed := pyjson.NewObject()
	for _, member := range function.Members {
		if member.Key != "defer_loading" {
			trimmed.Members = append(trimmed.Members, member)
		}
	}
	return trimmed.Dumps(), nil
}

// checkRole は、役割が扱えるものかを確かめます。例: checkRole("developer") → エラー
func checkRole(role string) error {
	switch role {
	case RoleSystem, RoleUser, RoleAssistant, RoleTool:
		return nil
	default:
		return fmt.Errorf("chat: role %q is not supported", role)
	}
}

// lastUserIndex は、最後の user の発言の番号を返します（なければ -1）。例: [user, assistant, user] → 2
func lastUserIndex(messages []Message) int {
	last := -1
	for index, message := range messages {
		if message.Role == RoleUser {
			last = index
		}
	}
	return last
}

// renderAssistant は assistant の発言の中身を書きます。
// keepThinking（最後の user より後の発言）なら思考を残し、それより前の発言は思考を除いて
// "<think></think>" を付けます（chat_template.jinja と同じ規則）。本文の後にツールの呼び出しを書きます。
// 例: 最後の user より前の {Content: "a</think>\n答え"} → "<think></think>" + "答え"
func renderAssistant(w *writer, message Message, keepThinking bool) {
	body := message.Content
	switch {
	case keepThinking && message.Reasoning != nil:
		w.control(ThinkStart)
		w.content(*message.Reasoning)
		w.control(ThinkEnd)
	case keepThinking && !strings.Contains(body, ThinkEnd):
		w.control(EmptyThinking)
	case !keepThinking:
		if index := strings.LastIndex(body, ThinkEnd); index >= 0 {
			body = trimJinja(body[index+len(ThinkEnd):])
		}
		w.control(EmptyThinking)
	}
	w.content(body)
	renderToolCalls(w, message.ToolCalls)
}

// renderToolCalls はツールの呼び出しを書きます（テンプレートの render_tool_calls）。複数なら空行でつなぎます。
// 例: {get_weather {"city": "Paris"}} →
//
//	<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>
func renderToolCalls(w *writer, calls []ToolCall) {
	for index, call := range calls {
		if index > 0 {
			w.control(toolCallSeparator)
		}
		w.control(ToolCallStart + "\n" + FunctionStart)
		w.content(call.Name)
		w.control(">\n")
		if call.Arguments.Kind == pyjson.Object {
			for _, argument := range call.Arguments.Members {
				w.control(ParameterStart)
				w.content(argument.Key)
				w.control(">\n")
				w.content(argument.Value.PythonStr())
				w.control("\n" + ParameterEnd + "\n")
			}
		}
		w.control(FunctionEnd + "\n" + ToolCallEnd)
	}
}

// mergesWithNext は、messages[index] の tool の応答を次の発言とまとめるか（テンプレートの _merge）を返します。
// 次も tool で、どちらも additional_tools がなく、tool_call_id が空でなく同じ場合です。
func mergesWithNext(messages []Message, index int) bool {
	if index+1 >= len(messages) {
		return false
	}
	current, next := messages[index], messages[index+1]
	return current.Role == RoleTool && current.AdditionalTools == nil &&
		next.Role == RoleTool && next.AdditionalTools == nil &&
		current.ToolCallID != "" && next.ToolCallID == current.ToolCallID
}

// toolItems は、tool の content を、まとめるための部品の並びにします（テンプレートの _items）。
// 例: "晴れ" → [{"type": "text", "text": "晴れ"}]、"" → []、null → []、[a, b] → [a, b]
func toolItems(content pyjson.Value) []pyjson.Value {
	switch content.Kind {
	case pyjson.Null:
		return nil
	case pyjson.String:
		if content.String == "" {
			return nil
		}
		item := pyjson.NewObject()
		item.Set("type", pyjson.NewString("text"))
		item.Set("text", content)
		return []pyjson.Value{item}
	case pyjson.Array:
		return content.Items
	default:
		return []pyjson.Value{content}
	}
}

// noneAsEmpty は、null を空文字列にします（テンプレートの "" if content is none else content）。
func noneAsEmpty(content pyjson.Value) pyjson.Value {
	if content.Kind == pyjson.Null {
		return pyjson.NewString("")
	}
	return content
}

// renderToolResponse は tool の応答を "<tool_response>\n…\n</tool_response>" で包んで書きます
// （テンプレートの render_tool_response）。文字列がすでに "<tool_response>" を含むならそのまま書きます。
// 配列の部品は改行でつなぎ、tool_reference は参照した関数の定義を <additional_tools> に入れます。
// 例: "晴れ" → "<tool_response>\n晴れ\n</tool_response>"
func renderToolResponse(w *writer, content pyjson.Value, tools []pyjson.Value) error {
	if content.Kind == pyjson.String && strings.Contains(content.String, ToolResponseStart) {
		w.content(content.String)
		return nil
	}
	w.control(ToolResponseStart + "\n")
	switch content.Kind {
	case pyjson.String:
		w.content(content.String)
	case pyjson.Object:
		text, err := parseContent(content)
		if err != nil {
			return err
		}
		w.content(text)
	case pyjson.Array:
		if err := renderResponseItems(w, content.Items, tools); err != nil {
			return err
		}
	default:
		w.content(content.Dumps())
	}
	w.control("\n" + ToolResponseEnd)
	return nil
}

// renderResponseItems は tool の応答の部品を改行でつないで書きます。続いた tool_reference は 1 つの
// <additional_tools> にまとめます。result.tools を持つ部品は、その定義を <additional_tools> に入れます。
func renderResponseItems(w *writer, items []pyjson.Value, tools []pyjson.Value) error {
	for index, item := range items {
		if index > 0 {
			w.control("\n")
		}
		if isToolReference(item) {
			if index == 0 || !isToolReference(items[index-1]) {
				w.control(AdditionalToolsStart + "\n")
			}
			name, _ := item.Get("name")
			definition, err := referencedToolJSON(name, tools)
			if err != nil {
				return err
			}
			w.content(definition)
			if index == len(items)-1 || !isToolReference(items[index+1]) {
				w.control("\n" + AdditionalToolsEnd)
			}
			continue
		}
		if resultTools, ok := resultToolsOf(item); ok {
			w.control(AdditionalToolsStart + "\n")
			if err := writeToolList(w, resultTools); err != nil {
				return err
			}
			w.control("\n" + AdditionalToolsEnd)
			continue
		}
		text, err := parseContent(item)
		if err != nil {
			return err
		}
		w.content(text)
	}
	return nil
}

// isToolReference は、部品が {"type": "tool_reference", ...} かを返します。
func isToolReference(item pyjson.Value) bool {
	kind, ok := item.GetString("type")
	return ok && kind == "tool_reference"
}

// resultToolsOf は、部品が {"result": {"tools": ...}} の形なら tools の値を返します。
func resultToolsOf(item pyjson.Value) (pyjson.Value, bool) {
	result, ok := item.Get("result")
	if !ok || result.Kind != pyjson.Object {
		return pyjson.Value{}, false
	}
	return result.Get("tools")
}

// writeToolList は、ツールの定義（文字列・1 個・配列）を書きます。配列は 1 行 1 個です。
func writeToolList(w *writer, list pyjson.Value) error {
	switch list.Kind {
	case pyjson.String:
		w.content(list.String)
		return nil
	case pyjson.Object:
		line, err := toolToJSON(list)
		if err != nil {
			return err
		}
		w.content(line)
		return nil
	case pyjson.Array:
		lines := make([]string, 0, len(list.Items))
		for _, tool := range list.Items {
			line, err := toolToJSON(tool)
			if err != nil {
				return err
			}
			lines = append(lines, line)
		}
		w.content(strings.Join(lines, "\n"))
		return nil
	default:
		return fmt.Errorf("chat: tool list must be a string, an object or an array, got %s", list.Dumps())
	}
}

// referencedToolJSON は、tools から名前の合う関数を探し、定義の JSON を返します（同じ名前が複数あれば最後のもの）。
// 見つからなければエラーです（テンプレートも例外にする）。例: "get_weather" → `{"name": "get_weather", ...}`
func referencedToolJSON(name pyjson.Value, tools []pyjson.Value) (string, error) {
	var found *pyjson.Value
	for index := range tools {
		function := functionOf(tools[index])
		if functionName, ok := function.Get("name"); ok && functionName.Kind == name.Kind && functionName.Dumps() == name.Dumps() {
			found = &function
		}
	}
	if found == nil {
		return "", fmt.Errorf("chat: tool reference %s is not found in tools; every referenced tool must be in tools on every request", name.Dumps())
	}
	return toolToJSON(*found)
}

// renderAdditionalTools は、additional_tools を持つ tool の発言の中身を書きます。
// additional_tools がオブジェクトか配列ならそれを、そうでなければ content を、ツールの定義として書きます。
func renderAdditionalTools(w *writer, message Message) error {
	source := message.ToolContent
	if additional := *message.AdditionalTools; additional.Kind == pyjson.Object || additional.Kind == pyjson.Array {
		source = additional
	}
	switch {
	case source.Kind == pyjson.String && strings.Contains(source.String, ToolResponseStart):
		w.content(source.String)
	case source.Kind == pyjson.String && strings.Contains(source.String, AdditionalToolsStart):
		w.control(ToolResponseStart + "\n")
		w.content(source.String)
		w.control("\n" + ToolResponseEnd)
	case source.Truthy():
		w.control(ToolResponseStart + "\n" + AdditionalToolsStart + "\n")
		if err := writeToolList(w, source); err != nil {
			return err
		}
		w.control("\n" + AdditionalToolsEnd + "\n" + ToolResponseEnd)
	}
	return nil
}

// parseContent は content の値を文字列にします（テンプレートの parse_content。画像・動画は扱わない）。
// 文字列はそのまま、{"type": "text"} はその text、ほかのオブジェクトは JSON、配列は各部品をつなげたものです。
// 例: [{"type": "text", "text": "a"}, {"k": 1}] → `a{"k": 1}`
func parseContent(content pyjson.Value) (string, error) {
	switch content.Kind {
	case pyjson.Null:
		return "", nil
	case pyjson.String:
		return content.String, nil
	case pyjson.Object:
		return contentPart(content)
	case pyjson.Array:
		var builder strings.Builder
		for _, item := range content.Items {
			if item.Kind == pyjson.String {
				builder.WriteString(item.String)
				continue
			}
			if item.Kind != pyjson.Object {
				builder.WriteString(item.Dumps())
				continue
			}
			text, err := contentPart(item)
			if err != nil {
				return "", err
			}
			builder.WriteString(text)
		}
		return builder.String(), nil
	default:
		return content.Dumps(), nil
	}
}

// contentPart は content の部品（オブジェクト）1 個を文字列にします。画像・動画はエラーです。
// 例: {"type": "text", "text": "a"} → "a"、{"type": "text"} → ""、{"k": 1} → `{"k": 1}`
func contentPart(part pyjson.Value) (string, error) {
	kind, _ := part.GetString("type")
	switch kind {
	case "image", "video", "image_url", "input_audio":
		return "", fmt.Errorf("chat: content part type %q is not supported (text only)", kind)
	case "text":
		text, ok := part.Get("text")
		if !ok || !text.Truthy() {
			return "", nil
		}
		if text.Kind != pyjson.String && text.Kind != pyjson.Number && text.Kind != pyjson.Bool {
			return "", fmt.Errorf("chat: text part must be a string, got %s", text.Dumps())
		}
		return pythonString(text), nil
	default:
		return part.Dumps(), nil
	}
}

// pythonString は、文字列・数・真偽値を Python の str() にします（Jinja の string フィルター）。
// 例: "a" → "a"、3 → "3"、true → "True"
func pythonString(value pyjson.Value) string {
	switch value.Kind {
	case pyjson.String:
		return value.String
	case pyjson.Bool:
		if value.Bool {
			return "True"
		}
		return "False"
	default:
		return value.PythonNumber()
	}
}

// trimJinja は Jinja の trim フィルターと同じく、前後の空白を取り除きます（Python の str.strip()）。
// 例: "\n 答え \n" → "答え"
func trimJinja(text string) string {
	return strings.TrimSpace(text)
}
