package chat

import (
	"fmt"
	"github.com/cardy0828/pascallm/internal/pyjson"
	"github.com/cardy0828/pascallm/internal/tokenizer"
	"sort"
	"strings"
)

// Gemma の制御文字列です。12B QAT GGUF 内の tokenizer.chat_template に対応します。
const (
	GemmaTurnStart     = "<|turn>"
	GemmaTurnEnd       = "<turn|>"
	GemmaThink         = "<|think|>"
	GemmaChannelStart  = "<|channel>"
	GemmaChannelEnd    = "<channel|>"
	GemmaCallStart     = "<|tool_call>"
	GemmaCallEnd       = "<tool_call|>"
	GemmaResponseStart = "<|tool_response>"
	GemmaResponseEnd   = "<tool_response|>"
	GemmaQuote         = "<|\"|>"
)

// RenderGemma は Gemma 4 の文字対話と OpenAI 形式のツール履歴を描画します。
// 例: user「こんにちは」、思考なし → <bos><|turn>user\nこんにちは<turn|>\n<|turn>model\n<|channel>thought\n<channel|>
func RenderGemma(messages []Message, options Options) ([]tokenizer.Segment, error) {
	w := &writer{}
	w.control("<bos>")
	firstSystem := len(messages) > 0 && (messages[0].Role == RoleSystem || messages[0].Role == "developer")
	if options.EnableThinking || len(options.Tools) > 0 || firstSystem {
		w.control(GemmaTurnStart + "system\n")
		if options.EnableThinking {
			w.control(GemmaThink + "\n")
		}
		if firstSystem {
			if messages[0].ContentParts != nil {
				for _, part := range messages[0].ContentParts {
					w.content(strings.TrimSpace(part) + " ")
				}
			} else {
				w.content(strings.TrimSpace(messages[0].Content))
			}
			messages = messages[1:]
		}
		for _, tool := range options.Tools {
			w.control("<|tool>")
			if err := gemmaDeclaration(w, functionOf(tool)); err != nil {
				return nil, err
			}
			w.control("<tool|>")
		}
		w.control(GemmaTurnEnd + "\n")
	}
	previousRole, previousType := "", ""
	lastUser := lastUserIndex(messages)
	for index, message := range messages {
		if message.Role == RoleTool {
			// tool は直前の assistant に続くブロックとして描画済みです。孤立した応答は拒否します。
			owner := index - 1
			for owner >= 0 && messages[owner].Role == RoleTool {
				owner--
			}
			if owner < 0 || messages[owner].Role != RoleAssistant || len(messages[owner].ToolCalls) == 0 {
				return nil, fmt.Errorf("chat: orphan tool response")
			}
			continue
		}
		if message.Role != RoleUser && message.Role != RoleAssistant && message.Role != RoleSystem && message.Role != "developer" {
			return nil, fmt.Errorf("chat: unsupported Gemma role %q", message.Role)
		}
		previousType = ""
		role := message.Role
		if role == RoleAssistant {
			role = "model"
		}
		if !(role == "model" && previousRole == RoleAssistant) {
			w.control(GemmaTurnStart + role + "\n")
		}
		if message.Reasoning != nil && *message.Reasoning != "" && index > lastUser {
			w.control(GemmaChannelStart + "thought\n")
			w.content(*message.Reasoning)
			w.control("\n" + GemmaChannelEnd)
		}
		for _, call := range message.ToolCalls {
			if !GemmaIdentifier(call.Name) {
				return nil, fmt.Errorf("chat: invalid Gemma function name %q", call.Name)
			}
			w.control(GemmaCallStart + "call:")
			w.content(call.Name)
			args := call.Arguments
			if args.Kind == pyjson.Null {
				args = pyjson.NewObject()
			}
			gemmaArgument(w, args, false)
			w.control(GemmaCallEnd)
			previousType = "tool_call"
		}
		hasResponses := false
		nextIndex := index + 1
		for nextIndex < len(messages) && messages[nextIndex].Role == RoleTool {
			follow := messages[nextIndex]
			if follow.AdditionalTools != nil {
				return nil, fmt.Errorf("chat: additional_tools is only supported for LFM3")
			}
			name := follow.Name
			for _, call := range message.ToolCalls {
				if call.ID != "" && call.ID == follow.ToolCallID {
					name = call.Name
				}
			}
			if name == "" {
				return nil, fmt.Errorf("chat: tool response has no matching tool_call_id or name")
			}
			if !GemmaIdentifier(name) {
				return nil, fmt.Errorf("chat: invalid tool response name")
			}
			body := follow.ToolContent
			if body.Kind == pyjson.Array {
				for _, part := range body.Items {
					kind, _ := part.GetString("type")
					if kind != "text" {
						return nil, fmt.Errorf("chat: Gemma tool content type %q is not supported (text only)", kind)
					}
				}
				text, err := parseContent(body)
				if err != nil {
					return nil, err
				}
				body = pyjson.NewString(text)
			}
			w.control(GemmaResponseStart + "response:")
			w.content(name)
			if body.Kind != pyjson.Object {
				body = pyjson.Value{Kind: pyjson.Object, Members: []pyjson.Member{{Key: "value", Value: body}}}
			}
			gemmaArgument(w, body, false)
			w.control(GemmaResponseEnd)
			hasResponses = true
			previousType = "tool_response"
			nextIndex++
		}
		content := gemmaMessageContent(message)
		w.content(content)
		continues := role == "model" && nextIndex < len(messages) && messages[nextIndex].Role == RoleAssistant && (len(message.ToolCalls) == 0 || hasResponses)
		switch {
		case previousType == "tool_call" && !hasResponses:
			w.control(GemmaResponseStart)
		case continues:
		case hasResponses && strings.TrimSpace(content) == "" && nextIndex == len(messages):
		default:
			w.control(GemmaTurnEnd + "\n")
		}
		previousRole = message.Role
	}
	if options.AddGenerationPrompt {
		if previousType != "tool_response" && previousType != "tool_call" {
			w.control(GemmaTurnStart + "model\n")
			if !options.EnableThinking {
				w.control(GemmaChannelStart + "thought\n" + GemmaChannelEnd)
			}
		} else if previousType == "tool_response" && options.EnableThinking {
			w.control(GemmaChannelStart + "thought\n")
		}
	}
	return w.segments, nil
}

// gemmaMessageContent は部品ごとに空白を除き、assistant の過去の思考チャンネルを取り除きます。
func gemmaMessageContent(message Message) string {
	parts := message.ContentParts
	if parts == nil {
		parts = []string{message.Content}
	}
	var out strings.Builder
	for _, part := range parts {
		if message.Role == RoleAssistant {
			var clean strings.Builder
			for _, segment := range strings.Split(part, GemmaChannelEnd) {
				before, _, _ := strings.Cut(segment, GemmaChannelStart)
				clean.WriteString(before)
			}
			part = clean.String()
		}
		out.WriteString(strings.TrimSpace(part))
	}
	return out.String()
}

// GemmaIdentifier は DSL の裸の名前として使えるかを判定します。例: get_weather・city は true。
func GemmaIdentifier(name string) bool {
	if name == "" {
		return false
	}
	for _, r := range name {
		if !(r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '_' || r == '-' || r == '.') {
			return false
		}
	}
	return true
}

// gemmaMembers は Jinja の dictsort と同じく、キーの大小文字を区別せず安定して並べます。
func gemmaMembers(value pyjson.Value) []pyjson.Member {
	result := append([]pyjson.Member(nil), value.Members...)
	sort.SliceStable(result, func(i, j int) bool { return strings.ToLower(result[i].Key) < strings.ToLower(result[j].Key) })
	return result
}

// gemmaString は引用符だけを制御トークンとして書き、利用者の文字列は通常の本文として書きます。
func gemmaString(w *writer, text string) {
	w.control(GemmaQuote)
	w.content(text)
	w.control(GemmaQuote)
}

// gemmaArgument はテンプレートの format_argument です。例: {city:"東京"} → {city:<|"|>東京<|"|>}。
func gemmaArgument(w *writer, value pyjson.Value, quoteKeys bool) {
	switch value.Kind {
	case pyjson.String:
		gemmaString(w, value.String)
	case pyjson.Object:
		w.control("{")
		for i, m := range gemmaMembers(value) {
			if i > 0 {
				w.control(",")
			}
			if quoteKeys {
				gemmaString(w, m.Key)
			} else {
				w.content(m.Key)
			}
			w.control(":")
			gemmaArgument(w, m.Value, quoteKeys)
		}
		w.control("}")
	case pyjson.Array:
		w.control("[")
		for i, item := range value.Items {
			if i > 0 {
				w.control(",")
			}
			gemmaArgument(w, item, quoteKeys)
		}
		w.control("]")
	default:
		w.content(value.Dumps())
	}
}

// gemmaDeclaration はツール宣言をテンプレートと同じキー順で書きます。
func gemmaDeclaration(w *writer, function pyjson.Value) error {
	name, _ := function.GetString("name")
	if !GemmaIdentifier(name) {
		return fmt.Errorf("chat: invalid Gemma function name %q", name)
	}
	description, _ := function.GetString("description")
	w.control("declaration:")
	w.content(name)
	w.control("{description:")
	gemmaString(w, description)
	params, _ := function.Get("parameters")
	if params.Truthy() {
		w.control(",parameters:{")
		props, _ := params.Get("properties")
		if props.Truthy() {
			w.control("properties:{")
			if err := gemmaProperties(w, props, false); err != nil {
				return err
			}
			w.control("},")
		}
		required, _ := params.Get("required")
		if required.Truthy() {
			w.control("required:")
			gemmaArgument(w, required, true)
			w.control(",")
		}
		kind, _ := params.GetString("type")
		if kind != "" {
			w.control("type:")
			gemmaString(w, strings.ToUpper(kind))
			w.control("}")
		} else {
			return fmt.Errorf("chat: Gemma tool parameters require type")
		}
	}
	if response, ok := function.Get("response"); ok {
		w.control(",response:{")
		description, _ := response.GetString("description")
		if description != "" {
			w.control("description:")
			gemmaString(w, description)
			w.control(",")
		}
		kind, _ := response.GetString("type")
		if strings.EqualFold(kind, "object") {
			w.control("type:")
			gemmaString(w, "OBJECT")
			w.control("}")
		} else {
			return fmt.Errorf("chat: Gemma response declaration requires object type")
		}
	}
	w.control("}")
	return nil
}

// gemmaProperties は format_parameters の再帰部分です。配列の items もキー順を合わせます。
func gemmaProperties(w *writer, props pyjson.Value, filter bool) error {
	first := true
	for _, m := range gemmaMembers(props) {
		if filter && (m.Key == "description" || m.Key == "type" || m.Key == "properties" || m.Key == "required" || m.Key == "nullable") {
			continue
		}
		if !GemmaIdentifier(m.Key) {
			return fmt.Errorf("chat: invalid Gemma property name %q", m.Key)
		}
		if !first {
			w.control(",")
		}
		first = false
		w.content(m.Key)
		w.control(":{")
		fields := 0
		field := func(name string) {
			if fields > 0 {
				w.control(",")
			}
			w.control(name + ":")
			fields++
		}
		value := m.Value
		kind, _ := value.GetString("type")
		description, _ := value.GetString("description")
		if description != "" {
			field("description")
			gemmaString(w, description)
		}
		switch strings.ToUpper(kind) {
		case "STRING":
			if values, ok := value.Get("enum"); ok && values.Truthy() {
				field("enum")
				gemmaArgument(w, values, true)
			}
		case "ARRAY":
			if items, ok := value.Get("items"); ok && items.Kind == pyjson.Object && items.Truthy() {
				field("items")
				w.control("{")
				count := 0
				for _, item := range gemmaMembers(items) {
					if item.Value.Kind == pyjson.Null {
						continue
					}
					if count > 0 {
						w.control(",")
					}
					count++
					w.content(item.Key)
					w.control(":")
					switch item.Key {
					case "properties":
						w.control("{")
						if err := gemmaProperties(w, item.Value, false); err != nil {
							return err
						}
						w.control("}")
					case "type":
						v := item.Value
						if v.Kind == pyjson.String {
							v = pyjson.NewString(strings.ToUpper(v.String))
						} else if v.Kind == pyjson.Array {
							v.Items = append([]pyjson.Value(nil), v.Items...)
							for i := range v.Items {
								v.Items[i] = pyjson.NewString(strings.ToUpper(v.Items[i].String))
							}
						}
						gemmaArgument(w, v, true)
					default:
						gemmaArgument(w, item.Value, true)
					}
				}
				w.control("}")
			}
		}
		if nullable, _ := value.Get("nullable"); nullable.Truthy() {
			field("nullable")
			w.control("true")
		}
		if strings.EqualFold(kind, "object") {
			field("properties")
			w.control("{")
			nested, ok := value.Get("properties")
			if ok && nested.Kind == pyjson.Object {
				if err := gemmaProperties(w, nested, false); err != nil {
					return err
				}
			} else {
				if err := gemmaProperties(w, value, true); err != nil {
					return err
				}
			}
			w.control("}")
			if required, _ := value.Get("required"); required.Truthy() {
				field("required")
				gemmaArgument(w, required, true)
			}
		}
		field("type")
		gemmaString(w, strings.ToUpper(kind))
		w.control("}")
	}
	return nil
}
