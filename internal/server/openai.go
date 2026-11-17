package server

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/cardy0828/pascallm/internal/chat"
	"github.com/cardy0828/pascallm/internal/pyjson"
)

// この file は、OpenAI の Chat Completions API の要求と応答の形です。扱わない項目は読み飛ばします。

// chatRequest は POST /v1/chat/completions の要求です。
type chatRequest struct {
	Model               string          `json:"model"`
	Messages            []chatMessage   `json:"messages"`
	MaxTokens           *int            `json:"max_tokens"`
	MaxCompletionTokens *int            `json:"max_completion_tokens"`
	Temperature         *float64        `json:"temperature"`
	TopP                *float64        `json:"top_p"`
	TopK                *int            `json:"top_k"`              // OpenAI にはない（vLLM・llama.cpp と同じ名前）
	MinP                *float64        `json:"min_p"`              // 同上
	RepetitionPenalty   *float64        `json:"repetition_penalty"` // 同上（vLLM の名前）
	RepeatPenalty       *float64        `json:"repeat_penalty"`     // 同上（llama.cpp の名前）
	PresencePenalty     *float64        `json:"presence_penalty"`
	FrequencyPenalty    *float64        `json:"frequency_penalty"`
	Seed                *int64          `json:"seed"`
	Stop                json.RawMessage `json:"stop"` // 文字列または文字列の配列
	Stream              bool            `json:"stream"`
	StreamOptions       *struct {
		IncludeUsage bool `json:"include_usage"`
	} `json:"stream_options"`
	N                  *int                       `json:"n"`
	Tools              json.RawMessage            `json:"tools"`                // [{"type": "function", "function": {...}}]
	ToolChoice         json.RawMessage            `json:"tool_choice"`          // "auto"・"none"・"required"・{"type": "function", ...}
	ParallelToolCalls  *bool                      `json:"parallel_tool_calls"`  // false なら呼び出しは 1 件まで
	ResponseFormat     json.RawMessage            `json:"response_format"`      // {"type": "json_object"} など
	ChatTemplateKwargs map[string]json.RawMessage `json:"chat_template_kwargs"` // 例: {"enable_thinking": false}
}

// chatMessage は要求の 1 発言です。content は文字列、null、または {"type": "text", "text": ...} の配列です
// （tool の content は tool_reference などの部品も含められる）。
type chatMessage struct {
	Role             string          `json:"role"`
	Content          json.RawMessage `json:"content"`
	ReasoningContent *string         `json:"reasoning_content"`
	Reasoning        *string         `json:"reasoning"`  // reasoning_content の別名（vLLM の新しい名前）
	ToolCalls        json.RawMessage `json:"tool_calls"` // assistant の呼び出し
	Name             string          `json:"name"`
	ToolCallID       string          `json:"tool_call_id"`     // tool が答える呼び出しの id
	AdditionalTools  json.RawMessage `json:"additional_tools"` // tool で渡す追加のツールの定義（LFM3 の拡張）
}

// maxStops は stop に指定できる文字列の最大数です（OpenAI と同じ）。
const maxStops = 4

// toChatMessage は要求の発言を chat.Message にします。
// 例: {"role": "user", "content": [{"type": "text", "text": "こんにちは"}]} → {user こんにちは}
func (m chatMessage) toChatMessage() (chat.Message, error) {
	if m.Role == chat.RoleTool {
		return m.toToolMessage()
	}
	content, err := parseContent(m.Content)
	if err != nil {
		return chat.Message{}, fmt.Errorf("message with role %q: %w", m.Role, err)
	}
	message := chat.Message{Role: m.Role, Content: content}
	if bytes.HasPrefix(bytes.TrimSpace(m.Content), []byte("[")) {
		var parts []struct {
			Text string `json:"text"`
		}
		if err := json.Unmarshal(m.Content, &parts); err != nil {
			return chat.Message{}, err
		}
		message.ContentParts = []string{}
		for _, part := range parts {
			message.ContentParts = append(message.ContentParts, part.Text)
		}
	}
	if m.Role != chat.RoleAssistant {
		return message, nil
	}
	message.Reasoning = m.ReasoningContent
	if message.Reasoning == nil {
		message.Reasoning = m.Reasoning
	}
	if message.ToolCalls, err = parseToolCalls(m.ToolCalls); err != nil {
		return chat.Message{}, fmt.Errorf("message with role %q: %w", m.Role, err)
	}
	return message, nil
}

// toToolMessage は tool の発言を chat.Message にします。content は部品の形のまま渡します（テンプレートが解釈する）。
func (m chatMessage) toToolMessage() (chat.Message, error) {
	content, err := parseOptionalJSON(m.Content)
	if err != nil {
		return chat.Message{}, fmt.Errorf("message with role %q: content: %w", m.Role, err)
	}
	message := chat.Message{Role: chat.RoleTool, ToolCallID: m.ToolCallID, Name: m.Name, ToolContent: content}
	additional, err := parseOptionalJSON(m.AdditionalTools)
	if err != nil {
		return chat.Message{}, fmt.Errorf("message with role %q: additional_tools: %w", m.Role, err)
	}
	if additional.Truthy() {
		message.AdditionalTools = &additional
	}
	return message, nil
}

// parseOptionalJSON は JSON を読みます。空か null なら null の値です。例: `"晴れ"` → 文字列の値
func parseOptionalJSON(raw json.RawMessage) (pyjson.Value, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 {
		return pyjson.Value{}, nil
	}
	return pyjson.Parse(trimmed)
}

// parseContent は content を文字列にします。配列なら text の部分をつなげます（テンプレートと同じ）。
// 例: "abc" → "abc"、null → ""、[{"type":"text","text":"a"},{"type":"text","text":"b"}] → "ab"
func parseContent(raw json.RawMessage) (string, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || string(trimmed) == "null" {
		return "", nil
	}
	var text string
	if err := json.Unmarshal(trimmed, &text); err == nil {
		return text, nil
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(trimmed, &parts); err != nil {
		return "", errors.New("content must be a string or an array of text parts")
	}
	var joined string
	for _, part := range parts {
		if part.Type != "text" {
			return "", fmt.Errorf("content part type %q is not supported (text only)", part.Type)
		}
		joined += part.Text
	}
	return joined, nil
}

// requestToolCall は、要求の assistant の tool_calls の 1 件です。
type requestToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"` // JSON の文字列（OpenAI の形）またはオブジェクト
	} `json:"function"`
}

// parseToolCalls は assistant の tool_calls を読みます。arguments は JSON の文字列を読み直してオブジェクトにします
// （テンプレートは文字列を受け付けないため。vLLM も同じことをする）。
// 例: [{"function": {"name": "f", "arguments": "{\"x\": 1}"}}] → [{f {"x": 1}}]
func parseToolCalls(raw json.RawMessage) ([]chat.ToolCall, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || string(trimmed) == "null" {
		return nil, nil
	}
	var calls []requestToolCall
	if err := json.Unmarshal(trimmed, &calls); err != nil {
		return nil, errors.New("tool_calls must be an array of {\"type\": \"function\", \"function\": {...}}")
	}
	result := make([]chat.ToolCall, 0, len(calls))
	for index, call := range calls {
		if call.Function.Name == "" {
			return nil, fmt.Errorf("tool_calls[%d].function.name must not be empty", index)
		}
		arguments, err := parseArguments(call.Function.Arguments)
		if err != nil {
			return nil, fmt.Errorf("tool_calls[%d].function.arguments: %w", index, err)
		}
		result = append(result, chat.ToolCall{ID: call.ID, Name: call.Function.Name, Arguments: arguments})
	}
	return result, nil
}

// parseArguments は arguments を読みます。JSON の文字列ならその中身を、オブジェクトならそのまま読みます。
// 空・null・"" は引数なし（null の値）です。中身がオブジェクトでなければエラーです。
// 例: `"{\"city\": \"Paris\"}"` → {"city": "Paris"}
func parseArguments(raw json.RawMessage) (pyjson.Value, error) {
	value, err := parseOptionalJSON(raw)
	if err != nil {
		return pyjson.Value{}, err
	}
	if value.Kind == pyjson.String {
		inner := bytes.TrimSpace([]byte(value.String))
		if len(inner) == 0 {
			return pyjson.Value{}, nil
		}
		if value, err = pyjson.Parse(inner); err != nil {
			return pyjson.Value{}, fmt.Errorf("must be a JSON object: %w", err)
		}
	}
	switch value.Kind {
	case pyjson.Null, pyjson.Object:
		return value, nil
	default:
		return pyjson.Value{}, errors.New("must be a JSON object")
	}
}

// parseStop は stop を文字列の配列にします。例: "\n" → ["\n"]、["a", "b"] → ["a", "b"]、null → []
func parseStop(raw json.RawMessage) ([]string, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || string(trimmed) == "null" {
		return nil, nil
	}
	var single string
	if err := json.Unmarshal(trimmed, &single); err == nil {
		return []string{single}, nil
	}
	var list []string
	if err := json.Unmarshal(trimmed, &list); err != nil {
		return nil, errors.New("stop must be a string or an array of strings")
	}
	if len(list) > maxStops {
		return nil, fmt.Errorf("stop accepts at most %d strings", maxStops)
	}
	return list, nil
}

// responseMessage は応答の assistant の発言です。ツールを呼んで本文がなければ content は null です。
type responseMessage struct {
	Role             string         `json:"role"`
	Content          *string        `json:"content"`
	ReasoningContent string         `json:"reasoning_content,omitempty"`
	ToolCalls        []toolCallJSON `json:"tool_calls,omitempty"`
}

// toolCallJSON は応答のツールの呼び出し 1 件です。index はストリームのときだけ付けます。
type toolCallJSON struct {
	Index    *int             `json:"index,omitempty"`
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function toolCallFunction `json:"function"`
}

// toolCallFunction は呼び出す関数の名前と引数（JSON の文字列）です。
type toolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // 例: `{"city": "Paris"}`
}

// chatChoice は、まとめて返す応答の候補 1 個です（n は 1 だけ扱う）。
type chatChoice struct {
	Index        int             `json:"index"`
	Message      responseMessage `json:"message"`
	FinishReason string          `json:"finish_reason"`
}

// usage はトークン数です。cached_tokens は KV キャッシュを再利用したプロンプトのトークン数です。
type usage struct {
	PromptTokens        int `json:"prompt_tokens"`
	CompletionTokens    int `json:"completion_tokens"`
	TotalTokens         int `json:"total_tokens"`
	PromptTokensDetails struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
}

// timings は処理時間です（OpenAI にはない。llama.cpp の llama-server と同じ名前）。
type timings struct {
	QueueMS            float64 `json:"queue_ms"`             // GPU workerの待ち時間。
	CacheN             int     `json:"cache_n"`              // 再利用したプロンプトのトークン数
	PromptN            int     `json:"prompt_n"`             // 処理したプロンプトのトークン数
	PromptMS           float64 `json:"prompt_ms"`            // プロンプトの処理（最初のトークンを選ぶまで）
	PromptPerSecond    float64 `json:"prompt_per_second"`    // 例: 300
	PredictedN         int     `json:"predicted_n"`          // 生成したトークン数
	PredictedMS        float64 `json:"predicted_ms"`         // 2 個目以降の生成にかかった時間
	PredictedPerSecond float64 `json:"predicted_per_second"` // 2 個目以降の生成の速さ。例: 290
}

// chatResponse は、まとめて返す応答です。
type chatResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []chatChoice `json:"choices"`
	Usage   usage        `json:"usage"`
	Timings timings      `json:"timings"`
}

// chunkDelta は、ストリームの 1 回分の増分です。
type chunkDelta struct {
	Role             string         `json:"role,omitempty"`
	Content          *string        `json:"content,omitempty"`
	ReasoningContent *string        `json:"reasoning_content,omitempty"`
	ToolCalls        []toolCallJSON `json:"tool_calls,omitempty"`
}

// chunkChoice は、ストリームの 1 回分の候補です。
type chunkChoice struct {
	Index        int        `json:"index"`
	Delta        chunkDelta `json:"delta"`
	FinishReason *string    `json:"finish_reason"`
}

// chatChunk は、ストリームで送る 1 回分（chat.completion.chunk）です。
type chatChunk struct {
	ID      string        `json:"id"`
	Object  string        `json:"object"`
	Created int64         `json:"created"`
	Model   string        `json:"model"`
	Choices []chunkChoice `json:"choices"`
	Usage   *usage        `json:"usage,omitempty"`
	Timings *timings      `json:"timings,omitempty"`
}

// errorBody は誤りの応答です。例: {"error": {"message": "...", "type": "invalid_request_error"}}
type errorBody struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error"`
}

// modelList は GET /v1/models の応答です。
type modelList struct {
	Object string      `json:"object"`
	Data   []modelInfo `json:"data"`
}

// modelInfo はモデル 1 個の情報です。
type modelInfo struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}
