package server

import (
	"encoding/json"
	"fmt"

	"github.com/cardy0828/pascallm/internal/chat"
	"github.com/cardy0828/pascallm/internal/tokenizer"
)

// RenderChat は、OpenAI の API の形の messages と tools から、API と同じ変換でテンプレートを当てはめた区間の並びを
// 作ります（生成の入口を付ける）。Hugging Face との突き合わせ（cmd/tokenizer-check）で使います。
// 例: RenderChat([]byte(`[{"role": "user", "content": "こんにちは"}]`), nil, true)
func RenderChat(messagesJSON, toolsJSON json.RawMessage, thinking bool) ([]tokenizer.Segment, error) {
	return RenderChatForModel("lfm3", messagesJSON, toolsJSON, thinking)
}

// RenderChatForModel は指定モデルのテンプレートを使います。検証ツールと API で同じ発言変換を使います。
func RenderChatForModel(architecture string, messagesJSON, toolsJSON json.RawMessage, thinking bool) ([]tokenizer.Segment, error) {
	var raw []chatMessage
	if err := json.Unmarshal(messagesJSON, &raw); err != nil {
		return nil, fmt.Errorf("messages: %w", err)
	}
	tools, err := parseTools(toolsJSON)
	if err != nil {
		return nil, err
	}
	messages := make([]chat.Message, 0, len(raw))
	for _, item := range raw {
		message, err := item.toChatMessage()
		if err != nil {
			return nil, err
		}
		messages = append(messages, message)
	}
	if architecture == "gemma4" {
		return chat.RenderGemma(messages, chat.Options{EnableThinking: thinking, AddGenerationPrompt: true, Tools: tools})
	}
	return chat.Render(messages, chat.Options{EnableThinking: thinking, AddGenerationPrompt: true, Tools: tools})
}
