package server

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/cardy0828/pascallm/internal/chat"
	"github.com/cardy0828/pascallm/internal/generate"
	"github.com/cardy0828/pascallm/internal/grammar"
	"github.com/cardy0828/pascallm/internal/pyjson"
)

// この file は、tools・tool_choice・response_format を読み、出力の読み方と、選び方の制約を決める処理です（M6）。

// tool_choice の値です。
const (
	toolChoiceAuto     = "auto"     // ツールを呼ぶかはモデルが決める（tools があるときの既定）
	toolChoiceNone     = "none"     // ツールを呼ばない（<tool_call> を選べなくする）
	toolChoiceRequired = "required" // 1 件以上呼ぶ（文法で縛る）
	toolChoiceFunction = "function" // 指定の関数を 1 件だけ呼ぶ（文法で縛る）
)

// response_format の type の値です。
const (
	formatText       = "text"
	formatJSONObject = "json_object"
	formatJSONSchema = "json_schema"
)

// toolSettings は、要求のツールと出力の形の設定を読んだものです。
type toolSettings struct {
	tools        []pyjson.Value // 要求の tools（キーの順を保つ）
	choice       string         // toolChoice* のどれか
	functionName string         // choice が toolChoiceFunction のときの関数の名前
	parallel     bool           // parallel_tool_calls（既定 true）
	format       string         // format* のどれか
	schema       pyjson.Value   // format が formatJSONSchema のときの Schema
}

// parseToolSettings は tools・tool_choice・parallel_tool_calls・response_format を読み、矛盾がないか確かめます。
// 例: tools あり、tool_choice なし → choice "auto"
func parseToolSettings(request *chatRequest) (toolSettings, error) {
	settings := toolSettings{parallel: true, format: formatText}
	tools, err := parseTools(request.Tools)
	if err != nil {
		return toolSettings{}, err
	}
	settings.tools = tools
	if request.ParallelToolCalls != nil {
		settings.parallel = *request.ParallelToolCalls
	}
	if settings.choice, settings.functionName, err = parseToolChoice(request.ToolChoice, len(tools) > 0); err != nil {
		return toolSettings{}, err
	}
	if settings.choice == toolChoiceFunction {
		if _, ok := chat.NewToolSet(tools).Function(settings.functionName); !ok {
			return toolSettings{}, fmt.Errorf("tool_choice names function %q, which is not in tools", settings.functionName)
		}
	}
	if settings.format, settings.schema, err = parseResponseFormat(request.ResponseFormat); err != nil {
		return toolSettings{}, err
	}
	return settings, nil
}

// parseTools は tools を読みます。各要素は {"type": "function", "function": {"name": ...}} の形です。
// 例: null → nil、[] → nil
func parseTools(raw json.RawMessage) ([]pyjson.Value, error) {
	value, err := parseOptionalJSON(raw)
	if err != nil {
		return nil, fmt.Errorf("tools: %w", err)
	}
	switch value.Kind {
	case pyjson.Null:
		return nil, nil
	case pyjson.Array:
	default:
		return nil, errors.New("tools must be an array")
	}
	for index, tool := range value.Items {
		function, ok := tool.Get("function")
		if !ok || function.Kind != pyjson.Object {
			return nil, fmt.Errorf("tools[%d] must be {\"type\": \"function\", \"function\": {...}}", index)
		}
		if name, ok := function.GetString("name"); !ok || name == "" {
			return nil, fmt.Errorf("tools[%d].function.name must be a non-empty string", index)
		}
		if kind, ok := tool.GetString("type"); ok && kind != "function" {
			return nil, fmt.Errorf("tools[%d].type %q is not supported (function only)", index, kind)
		}
	}
	if len(value.Items) == 0 {
		return nil, nil
	}
	return value.Items, nil
}

// parseToolChoice は tool_choice を読み、選び方と（関数を指定したなら）関数の名前を返します。
// 指定がなければ、tools があれば "auto"、なければ "none" です。tools なしで required・関数の指定はエラーです。
// 例: `{"type": "function", "function": {"name": "f"}}` → ("function", "f")
func parseToolChoice(raw json.RawMessage, hasTools bool) (string, string, error) {
	value, err := parseOptionalJSON(raw)
	if err != nil {
		return "", "", fmt.Errorf("tool_choice: %w", err)
	}
	choice, name := toolChoiceNone, ""
	if hasTools {
		choice = toolChoiceAuto
	}
	switch value.Kind {
	case pyjson.Null:
	case pyjson.String:
		switch value.String {
		case toolChoiceAuto, toolChoiceNone, toolChoiceRequired:
			choice = value.String
		default:
			return "", "", fmt.Errorf("tool_choice %q is not supported", value.String)
		}
	case pyjson.Object:
		function, _ := value.Get("function")
		functionName, ok := function.GetString("name")
		if !ok || functionName == "" {
			return "", "", errors.New("tool_choice must be {\"type\": \"function\", \"function\": {\"name\": ...}}")
		}
		choice, name = toolChoiceFunction, functionName
	default:
		return "", "", errors.New("tool_choice must be a string or an object")
	}
	if !hasTools && (choice == toolChoiceRequired || choice == toolChoiceFunction) {
		return "", "", errors.New("tool_choice requires tools")
	}
	if !hasTools && choice == toolChoiceAuto {
		choice = toolChoiceNone
	}
	return choice, name, nil
}

// parseResponseFormat は response_format を読みます。json_schema なら Schema も返します。
// 例: {"type": "json_schema", "json_schema": {"name": "x", "schema": {...}}} → ("json_schema", {...})
func parseResponseFormat(raw json.RawMessage) (string, pyjson.Value, error) {
	value, err := parseOptionalJSON(raw)
	if err != nil {
		return "", pyjson.Value{}, fmt.Errorf("response_format: %w", err)
	}
	if value.Kind == pyjson.Null {
		return formatText, pyjson.Value{}, nil
	}
	kind, _ := value.GetString("type")
	switch kind {
	case formatText, formatJSONObject:
		return kind, pyjson.Value{}, nil
	case formatJSONSchema:
		wrapper, _ := value.Get("json_schema")
		schema, ok := wrapper.Get("schema")
		if !ok {
			return "", pyjson.Value{}, errors.New("response_format.json_schema.schema is required")
		}
		return kind, schema, nil
	default:
		return "", pyjson.Value{}, fmt.Errorf("response_format type %q is not supported", kind)
	}
}

// outputOptions は、出力の読み方の設定を作ります。tool_choice が none ならツールの呼び出しを読みません。
func (t toolSettings) outputOptions(thinking bool, stops []string) chat.OutputOptions {
	options := chat.OutputOptions{StartsInReasoning: thinking, Stops: stops}
	if len(t.tools) > 0 && t.choice != toolChoiceNone {
		options.Tools = chat.NewToolSet(t.tools)
		options.SingleToolCall = !t.parallel || t.choice == toolChoiceFunction
	}
	return options
}

// specialTokens は、制約を作るのに使うトークンの番号です。
type specialTokens struct {
	thinkEnd int
	tools    grammar.ToolTokens
}

// lookupSpecialTokens は、"</think>" とツールのタグのトークン番号を引きます。
func (s *Server) lookupSpecialTokens() (specialTokens, error) {
	names := []string{chat.ThinkEnd, chat.ToolCallStart, chat.ToolCallEnd, chat.FunctionStart, chat.FunctionEnd,
		chat.ParameterStart, chat.ParameterEnd}
	ids := make(map[string]int, len(names))
	for _, name := range names {
		id, ok := s.config.Tokenizer.TokenID(name)
		if !ok {
			return specialTokens{}, fmt.Errorf("tokenizer has no %s token", name)
		}
		ids[name] = id
	}
	return specialTokens{
		thinkEnd: ids[chat.ThinkEnd],
		tools: grammar.ToolTokens{
			CallStart: ids[chat.ToolCallStart], CallEnd: ids[chat.ToolCallEnd],
			FunctionStart: ids[chat.FunctionStart], FunctionEnd: ids[chat.FunctionEnd],
			ParameterStart: ids[chat.ParameterStart], ParameterEnd: ids[chat.ParameterEnd],
		},
	}, nil
}

// constraint は、選び方の制約を作ります。制約がなければ nil です。
//   - tool_choice が required・関数の指定: ツールの呼び出しの文法
//   - response_format が json_object・json_schema: JSON の文法（tool_choice が auto なら、呼び出しとのどちらか）
//   - tool_choice が none（tools あり）: <tool_call> を選べなくする
func (t toolSettings) constraint(s *Server, thinking bool) (generate.Constraint, error) {
	if s.config.Architecture == "gemma4" {
		return t.gemmaConstraint(s, thinking)
	}
	tokens, err := s.lookupSpecialTokens()
	if err != nil {
		return nil, err
	}
	var toolGrammar, formatGrammar *grammar.Grammar
	switch t.choice {
	case toolChoiceRequired:
		toolGrammar, err = grammar.ToolCalls(functionsOf(t.tools, ""), tokens.tools, t.parallel)
	case toolChoiceFunction:
		toolGrammar, err = grammar.ToolCalls(functionsOf(t.tools, t.functionName), tokens.tools, false)
	}
	if err != nil {
		return nil, fmt.Errorf("tools: %w", err)
	}
	switch t.format {
	case formatJSONObject:
		formatGrammar = grammar.JSONObject()
	case formatJSONSchema:
		if formatGrammar, err = grammar.JSONSchema(t.schema); err != nil {
			return nil, fmt.Errorf("response_format: %w", err)
		}
	}
	if formatGrammar != nil && t.choice == toolChoiceAuto {
		calls, err := grammar.ToolCalls(functionsOf(t.tools, ""), tokens.tools, t.parallel)
		if err != nil {
			return nil, fmt.Errorf("tools: %w", err)
		}
		formatGrammar = grammar.Either(formatGrammar, calls)
	}
	chosen := toolGrammar
	if chosen == nil {
		chosen = formatGrammar
	}
	if chosen != nil {
		return grammar.NewConstraint(chosen, s.config.Tokenizer, thinking, tokens.thinkEnd, s.config.StopTokens), nil
	}
	if len(t.tools) > 0 && t.choice == toolChoiceNone {
		return grammar.NewTokenBan(s.config.Tokenizer.Size(), tokens.tools.CallStart), nil
	}
	return nil, nil
}

// functionsOf は tools の関数の定義を返します。name が空でなければ、その名前の関数だけです。
func functionsOf(tools []pyjson.Value, name string) []pyjson.Value {
	set := chat.NewToolSet(tools)
	var functions []pyjson.Value
	for _, functionName := range set.Names() {
		if name == "" || functionName == name {
			function, _ := set.Function(functionName)
			functions = append(functions, function)
		}
	}
	return functions
}

// toolCallsJSON は、読み取った呼び出しを応答の形にします。first は最初の呼び出しの通し番号（ストリームの index）、
// withIndex が true なら index を付けます。例: [{get_weather {"city": "Paris"}}] → [{id: "call_…", ...}]
func toolCallsJSON(calls []chat.ParsedToolCall, first int, withIndex bool) []toolCallJSON {
	result := make([]toolCallJSON, 0, len(calls))
	for offset, call := range calls {
		item := toolCallJSON{
			ID:       newToolCallID(),
			Type:     "function",
			Function: toolCallFunction{Name: call.Name, Arguments: call.Arguments.Dumps()},
		}
		if withIndex {
			index := first + offset
			item.Index = &index
		}
		result = append(result, item)
	}
	return result
}

// gemmaConstraint は Gemma のツール DSL または JSON の制約を作ります。思考チャンネルは準備済みプロンプトに合わせます。
func (t toolSettings) gemmaConstraint(s *Server, thinking bool) (generate.Constraint, error) {
	tok := s.config.Tokenizer
	start, ok := tok.TokenID(chat.GemmaCallStart)
	if !ok {
		return nil, fmt.Errorf("missing Gemma tool token")
	}
	end, _ := tok.TokenID(chat.GemmaCallEnd)
	quote, _ := tok.TokenID(chat.GemmaQuote)
	thinkEnd, _ := tok.TokenID(chat.GemmaChannelEnd)
	var chosen, format *grammar.Grammar
	var err error
	switch t.choice {
	case toolChoiceRequired:
		chosen, err = grammar.GemmaToolCalls(functionsOf(t.tools, ""), start, end, quote, t.parallel)
	case toolChoiceFunction:
		chosen, err = grammar.GemmaToolCalls(functionsOf(t.tools, t.functionName), start, end, quote, false)
	}
	if err != nil {
		return nil, err
	}
	switch t.format {
	case formatJSONObject:
		format = grammar.JSONObject()
	case formatJSONSchema:
		format, err = grammar.JSONSchema(t.schema)
	}
	if err != nil {
		return nil, err
	}
	if chosen == nil && format != nil {
		chosen = format
		if t.choice == toolChoiceAuto {
			calls, err := grammar.GemmaToolCalls(functionsOf(t.tools, ""), start, end, quote, t.parallel)
			if err != nil {
				return nil, err
			}
			chosen = grammar.Either(format, calls)
		}
	}
	if chosen != nil {
		return grammar.NewConstraint(chosen, tok, thinking, thinkEnd, s.config.StopTokens), nil
	}
	if t.choice == toolChoiceNone {
		return grammar.NewTokenBan(tok.Size(), start), nil
	}
	return nil, nil
}
