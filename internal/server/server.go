// Package server は、OpenAI 互換の HTTP API（M3。ツールコールと構造化出力は M6）です。
//
//   - GET  /health               起動しているか
//   - GET  /v1/models            モデルの一覧（1 個）
//   - POST /v1/chat/completions  対話の生成（stream: true なら SSE で少しずつ返す）
//
// GPU は 1 つの OS スレッドからしか使えないため、生成は RunWorker を呼んだゴルーチンだけで行います。
// HTTP のハンドラーは仕事を渡して終わりを待ちます。仕事は 1 件ずつ処理し、ほかの要求は順番を待ちます。
package server

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	mathrand "math/rand/v2"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	"github.com/cardy0828/pascallm/internal/chat"
	"github.com/cardy0828/pascallm/internal/generate"
	"github.com/cardy0828/pascallm/internal/sampling"
	"github.com/cardy0828/pascallm/internal/tokenizer"
)

// maxRequestBytes は要求の本文の上限です（長い会話でも十分な大きさ）。
const maxRequestBytes = 8 << 20

// maxQueuedJobs は実行中の1件とは別の待ち行列上限です。
const maxQueuedJobs = 16

// jobの状態は待機から実行または取消へ一度だけ進みます。
const (
	jobWaiting int32 = iota
	jobRunning
	jobCancelled
)

var errQueueFull = errors.New("generation queue is full")

// finishToolCalls は、ツールを呼んで止まったときの finish_reason です（OpenAI と同じ）。
const finishToolCalls = "tool_calls"

// Config はサーバーの設定です。
type Config struct {
	Architecture string               // lfm3（空も同じ）または gemma4。
	ModelName    string               // /v1/models と応答の model に出す名前。例: "lfm3-1.7b-pq4"
	Tokenizer    *tokenizer.Tokenizer // トークナイザー
	Defaults     Defaults             // 要求で指定がないときの値
	MaxContext   int                  // プロンプトと生成を合わせた上限。例: 4096
	StopTokens   []int                // 生成を止めるトークン（generate.Engine と同じ）。例: <|im_end|>・<|endoftext|>
}

// Server は HTTP のハンドラーと、生成の仕事の受け渡し口を持ちます。
type Server struct {
	stopped chan struct{}
	config  Config
	jobs    chan *job
	created int64 // 起動した時刻（/v1/models の created）
}

// job は、作業ゴルーチンで行う生成 1 件です。
type job struct {
	state atomic.Int32 // jobWaiting / jobRunning / jobCancelled
	ctx   context.Context
	run   func(engine *generate.Engine)
	done  chan struct{}
}

// New は Server を作ります。例: New(Config{ModelName: "lfm3-1.7b-pq4", Tokenizer: tok, ...})
func New(config Config) *Server {
	return &Server{stopped: make(chan struct{}), config: config, jobs: make(chan *job, maxQueuedJobs), created: time.Now().Unix()}
}

// Handler は、API の経路を登録した http.Handler を返します。
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", s.handleHealth)
	mux.HandleFunc("GET /v1/models", s.handleModels)
	mux.HandleFunc("POST /v1/chat/completions", s.handleChat)
	return mux
}

// RunWorker は、ctx が取り消されるまで生成の仕事を 1 件ずつ実行します。CUDA のコンテキストを持つ
// （runtime.LockOSThread を呼んだ）ゴルーチンから呼び、そのゴルーチンを占有します。
// 例: go httpServer.Serve(listener); srv.RunWorker(ctx, engine)
func (s *Server) RunWorker(ctx context.Context, engine *generate.Engine) {
	defer close(s.stopped)
	for {
		select {
		case <-ctx.Done():
			return
		case next := <-s.jobs:
			if next.state.CompareAndSwap(jobWaiting, jobRunning) && next.ctx.Err() == nil {
				next.run(engine)
			}
			close(next.done)
		}
	}
}

// submit は仕事を作業ゴルーチンに渡し、終わるまで待ちます。順番を待つ間に ctx が取り消されたら、
// 待機中の仕事を取り消して ctx.Err() を返します。実行済みの場合は完了まで待ちます。
func (s *Server) submit(ctx context.Context, run func(engine *generate.Engine)) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	next := &job{ctx: ctx, run: run, done: make(chan struct{})}
	select {
	case <-s.stopped:
		return errors.New("generation worker stopped")
	case s.jobs <- next:
	case <-ctx.Done():
		return ctx.Err()
	default:
		return errQueueFull
	}
	select {
	case <-s.stopped:
		return errors.New("generation worker stopped")
	case <-next.done:
		return ctx.Err()
	case <-ctx.Done():
		if next.state.CompareAndSwap(jobWaiting, jobCancelled) {
			return ctx.Err()
		}
		// 実行開始済みなら handler が応答を書き終えるまで待ち、HTTPとの競合を避ける。
		<-next.done
		return ctx.Err()
	}
}

// handleHealth は {"status": "ok"} を返します。
func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

// handleModels はモデルの一覧（1 個）を返します。
func (s *Server) handleModels(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, modelList{
		Object: "list",
		Data:   []modelInfo{{ID: s.config.ModelName, Object: "model", Created: s.created, OwnedBy: "pascallm"}},
	})
}

// preparedChat は、要求を検査して生成に必要な形にしたものです。
type preparedChat struct {
	request      generate.Request
	output       chat.OutputOptions // 出力の読み方（思考・stop・ツールの呼び出し）
	includeUsage bool               // ストリームの最後に usage を送るか
}

// handleChat は POST /v1/chat/completions を処理します。
func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, maxRequestBytes)
	var request chatRequest
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}
	prepared, err := s.prepare(&request)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if request.Stream {
		s.streamChat(w, r, prepared)
		return
	}
	s.completeChat(w, r, prepared)
}

// prepare は要求を検査し、テンプレートを当てはめてトークン列にし、選び方の設定と制約を決めます。
func (s *Server) prepare(request *chatRequest) (preparedChat, error) {
	if request.N != nil && *request.N != 1 {
		return preparedChat{}, errors.New("only n = 1 is supported")
	}
	if len(request.Messages) == 0 {
		return preparedChat{}, errors.New("messages must not be empty")
	}
	settings, err := parseToolSettings(request)
	if err != nil {
		return preparedChat{}, err
	}
	messages := make([]chat.Message, 0, len(request.Messages))
	for _, raw := range request.Messages {
		message, err := raw.toChatMessage()
		if err != nil {
			return preparedChat{}, err
		}
		messages = append(messages, message)
	}
	thinking, err := s.enableThinking(request.ChatTemplateKwargs)
	if err != nil {
		return preparedChat{}, err
	}
	render := chat.Render
	if s.config.Architecture == "gemma4" {
		render = chat.RenderGemma
	}
	renderTools := settings.tools
	// none は呼び出しを禁止するだけでなく、Gemma のプロンプトから利用可能なツール宣言を外します。
	// 宣言を残すとモデルが直ちに tool_response を待って空の応答になる場合があります。
	if s.config.Architecture == "gemma4" && settings.choice == toolChoiceNone {
		renderTools = nil
	}
	segments, err := render(messages, chat.Options{EnableThinking: thinking, AddGenerationPrompt: true, Tools: renderTools})
	if err != nil {
		return preparedChat{}, err
	}
	output := settings.outputOptions(thinking, nil)
	if s.config.Architecture == "gemma4" {
		output.Gemma = true
		// 文法で本文を縛るときだけ、思考チャンネルをプロンプトで開きます。
		prefix := chat.GemmaChannelStart + "thought\n"
		var rendered strings.Builder
		for _, segment := range segments {
			rendered.WriteString(segment.Text)
		}
		starts := strings.HasSuffix(rendered.String(), prefix)
		constrained := settings.format != formatText || settings.choice == toolChoiceRequired || settings.choice == toolChoiceFunction
		if thinking && constrained && !starts {
			segments = append(segments, tokenizer.Segment{Text: prefix, Control: true})
			starts = true
		}
		output.StartsInReasoning = starts
	}
	prompt, err := s.config.Tokenizer.Encode(segments...)
	if err != nil {
		return preparedChat{}, err
	}
	if len(prompt) >= s.config.MaxContext {
		return preparedChat{}, fmt.Errorf("prompt has %d tokens; the context limit is %d", len(prompt), s.config.MaxContext)
	}
	maxTokens, err := s.maxTokens(request, len(prompt))
	if err != nil {
		return preparedChat{}, err
	}
	params := s.samplingParams(request)
	if err := params.Validate(); err != nil {
		return preparedChat{}, err
	}
	stops, err := parseStop(request.Stop)
	if err != nil {
		return preparedChat{}, err
	}
	output.Stops = stops
	constraint, err := settings.constraint(s, output.StartsInReasoning)
	if err != nil {
		return preparedChat{}, err
	}
	snapshotTokens := 0
	if s.config.Architecture == "gemma4" {
		channel, _ := s.config.Tokenizer.TokenID(chat.GemmaChannelStart)
		// 生成の入口の思考タグは次の履歴では取り除かれるため、その直前の安定した部分を保存します。
		for index := len(prompt) - 1; index >= 0; index-- {
			if prompt[index] == channel {
				snapshotTokens = index
				break
			}
		}
		// 過去の発言内のチャンネルを保存位置として選ばないよう、生成の入口にある場合だけ採用します。
		var rendered strings.Builder
		for _, segment := range segments {
			rendered.WriteString(segment.Text)
		}
		suffix := chat.GemmaChannelStart + "thought\n"
		if !strings.HasSuffix(rendered.String(), suffix) && !strings.HasSuffix(rendered.String(), suffix+chat.GemmaChannelEnd) {
			snapshotTokens = 0
		}
	}
	return preparedChat{
		request:      generate.Request{SnapshotTokens: snapshotTokens, Prompt: prompt, MaxTokens: maxTokens, Sampling: params, Constraint: constraint},
		output:       output,
		includeUsage: request.StreamOptions != nil && request.StreamOptions.IncludeUsage,
	}, nil
}

// enableThinking は chat_template_kwargs の enable_thinking を読みます（なければ既定値）。
// 例: {"enable_thinking": false} → false
func (s *Server) enableThinking(kwargs map[string]json.RawMessage) (bool, error) {
	raw, ok := kwargs["enable_thinking"]
	if !ok {
		return s.config.Defaults.EnableThinking, nil
	}
	var enabled bool
	if err := json.Unmarshal(raw, &enabled); err != nil {
		return false, errors.New("chat_template_kwargs.enable_thinking must be a boolean")
	}
	return enabled, nil
}

// maxTokens は生成する最大のトークン数を決めます。max_completion_tokens を優先し、どちらもなければ
// 文脈の残り全部です。例: プロンプト 100、上限 4096、指定なし → 3996
func (s *Server) maxTokens(request *chatRequest, promptTokens int) (int, error) {
	remaining := s.config.MaxContext - promptTokens
	limit := request.MaxCompletionTokens
	if limit == nil {
		limit = request.MaxTokens
	}
	if limit == nil {
		return remaining, nil
	}
	if *limit < 1 {
		return 0, errors.New("max_tokens must be at least 1")
	}
	return min(*limit, remaining), nil
}

// samplingParams は、要求の値と既定値から選び方の設定を作ります。seed がなければ乱数で決めます。
func (s *Server) samplingParams(request *chatRequest) sampling.Params {
	defaults := s.config.Defaults
	params := sampling.Params{
		Temperature:       valueOr(request.Temperature, defaults.Temperature),
		TopK:              valueOr(request.TopK, defaults.TopK),
		TopP:              valueOr(request.TopP, defaults.TopP),
		MinP:              valueOr(request.MinP, defaults.MinP),
		RepetitionPenalty: valueOr(request.RepeatPenalty, valueOr(request.RepetitionPenalty, defaults.RepetitionPenalty)),
		PresencePenalty:   valueOr(request.PresencePenalty, 0),
		FrequencyPenalty:  valueOr(request.FrequencyPenalty, 0),
		Seed:              mathrand.Uint64(),
	}
	if request.Seed != nil {
		params.Seed = uint64(*request.Seed)
	}
	return params
}

// valueOr は、value が nil でなければその値、nil なら fallback を返します。例: valueOr(nil, 0.1) → 0.1
func valueOr[T any](value *T, fallback T) T {
	if value == nil {
		return fallback
	}
	return *value
}

// finishReason は、生成の終わり方と、ツールを呼んだかから finish_reason を決めます。
// 呼び出しがあって stop で止まったなら "tool_calls" です。例: ("stop", 1) → "tool_calls"
func finishReason(result generate.Result, toolCalls int) string {
	if toolCalls > 0 && result.FinishReason == generate.FinishStop {
		return finishToolCalls
	}
	return result.FinishReason
}

// completeChat は、生成が終わってからまとめて応答を返します。
func (s *Server) completeChat(w http.ResponseWriter, r *http.Request, prepared preparedChat) {
	parser, err := chat.NewOutputParser(s.config.Tokenizer, prepared.output)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	var reasoning, content strings.Builder
	var calls []chat.ParsedToolCall
	add := func(delta chat.Delta) {
		reasoning.WriteString(delta.Reasoning)
		content.WriteString(delta.Content)
		calls = append(calls, delta.ToolCalls...)
	}
	handler := func(id int) (bool, error) {
		delta, err := parser.Push(id)
		add(delta)
		return parser.Stopped(), err
	}
	result, err := s.generate(r.Context(), prepared.request, handler)
	if err != nil {
		if r.Context().Err() == nil {
			status := http.StatusInternalServerError
			if errors.Is(err, errQueueFull) {
				status = http.StatusServiceUnavailable
			}
			writeError(w, status, err.Error())
		}
		return
	}
	add(parser.Finish())
	message := responseMessage{Role: chat.RoleAssistant, ReasoningContent: reasoning.String()}
	text := content.String()
	if len(calls) == 0 || strings.TrimSpace(text) != "" {
		message.Content = &text
	}
	if len(calls) > 0 {
		message.ToolCalls = toolCallsJSON(calls, 0, false)
	}
	writeJSON(w, http.StatusOK, chatResponse{
		ID:      newID(),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   s.config.ModelName,
		Choices: []chatChoice{{Message: message, FinishReason: finishReason(result, len(calls))}},
		Usage:   makeUsage(result),
		Timings: makeTimings(result),
	})
}

// streamChat は、生成しながら SSE（text/event-stream）で増分を送ります。最後に "data: [DONE]" を送ります。
// ツールの呼び出しは、1 件読み終えるごとに、名前と引数をまとめて 1 回で送ります。
func (s *Server) streamChat(w http.ResponseWriter, r *http.Request, prepared preparedChat) {
	parser, err := chat.NewOutputParser(s.config.Tokenizer, prepared.output)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming is not supported by this connection")
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)

	id, created := newID(), time.Now().Unix()
	send := func(chunk chatChunk) error {
		chunk.ID, chunk.Object, chunk.Created, chunk.Model = id, "chat.completion.chunk", created, s.config.ModelName
		return writeEvent(w, flusher, chunk)
	}
	sentCalls := 0
	sendDelta := func(delta chat.Delta) error {
		if delta.Empty() {
			return nil
		}
		fields := deltaFields(delta)
		if len(delta.ToolCalls) > 0 {
			fields.ToolCalls = toolCallsJSON(delta.ToolCalls, sentCalls, true)
			sentCalls += len(delta.ToolCalls)
		}
		return send(chatChunk{Choices: []chunkChoice{{Delta: fields}}})
	}
	empty := ""
	if err := send(chatChunk{Choices: []chunkChoice{{Delta: chunkDelta{Role: chat.RoleAssistant, Content: &empty}}}}); err != nil {
		return
	}
	handler := func(token int) (bool, error) {
		delta, err := parser.Push(token)
		if err != nil {
			return false, err
		}
		return parser.Stopped(), sendDelta(delta)
	}
	result, err := s.generate(r.Context(), prepared.request, handler)
	if err != nil {
		if r.Context().Err() == nil {
			var body errorBody
			body.Error.Message, body.Error.Type = err.Error(), "server_error"
			_ = writeEvent(w, flusher, body)
		}
		return
	}
	if err := sendDelta(parser.Finish()); err != nil {
		return
	}
	finish := finishReason(result, parser.ToolCallCount())
	finishTimings := makeTimings(result)
	if err := send(chatChunk{Choices: []chunkChoice{{FinishReason: &finish}}, Timings: &finishTimings}); err != nil {
		return
	}
	if prepared.includeUsage {
		resultUsage := makeUsage(result)
		if err := send(chatChunk{Choices: []chunkChoice{}, Usage: &resultUsage}); err != nil {
			return
		}
	}
	_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
}

// generate は、作業ゴルーチンで生成を 1 件行い、結果を記録に残します（本文は記録しない）。
func (s *Server) generate(ctx context.Context, request generate.Request, handler generate.TokenHandler) (generate.Result, error) {
	var result generate.Result
	var generateErr error
	queued := time.Now()
	var queueWait time.Duration
	if err := s.submit(ctx, func(engine *generate.Engine) {
		queueWait = time.Since(queued)
		result, generateErr = engine.Generate(ctx, request, handler)
	}); err != nil {
		return result, err
	}
	result.QueueDuration = queueWait
	t := makeTimings(result)
	log.Printf("chat: prompt=%d cached=%d completion=%d finish=%s prompt_ms=%.1f decode=%.1f tokens/s full_logits_steps=%d constrained=%t err=%v",
		result.PromptTokens, result.CachedTokens, result.CompletionTokens, result.FinishReason,
		t.PromptMS, t.PredictedPerSecond, result.FullLogitsSteps, request.Constraint != nil, generateErr)
	if result.PrefixMetrics != nil {
		if data, err := json.Marshal(result.PrefixMetrics); err == nil {
			log.Printf("prefix_cache=%s queue_ms=%.3f", data, milliseconds(queueWait))
		}
	}
	if result.Metrics != nil {
		if data, err := json.Marshal(result.Metrics); err == nil {
			log.Printf("generation_metrics=%s", data)
		}
	}
	return result, generateErr
}

// deltaFields は chat.Delta の文字列の部分をストリームの増分の形にします（空の項目は送らない）。
func deltaFields(delta chat.Delta) chunkDelta {
	var fields chunkDelta
	if delta.Reasoning != "" {
		fields.ReasoningContent = &delta.Reasoning
	}
	if delta.Content != "" {
		fields.Content = &delta.Content
	}
	return fields
}

// makeUsage は生成の結果からトークン数を作ります。
func makeUsage(result generate.Result) usage {
	var u usage
	u.PromptTokens = result.PromptTokens
	u.CompletionTokens = result.CompletionTokens
	u.TotalTokens = result.PromptTokens + result.CompletionTokens
	u.PromptTokensDetails.CachedTokens = result.CachedTokens
	return u
}

// makeTimings は生成の結果から処理時間を作ります。例: 255 ステップを 0.88 秒 → predicted_per_second 290
func makeTimings(result generate.Result) timings {
	promptN := result.PromptTokens - result.CachedTokens
	promptMS := milliseconds(result.PromptDuration)
	decodeMS := milliseconds(result.DecodeDuration)
	return timings{
		QueueMS:            milliseconds(result.QueueDuration),
		CacheN:             result.CachedTokens,
		PromptN:            promptN,
		PromptMS:           promptMS,
		PromptPerSecond:    perSecond(promptN, promptMS),
		PredictedN:         result.CompletionTokens,
		PredictedMS:        decodeMS,
		PredictedPerSecond: perSecond(result.DecodeSteps, decodeMS),
	}
}

// milliseconds は時間をミリ秒にします。例: 1500 * time.Microsecond → 1.5
func milliseconds(duration time.Duration) float64 {
	return float64(duration) / float64(time.Millisecond)
}

// perSecond は、count 個を ms ミリ秒で処理したときの 1 秒あたりの数を返します（0 ミリ秒なら 0）。
// 例: perSecond(300, 1000) → 300
func perSecond(count int, ms float64) float64 {
	if ms <= 0 {
		return 0
	}
	const millisecondsPerSecond = 1000
	return float64(count) * millisecondsPerSecond / ms
}

// idRandomBytes は応答の id に使う乱数のバイト数です。
const idRandomBytes = 12

// newID は応答の id を作ります。例: "chatcmpl-3f9a0c..."
func newID() string {
	return "chatcmpl-" + randomHex()
}

// newToolCallID はツールの呼び出しの id を作ります。例: "call_3f9a0c..."
func newToolCallID() string {
	return "call_" + randomHex()
}

// randomHex は idRandomBytes バイトの乱数を 16 進の文字列にします。
func randomHex() string {
	buffer := make([]byte, idRandomBytes)
	_, _ = rand.Read(buffer)
	return hex.EncodeToString(buffer)
}

// writeJSON は JSON の応答を書きます。
func writeJSON(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}

// writeError は誤りの応答を書きます。例: writeError(w, 400, "messages must not be empty")
func writeError(w http.ResponseWriter, status int, message string) {
	var body errorBody
	body.Error.Message = message
	body.Error.Type = "invalid_request_error"
	if status >= http.StatusInternalServerError {
		body.Error.Type = "server_error"
	}
	writeJSON(w, status, body)
}

// writeEvent は SSE の 1 件（"data: {...}\n\n"）を書いて送り出します。
func writeEvent(w http.ResponseWriter, flusher http.Flusher, body any) error {
	data, err := json.Marshal(body)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "data: %s\n\n", data); err != nil {
		return err
	}
	flusher.Flush()
	return nil
}
