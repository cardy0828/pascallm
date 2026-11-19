// Package generate は、GPU 実装（LFM3・Gemma 4）でプロンプトを処理し、トークンを 1 個ずつ生成します（M3）。
//
//   - プロンプトは、最後の 1 個以外を Model.Prefill でまとめて入れます（M4）。
//   - 前回の要求で KV キャッシュに入れたトークン列で新しいプロンプトが始まるなら、その分は入れ直しません。
//   - そうでなくても、前回のプロンプト（最後の 1 個を除く）で始まるなら、その位置で保存した状態
//     （Model.SaveState）に戻し、生成した部分だけを捨てて続けます。続きの会話では、モデルが生成した
//     トークン列と、返答の文字列を符号化し直したトークン列が違うことがあるためです（例: 46 個と 45 個）。
//   - どちらにも当たらなければ最初から入れ直します（Gemma の輪状 KV が保存位置を上書きした場合も再入力する）。
//   - ペナルティも temperature もなければ GPU の argmax を使い、logits を CPU に移しません。
//   - それ以外は、GPU でペナルティを適用して上位の候補（gpucore.CandidateCount 個）を絞り、候補だけを CPU に移して
//     選びます。候補だけで決まらない設定（top_k が候補の数より大きいなど）のときだけ、logits 全体を移します。
//   - 選んでよいトークンの制限（Request.Constraint、M6）があれば、候補を制限で確かめてから選びます。
//     最大のトークンが許されるとは限らないため、GPU の argmax だけの選び方は使いません。
//
// Engine は CUDA のコンテキストを持つ OS スレッドから使います（lfm3gpu の説明を参照）。同時には 1 件だけ処理します。
package generate

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"time"

	"github.com/cardy0828/pascallm/internal/gpucore"
	"github.com/cardy0828/pascallm/internal/sampling"
)

// 生成の終わり方です（OpenAI の finish_reason と同じ値）。
const (
	FinishStop   = "stop"   // 終わりのトークン、または stop の文字列で止まった
	FinishLength = "length" // max_tokens または文脈の上限で止まった
)

// Engine は、GPU のモデルと、KV キャッシュに入っているトークン列を持ちます。
type Engine struct {
	prefix      *prefixCache
	model       Model
	validTokens int          // 選んでよい番号の上限（含まない）。トークナイザーの番号の数。例: 125033
	stopTokens  map[int]bool // 生成を止めるトークン。例: <|im_end|>・<|endoftext|>
	cached      []int        // KV キャッシュに入っているトークン列（長さは model.Position() と同じ）
	// saved は、モデルが状態を保存した位置までのトークン列です（長さは model.SavedPosition() と同じ。なければ nil）。
	saved []int
	// gpuCandidates が true なら、GPU で候補を絞ります（既定）。false なら毎回 logits 全体を CPU に移します（比較用）。
	gpuCandidates bool
	// batchPrefill が true なら、プロンプトを Prefill でまとめて入れます（既定）。false なら 1 個ずつ Feed します（比較用）。
	batchPrefill bool
}

// New は Engine を作ります。例: New(model, tok.Size(), []int{124900, 124895})
func New(model Model, validTokens int, stopTokens []int) (*Engine, error) {
	if validTokens <= 0 || validTokens > model.VocabSize() {
		return nil, fmt.Errorf("generate: valid token count %d is outside the model vocabulary %d", validTokens, model.VocabSize())
	}
	stops := make(map[int]bool, len(stopTokens))
	for _, id := range stopTokens {
		stops[id] = true
	}
	return &Engine{model: model, validTokens: validTokens, stopTokens: stops, gpuCandidates: true, batchPrefill: true}, nil
}

// SetGPUCandidates は、GPU で候補を絞るか（true、既定）、毎回 logits 全体を CPU に移すか（false）を選びます。
// 結果の分布は同じで、速さだけが変わります（比較用）。
func (e *Engine) SetGPUCandidates(enabled bool) { e.gpuCandidates = enabled }

// SetBatchPrefill は、プロンプトを Prefill でまとめて入れるか（true、既定）、1 個ずつ Feed するか（false）を選びます。
// 例: engine.SetBatchPrefill(false)
func (e *Engine) SetBatchPrefill(enabled bool) { e.batchPrefill = enabled }

// MaxContext は、プロンプトと生成を合わせた最大のトークン数を返します。例: 4096
func (e *Engine) MaxContext() int { return e.model.MaxContext() }

// Request は 1 回の生成の指定です。
type Request struct {
	SnapshotTokens int             // 0 は最後の入力直前。Gemma の生成用思考タグより前を保存する場合は正の位置。
	Prompt         []int           // プロンプトのトークン列（BOS を含む）
	MaxTokens      int             // 生成する最大のトークン数（終わりのトークンは数えない）。例: 512
	Sampling       sampling.Params // 選び方
	// Constraint は、選んでよいトークンの制限です（nil なら制限なし）。制約付き生成（M6）で使います。
	Constraint Constraint
}

// Constraint は、生成の途中で選んでよいトークンを決めるものです（grammar.Constraint など）。
type Constraint interface {
	// Allows は、今の状態で id を選んでよいかを返します（状態は変えない）。
	Allows(id int) bool
	// Accept は、選んだトークンで状態を進めます。
	Accept(id int) error
}

// Result は 1 回の生成の結果です。
type Result struct {
	QueueDuration time.Duration // API workerの待ち時間。
	// PrefixMetrics と Metrics は任意の診断情報です。API 応答には含めずサーバーログへ記録します。
	PrefixMetrics    *PrefixMetrics
	Metrics          any
	PromptTokens     int           // プロンプトのトークン数
	CachedTokens     int           // そのうち、前回の KV キャッシュを使って入れ直さなかった数
	CompletionTokens int           // 生成したトークン数（終わりのトークンは数えない）
	FinishReason     string        // FinishStop または FinishLength
	PromptDuration   time.Duration // プロンプトの処理（最初のトークンを選ぶまで）
	DecodeDuration   time.Duration // 2 個目以降のトークンの生成
	// DecodeSteps は DecodeDuration の間に処理したトークン数です（最後に選んだ終わりのトークンの分を含む）。
	// 生成の速さは DecodeSteps / DecodeDuration です。
	DecodeSteps int
	// FullLogitsSteps は、logits 全体を CPU に移して選んだ回数です（GPU の候補で足りなかった回数を含む）。
	FullLogitsSteps int
}

// TokenHandler は、生成したトークン（終わりのトークンを除く）を 1 個ずつ受け取ります。
// stop に true を返すと、そこで生成を止めます（stop の文字列が見つかったときなど）。
type TokenHandler func(id int) (stop bool, err error)

// Generate はプロンプトを処理し、終わりのトークン・MaxTokens・文脈の上限・handler の指示のどれかまで生成します。
// ctx が取り消されたら、そこまでの結果と ctx.Err() を返します（KV キャッシュの内容は正しいまま残す）。
// GPU の処理が失敗した場合は、KV キャッシュを使えないものとして捨てます。
// 例: result, err := engine.Generate(ctx, Request{Prompt: ids, MaxTokens: 256, Sampling: params}, handler)
func (e *Engine) Generate(ctx context.Context, request Request, handler TokenHandler) (result Result, err error) {
	// 戻り値に名前を付けているのは、defer で DecodeDuration を結果に書くためです。
	result = Result{PromptTokens: len(request.Prompt)}
	if reporter, ok := e.model.(interface {
		ResetGenerationMetrics()
		GenerationMetrics() any
	}); ok {
		reporter.ResetGenerationMetrics()
		defer func() { result.Metrics = reporter.GenerationMetrics() }()
	}
	if len(request.Prompt) == 0 {
		return result, errors.New("generate: empty prompt")
	}
	if len(request.Prompt) >= e.MaxContext() {
		return result, fmt.Errorf("generate: prompt has %d tokens; the context limit is %d", len(request.Prompt), e.MaxContext())
	}
	if err := request.Sampling.Validate(); err != nil {
		return result, err
	}
	maxTokens := min(request.MaxTokens, e.MaxContext()-len(request.Prompt))

	if err := ctx.Err(); err != nil {
		return result, err
	}
	promptStart := time.Now()
	if e.prefix != nil {
		e.prefix.begin()
		defer func() { metrics := e.prefix.report(); result.PrefixMetrics = &metrics }()
	}
	start, err := e.reuseCache(request.Prompt)
	if err != nil {
		return result, e.fail(err)
	}
	result.CachedTokens = start
	chooser, err := e.newChooser(request)
	if err != nil {
		return result, e.fail(err)
	}
	defer func() { result.FullLogitsSteps = chooser.fullLogitsSteps }()

	last := len(request.Prompt) - 1
	snapshot := last
	if request.SnapshotTokens > 0 && request.SnapshotTokens <= last && request.SnapshotTokens >= start {
		snapshot = request.SnapshotTokens
	}
	if e.prefix != nil && start < sharedPrefixTokens && snapshot > sharedPrefixTokens {
		if err := e.feedPrompt(ctx, request.Prompt[start:sharedPrefixTokens]); err != nil {
			return result, err
		}
		if err := e.prefix.save(e.cached); err != nil {
			return result, e.fail(err)
		}
		start = sharedPrefixTokens
	}
	if err := e.feedPrompt(ctx, request.Prompt[start:snapshot]); err != nil {
		return result, err
	}
	if err := e.saveState(); err != nil {
		return result, e.fail(err)
	}
	if err := e.feedPrompt(ctx, request.Prompt[snapshot:last]); err != nil {
		return result, err
	}

	if chooser.mode == chooseGreedyGPU {
		if lookup, ok := e.model.(interface{ SetLookupHistory([]int) }); ok {
			lookup.SetLookupHistory(request.Prompt[:last])
			defer lookup.SetLookupHistory(nil)
		}
	}
	next, err := e.step(request.Prompt[last], chooser, false)
	if err != nil {
		return result, e.fail(err)
	}
	result.PromptDuration = time.Since(promptStart)

	decodeStart := time.Now()
	defer func() { result.DecodeDuration = time.Since(decodeStart) }()
	for {
		if request.Constraint != nil {
			if err := request.Constraint.Accept(next); err != nil {
				return result, err
			}
		}
		if e.stopTokens[next] {
			result.FinishReason = FinishStop
			return result, nil
		}
		result.CompletionTokens++
		chooser.sampler.Accept(next)
		stop, err := handler(next)
		if err != nil {
			return result, err
		}
		if stop {
			result.FinishReason = FinishStop
			return result, nil
		}
		if result.CompletionTokens >= maxTokens {
			result.FinishReason = FinishLength
			return result, nil
		}
		if err := ctx.Err(); err != nil {
			return result, err
		}
		if next, err = e.step(next, chooser, true); err != nil {
			return result, e.fail(err)
		}
		result.DecodeSteps++
	}
}

// feedPrompt は、プロンプトの最後の 1 個を除いたトークン列 tokens を入れます。batchPrefill なら
// モデルの区切りごとに Prefill し、そうでなければ 1 個ずつ Feed します。区切りごとに ctx を確かめ、
// 取り消されたらそこまでで止めます（入れた分は e.cached に記録済みで、KV キャッシュは正しいまま）。
// GPU の処理が失敗した場合は、KV キャッシュを捨てたうえでエラーを返します。
func (e *Engine) feedPrompt(ctx context.Context, tokens []int) error {
	pending := tokens
	for len(pending) > 0 {
		if err := ctx.Err(); err != nil {
			return err
		}
		count := 1
		if e.batchPrefill {
			count = min(len(pending), e.model.PrefillChunkSize())
		}
		part := pending[:count]
		if e.batchPrefill {
			if err := e.model.Prefill(part); err != nil {
				return e.fail(err)
			}
		} else if _, err := e.model.Feed(part[0]); err != nil {
			return e.fail(err)
		}
		e.cached = append(e.cached, part...)
		pending = pending[count:]
	}
	return nil
}

// reuseCache は、KV キャッシュのうち prompt の処理に使える長さを返し、モデルをその位置にします。
//   - KV キャッシュに入っているトークン列（e.cached）が prompt の先頭と一致すれば、その長さ（そのまま続ける）。
//   - そうでなく、状態を保存した位置までのトークン列（e.saved）が prompt の先頭と一致すれば、保存した位置に戻して
//     その長さ（その後に入れた分は捨てる）。
//   - どちらでもなければ、モデルを初めの状態に戻して 0。
//
// prompt と同じ長さ以上の一致は使いません（最後のトークンを入れて logits を得る必要があるため）。
// 例: cached = [1, 2, 3]、prompt = [1, 2, 3, 4, 5] → 3
// 例: cached = [1, 2, 3, 7, 8]、saved = [1, 2, 3]、prompt = [1, 2, 3, 7, 9, 4] → 3（位置 3 に戻す）
// 例: cached = [1, 2, 3]、saved = [1, 2]、prompt = [1, 9, 4] → 0
func (e *Engine) reuseCache(prompt []int) (int, error) {
	best := -1
	if e.prefix != nil {
		best = e.prefix.best(prompt)
	}
	bestLength := 0
	if best >= 0 {
		bestLength = len(e.prefix.entries[best].tokens)
	}
	if isUsablePrefix(e.cached, prompt) && len(e.cached) >= bestLength {
		return len(e.cached), nil
	}
	if e.saved != nil && e.model.CanRestore() && isUsablePrefix(e.saved, prompt) && len(e.saved) >= bestLength {
		if err := e.model.RestoreState(); err != nil {
			return 0, err
		}
		e.cached = append(e.cached[:0], e.saved...)
		return len(e.cached), nil
	}
	if best >= 0 {
		tokens, err := e.prefix.restore(best)
		if err != nil {
			return 0, err
		}
		e.cached = tokens
		e.saved = nil
		return len(tokens), nil
	}
	e.cached = e.cached[:0]
	e.saved = nil
	return 0, e.model.Reset()
}

// isUsablePrefix は、prefix が空でなく、prompt より短く、prompt の先頭と一致するかを返します。
// 例: isUsablePrefix([1 2], [1 2 3]) → true、isUsablePrefix([1 2 3], [1 2 3]) → false
func isUsablePrefix(prefix, prompt []int) bool {
	return len(prefix) > 0 && len(prefix) < len(prompt) && slices.Equal(prompt[:len(prefix)], prefix)
}

// saveState は、プロンプトの最後の 1 個を除いて入れ終えた位置で、モデルの状態を保存し、その位置までの
// トークン列を e.saved に記録します。次の要求で reuseCache が使います。
// 位置が 0（プロンプトが 1 個だけ）なら保存しません。
func (e *Engine) saveState() error {
	if len(e.cached) == 0 {
		e.saved = nil
		return nil
	}
	if err := e.model.SaveState(); err != nil {
		e.saved = nil
		return err
	}
	e.saved = append(e.saved[:0], e.cached...)
	if e.prefix != nil {
		return e.prefix.save(e.cached)
	}
	return nil
}

// chooseMode は、次のトークンの選び方です。
type chooseMode int

const (
	chooseGreedyGPU  chooseMode = iota // GPU の argmax だけ（temperature 0、ペナルティなし）
	chooseCandidates                   // GPU で候補を絞り、CPU で選ぶ
	chooseFullLogits                   // logits 全体を CPU に移して選ぶ（比較用）
)

// chooser は、1 回の生成での選び方と、選び手の状態です。
type chooser struct {
	mode            chooseMode
	sampler         *sampling.Sampler
	params          gpucore.CandidateParams // chooseCandidates で GPU に渡す設定
	fullLogitsSteps int                     // logits 全体を移した回数
	constraint      Constraint              // 選んでよいトークンの制限（nil なら制限なし）
}

// newChooser は、要求の選び方から chooser を作ります。GPU で候補を絞る場合は、ペナルティの対象（プロンプト）を
// GPU に送ります。文法の制限があれば候補を絞り、静的な禁止集合だけなら argmax を先に確認します。
// 例: temperature 0.1・repetition_penalty 1.1 → chooseCandidates
func (e *Engine) newChooser(request Request) (*chooser, error) {
	c := &chooser{sampler: sampling.New(request.Sampling, request.Prompt, e.validTokens), constraint: request.Constraint}
	switch {
	case !request.Sampling.NeedsLogits() && allowsGreedyFastPath(request.Constraint):
		c.mode = chooseGreedyGPU
	case e.gpuCandidates:
		c.mode = chooseCandidates
		c.params = candidateParams(request.Sampling, e.validTokens)
		if err := e.model.ResetPenalties(request.Prompt); err != nil {
			return nil, err
		}
	default:
		c.mode = chooseFullLogits
	}
	if selector, ok := e.model.(interface{ SetMTPSelectionDriven(bool) error }); ok {
		if err := selector.SetMTPSelectionDriven(c.mode != chooseGreedyGPU); err != nil {
			return nil, err
		}
	}
	return c, nil
}

// candidateParams は、選び方の設定を GPU の候補の処理の設定にします。temperature 0 なら逆数は 1 にします
// （順位だけを使うため）。例: temperature 0.1 → InverseTemperature 10
func candidateParams(params sampling.Params, validTokens int) gpucore.CandidateParams {
	inverseTemperature := float32(1)
	if params.Temperature > 0 {
		inverseTemperature = float32(1 / params.Temperature)
	}
	return gpucore.CandidateParams{
		ValidTokens:        validTokens,
		InverseTemperature: inverseTemperature,
		RepetitionPenalty:  float32(params.RepetitionPenalty),
		PresencePenalty:    float32(params.PresencePenalty),
		FrequencyPenalty:   float32(params.FrequencyPenalty),
	}
}

// step は token を 1 個処理して KV キャッシュに加え、次のトークンを選びます。
// generated は token が生成したトークンか（プロンプトの最後のトークンなら false）で、GPU のペナルティの記録に使います。
//   - chooseGreedyGPU: GPU の argmax。語彙外または禁止された候補なら logits を移して選び直す
//   - chooseCandidates: GPU で絞った候補から選ぶ。候補で決まらなければ logits 全体を移して選ぶ
//   - chooseFullLogits: logits 全体を移して選ぶ
func (e *Engine) step(token int, c *chooser, generated bool) (int, error) {
	switch c.mode {
	case chooseCandidates:
		candidates, err := e.model.FeedCandidates(token, generated, c.params)
		if err != nil {
			return 0, err
		}
		e.cached = append(e.cached, token)
		next, ok := c.chooseFromCandidates(sampling.Candidates{
			IDs: candidates.IDs, Logits: candidates.Logits, MaxLogit: candidates.MaxLogit, ExpSum: candidates.ExpSum,
		})
		if ok {
			return next, nil
		}
		return e.chooseFromFullLogits(c)
	case chooseFullLogits:
		logits, err := e.model.Forward(token)
		if err != nil {
			return 0, err
		}
		e.cached = append(e.cached, token)
		return c.chooseFromLogits(logits)
	default:
		next, err := e.model.Feed(token)
		if err != nil {
			return 0, err
		}
		e.cached = append(e.cached, token)
		if next >= 0 && next < e.validTokens && (c.constraint == nil || c.constraint.Allows(next)) {
			return next, nil
		}
		return e.chooseFromFullLogits(c)
	}
}

// chooseFromFullLogits は、直前に処理した位置の logits 全体を CPU に移して選びます（ペナルティは CPU で適用する）。
func (e *Engine) chooseFromFullLogits(c *chooser) (int, error) {
	logits, err := e.model.CopyLogits()
	if err != nil {
		return 0, err
	}
	return c.chooseFromLogits(logits)
}

// chooseFromCandidates は、GPU で絞った候補から選びます（制限があれば、許される候補だけから）。
// 候補だけで決まらなければ ok = false です。
func (c *chooser) chooseFromCandidates(candidates sampling.Candidates) (int, bool) {
	if c.constraint != nil {
		return c.sampler.ChooseFromCandidatesMasked(candidates, c.constraint.Allows)
	}
	return c.sampler.ChooseFromCandidates(candidates)
}

// chooseFromLogits は、logits 全体から選びます（制限があれば、許されるトークンだけから）。
// 許されるトークンが 1 個もなければエラーです。
func (c *chooser) chooseFromLogits(logits []float32) (int, error) {
	c.fullLogitsSteps++
	if c.constraint == nil {
		return c.sampler.Choose(logits), nil
	}
	next, ok := c.sampler.ChooseMasked(logits, c.constraint.Allows)
	if !ok {
		return 0, errors.New("generate: no token is allowed by the constraint")
	}
	return next, nil
}

// fail は、GPU の処理が失敗したときに KV キャッシュを捨て、元のエラーを返します。
// 次の要求では Reset から始めます（Reset が失敗してもここでは報告しない）。
func (e *Engine) fail(err error) error {
	e.cached = nil
	e.saved = nil
	return err
}

// allowsGreedyFastPath は静的な禁止集合だけの制約なら argmax の先行確認を許可します。
// JSON/tool 文法はこのインターフェースを実装しないため、従来の候補選択を使います。
func allowsGreedyFastPath(constraint Constraint) bool {
	if constraint == nil {
		return true
	}
	marker, ok := constraint.(interface{ AllowsGreedyFastPath() bool })
	return ok && marker.AllowsGreedyFastPath()
}
