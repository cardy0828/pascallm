package chat

import (
	"errors"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/cardy0828/pascallm/internal/pyjson"
	"github.com/cardy0828/pascallm/internal/tokenizer"
)

// Delta は、生成したトークンから得た文字列の増分です。思考（reasoning_content）と本文（content）に分け、
// 読み終えたツールの呼び出しも返します。
type Delta struct {
	Reasoning string           // 例: "まず計算します。"
	Content   string           // 例: "答えは 240 円です。"
	ToolCalls []ParsedToolCall // この増分で読み終えた呼び出し
}

// Empty は、増分が空かを返します。例: Delta{}.Empty() → true
func (d Delta) Empty() bool { return d.Reasoning == "" && d.Content == "" && len(d.ToolCalls) == 0 }

// ParsedToolCall は、出力から読み取ったツールの呼び出し 1 件です。
type ParsedToolCall struct {
	Name      string       // 例: "get_weather"
	Arguments pyjson.Value // 引数のオブジェクト（出力の順）。例: {"city": "Paris"}
}

// OutputOptions は出力の読み方の設定です。
type OutputOptions struct {
	Gemma bool // Gemma 4 のチャンネルとツール DSL を使う。
	// StartsInReasoning は、プロンプトが "<think>" で終わるか（Options.EnableThinking）です。
	StartsInReasoning bool
	// Stops は本文で止める文字列です。
	Stops []string
	// Tools は要求の tools です。nil なら <tool_call> を読まず、文字列のまま本文にします。
	Tools *ToolSet
	// SingleToolCall が true なら、最初の呼び出しを読み終えた時点で止めます（parallel_tool_calls: false）。
	SingleToolCall bool
}

// OutputParser は、生成したトークンを順に受け取り、思考・本文・ツールの呼び出しに分けます。
//   - "<think>"〜"</think>" の間を思考、それ以外を本文とします（思考ありのプロンプトは "<think>" で終わるため、
//     最初から思考の中として始めます）。
//   - 本文の先頭の空白（"</think>" の後の改行など）は取り除きます。ツールの呼び出しの直後の空白も同じです。
//   - 思考の外に出た "</think>" は捨てます。
//   - 本文の "<tool_call>"〜"</tool_call>" はツールの呼び出しとして読みます（Tools があるとき）。
//     読めなければ、元の文字列を本文に戻します。
//   - 特殊トークン（"<|im_end|>" など）は文字列にしません。止める判断は呼び出し側で行います。
//   - stop の文字列が本文に現れたら、その手前で止めます。stop の先頭と一致しうる末尾は、次のトークンまで返しません。
type OutputParser struct {
	tokenizer       *tokenizer.Tokenizer
	decoder         *tokenizer.StreamDecoder
	thinkStartID    int
	thinkEndID      int
	toolCallStartID int
	toolCallEndID   int
	options         OutputOptions
	inReasoning     bool
	contentBegun    bool            // 本文の空白以外の文字をすでに返したか
	trimLeading     bool            // 次の本文の先頭の空白を取り除くか
	stops           []string        // 本文で止める文字列
	held            string          // stop と照合するために返さずに持っている本文
	inToolCall      bool            // "<tool_call>" の後で "</tool_call>" を待っているか
	toolText        strings.Builder // 読んでいる途中のツールの呼び出しの文字列
	toolCalls       int             // 読み終えた呼び出しの数
	channelHeader   bool            // Gemma の <|channel> の後の thought 改行を待つ。
	headerText      string
	stopped         bool
}

// NewOutputParser は OutputParser を作ります。例: NewOutputParser(tok, OutputOptions{StartsInReasoning: true})
func NewOutputParser(tok *tokenizer.Tokenizer, options OutputOptions) (*OutputParser, error) {
	names := []string{ThinkStart, ThinkEnd, ToolCallStart, ToolCallEnd}
	if options.Gemma {
		names = []string{GemmaChannelStart, GemmaChannelEnd, GemmaCallStart, GemmaCallEnd}
	}
	ids := make([]int, len(names))
	for i, text := range names {
		id, ok := tok.TokenID(text)
		if !ok {
			return nil, errors.New("chat: tokenizer has no " + text + " token")
		}
		ids[i] = id
	}
	var nonEmptyStops []string
	for _, stop := range options.Stops {
		if stop != "" {
			nonEmptyStops = append(nonEmptyStops, stop)
		}
	}
	return &OutputParser{
		tokenizer:       tok,
		decoder:         tok.NewStreamDecoder(),
		thinkStartID:    ids[0],
		thinkEndID:      ids[1],
		toolCallStartID: ids[2],
		toolCallEndID:   ids[3],
		options:         options,
		inReasoning:     options.StartsInReasoning,
		trimLeading:     true,
		stops:           nonEmptyStops,
	}, nil
}

// Stopped は、stop の文字列で止まったか、1 件だけの呼び出しを読み終えたかを返します。
// true になった後の Push は何も返しません。
func (p *OutputParser) Stopped() bool { return p.stopped }

// ToolCallCount は、読み終えたツールの呼び出しの数を返します。
func (p *OutputParser) ToolCallCount() int { return p.toolCalls }

// Push は生成したトークンを 1 個受け取り、返せるようになった増分を返します。
// 例: 思考中に "</think>" → 思考の残りを返し、以後は本文として扱う
func (p *OutputParser) Push(id int) (Delta, error) {
	if p.stopped {
		return Delta{}, nil
	}
	switch {
	case p.inToolCall && id == p.toolCallEndID:
		return p.finishToolCall(), nil
	case p.inToolCall:
		if p.tokenizer.IsSpecial(id) && !(p.options.Gemma && string(p.tokenizer.TokenBytes(id)) == GemmaQuote) {
			return Delta{}, nil
		}
		text, err := p.decoder.Push(id)
		p.toolText.WriteString(text)
		return Delta{}, err
	case id == p.thinkEndID && p.inReasoning:
		delta := Delta{Reasoning: p.decoder.Flush()}
		p.inReasoning = false
		p.channelHeader = false
		p.headerText = ""
		return delta, nil
	case id == p.thinkStartID && !p.inReasoning && !p.contentBegun:
		p.inReasoning = true
		p.channelHeader = p.options.Gemma
		return Delta{}, nil
	case id == p.thinkEndID && !p.inReasoning:
		// 思考の外の "</think>" は、文字列にせず捨てます。思考なしでも、ツールの結果を受けた後に
		// 「答え</think>答え」のように出すことがある（M6 の確認で見つかった、モデルの癖）。
		return Delta{}, nil
	case id == p.toolCallStartID && !p.inReasoning && p.options.Tools != nil:
		// 持っていた本文（stop の照合待ちを含む）を出してから、呼び出しを読み始めます。
		delta := Delta{Content: p.addContent(p.decoder.Flush(), true)}
		p.inToolCall = true
		p.toolText.Reset()
		return delta, nil
	case p.tokenizer.IsSpecial(id):
		return Delta{}, nil
	}
	text, err := p.decoder.Push(id)
	if err != nil {
		return Delta{}, err
	}
	if p.inReasoning {
		if p.channelHeader {
			p.headerText += text
			_, after, ok := strings.Cut(p.headerText, "\n")
			if !ok {
				return Delta{}, nil
			}
			text = after
			p.channelHeader = false
			p.headerText = ""
		}
		return Delta{Reasoning: text}, nil
	}
	return Delta{Content: p.addContent(text, false)}, nil
}

// finishToolCall は "</tool_call>" を受け取ったときに、読んだ文字列を呼び出しにします。
// 読めなければ、"<tool_call>"〜"</tool_call>" の文字列をそのまま本文にします。
func (p *OutputParser) finishToolCall() Delta {
	p.toolText.WriteString(p.decoder.Flush())
	p.inToolCall = false
	call, ok := ParseToolCall(p.toolText.String(), p.options.Tools)
	if p.options.Gemma {
		call, ok = ParseGemmaCall(p.toolText.String(), p.options.Tools)
	}
	if !ok {
		return Delta{Content: p.addContent(p.callStart()+p.toolText.String()+p.callEnd(), false)}
	}
	p.toolCalls++
	p.trimLeading = true
	if p.options.SingleToolCall {
		p.stopped = true
	}
	return Delta{ToolCalls: []ParsedToolCall{call}}
}

// Finish は、持ち越していた文字列をすべて返します。生成が終わったときに 1 回呼びます。
// 読んでいる途中のツールの呼び出し（max_tokens で切れた場合など）は、元の文字列を本文にします。
func (p *OutputParser) Finish() Delta {
	if p.stopped {
		return Delta{}
	}
	rest := p.decoder.Flush()
	switch {
	case p.inToolCall:
		p.inToolCall = false
		return Delta{Content: p.addContent(p.callStart()+p.toolText.String()+rest, true)}
	case p.inReasoning:
		return Delta{Reasoning: rest}
	default:
		return Delta{Content: p.addContent(rest, true)}
	}
}

// addContent は本文の増分を受け取り、先頭の空白を取り除き、stop と照合して、返せる部分を返します。
// final が true なら、stop の先頭と一致しうる末尾も残さず返します。
// 例: stops = ["END"]、"abcE" → "abc"（"E" は持つ）、続いて "ND!" → ""（止まる）
func (p *OutputParser) addContent(text string, final bool) string {
	if p.trimLeading {
		text = strings.TrimLeftFunc(text, unicode.IsSpace)
		if text == "" {
			return ""
		}
		p.trimLeading = false
		p.contentBegun = true
	}
	p.held += text
	for _, stop := range p.stops {
		if index := strings.Index(p.held, stop); index >= 0 {
			out := p.held[:index]
			p.held = ""
			p.stopped = true
			return out
		}
	}
	keep := 0
	if !final {
		keep = longestStopPrefixSuffix(p.held, p.stops)
	}
	out := p.held[:len(p.held)-keep]
	p.held = p.held[len(p.held)-keep:]
	return out
}

// longestStopPrefixSuffix は、text の末尾のうち、いずれかの stop の先頭と一致する最も長い部分のバイト数を返します。
// 文字の途中では切りません。例: text = "abcEN"、stops = ["END"] → 2（"EN"）
func longestStopPrefixSuffix(text string, stops []string) int {
	longest := 0
	for _, stop := range stops {
		for length := min(len(stop)-1, len(text)); length > longest; length-- {
			start := len(text) - length
			if utf8.RuneStart(text[start]) && strings.HasPrefix(stop, text[start:]) {
				longest = length
				break
			}
		}
	}
	return longest
}

// callStart は選択したモデルの呼び出し開始タグを返します。不完全な出力の復元に使います。
func (p *OutputParser) callStart() string {
	if p.options.Gemma {
		return GemmaCallStart
	}
	return ToolCallStart
}

// callEnd は選択したモデルの呼び出し終了タグを返します。
func (p *OutputParser) callEnd() string {
	if p.options.Gemma {
		return GemmaCallEnd
	}
	return ToolCallEnd
}
