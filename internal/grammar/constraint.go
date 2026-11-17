package grammar

import "fmt"

// この file は、文法をトークンの選び方に当てはめる部分です（generate.Constraint を満たす）。

// Vocabulary は、トークンの番号から文字列のバイト列を引くためのものです（tokenizer.Tokenizer が満たす）。
type Vocabulary interface {
	Size() int                // 番号の数。例: 125033
	TokenBytes(id int) []byte // 復元したときのバイト列（追加トークンはその文字列）。ない番号なら nil
	IsAdded(id int) bool      // 追加トークン（<think>・<|im_end|> など）か
}

// Constraint は、生成の途中で、次に選んでよいトークンを文法で決めます。
//   - 思考の途中（"</think>" まで）は縛りません。ただし終わりのトークンは選べません（文法の出力に進ませるため）。
//   - "</think>" の後（思考なしなら最初）から文法で縛ります。
//   - 終わりのトークン（<|im_end|> など）は、文法を最後まで照合した状態でだけ選べます。
//   - 追加トークンは、文法がそのトークンを求めている位置でだけ選べます。ほかのトークンはバイト列で照合します。
type Constraint struct {
	grammar     *Grammar
	vocabulary  Vocabulary
	state       State
	inReasoning bool
	thinkEnd    int
	endTokens   map[int]bool
}

// NewConstraint は Constraint を作ります。startsInReasoning はプロンプトが "<think>" で終わるか、thinkEnd は
// "</think>" の番号、endTokens は生成を止めるトークンです。
// 例: NewConstraint(g, tok, true, 124902, []int{124900, 124895})
func NewConstraint(grammar *Grammar, vocabulary Vocabulary, startsInReasoning bool, thinkEnd int, endTokens []int) *Constraint {
	ends := make(map[int]bool, len(endTokens))
	for _, id := range endTokens {
		ends[id] = true
	}
	return &Constraint{
		grammar:     grammar,
		vocabulary:  vocabulary,
		state:       grammar.Initial(),
		inReasoning: startsInReasoning,
		thinkEnd:    thinkEnd,
		endTokens:   ends,
	}
}

// Allows は、今の状態で id を選んでよいかを返します。状態は変えません。例: c.Allows(279) → true
func (c *Constraint) Allows(id int) bool {
	if id < 0 || id >= c.vocabulary.Size() {
		return false
	}
	data := c.vocabulary.TokenBytes(id)
	switch {
	case data == nil:
		return false
	case c.inReasoning:
		return !c.endTokens[id]
	case c.endTokens[id]:
		return c.state.Accepting()
	case c.vocabulary.IsAdded(id):
		return !c.state.AdvanceToken(c.grammar, id).Dead()
	default:
		return !c.state.AdvanceBytes(c.grammar, data).Dead()
	}
}

// Accept は、選んだトークンで状態を進めます。文法に合わないトークンならエラーです（Allows で確かめてから選ぶ）。
func (c *Constraint) Accept(id int) error {
	switch {
	case c.inReasoning:
		if id == c.thinkEnd {
			c.inReasoning = false
		}
		return nil
	case c.endTokens[id]:
		return nil
	}
	var next State
	if c.vocabulary.IsAdded(id) {
		next = c.state.AdvanceToken(c.grammar, id)
	} else {
		next = c.state.AdvanceBytes(c.grammar, c.vocabulary.TokenBytes(id))
	}
	if next.Dead() {
		return fmt.Errorf("grammar: token %d does not match the grammar", id)
	}
	c.state = next
	return nil
}

// TokenBan は、決めたトークンだけを選べなくする制約です（tool_choice: "none" で <tool_call> を禁じる）。
type TokenBan struct {
	size   int
	banned map[int]bool
}

// NewTokenBan は TokenBan を作ります。size は番号の数です。例: NewTokenBan(125033, 125019)
func NewTokenBan(size int, banned ...int) *TokenBan {
	set := make(map[int]bool, len(banned))
	for _, id := range banned {
		set[id] = true
	}
	return &TokenBan{size: size, banned: set}
}

// Allows は、id が禁じたトークンでなく、番号の範囲内かを返します。
func (b *TokenBan) Allows(id int) bool { return id >= 0 && id < b.size && !b.banned[id] }

// Accept は何もしません（状態を持たない）。
func (b *TokenBan) Accept(int) error { return nil }

// AllowsGreedyFastPath は静的な禁止集合なので、まず argmax を確認してよいことを示します。
// 禁止された argmax を返してよいという意味ではありません。呼出側は Allows で必ず検査します。
func (b *TokenBan) AllowsGreedyFastPath() bool { return true }
