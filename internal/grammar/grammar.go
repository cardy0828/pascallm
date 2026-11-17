// Package grammar は、生成する文字列を文法で縛るための、文法の表現と照合です（M6）。
//
// 文法は規則の集まりで、規則は「選択肢」の並び、選択肢は「要素」の並びです。要素は次の 3 種類です。
//   - バイトの集合: 1 バイトがその集合に入れば進む（例: [0-9]）
//   - 別の規則: その規則のどれかの選択肢に進む
//   - トークン: 追加トークン 1 個（例: <tool_call>）。バイトではなくトークンの番号で照合する
//
// 照合の状態は、llama.cpp の文法と同じく「規則の位置のスタック」の集合です。スタックの一番上は、
// 次に照合する終端（バイトの集合かトークン）を指します。スタックは親をたどる連結リストで、
// 状態を進めても元の状態は変わらないため、候補のトークンを試すときに複製の手間がかかりません。
//
// 選択肢の最後の要素が規則のときは、戻り先を積まずに進みます（末尾呼び出し）。こうすると、
// 「S → 文字 S | 空」のような繰り返しで、スタックが文字数だけ深くなりません。
// 左再帰（S → S x）は扱えません。文法を作る側（json.go・toolcall.go）は左再帰を作りません。
package grammar

// ByteSet は 256 個のバイトの集合です。
type ByteSet [4]uint64

// wordBits は ByteSet の 1 要素のビット数です。
const wordBits = 64

// Add はバイト b を集合に加えます。例: set.Add('a')
func (s *ByteSet) Add(b byte) { s[b/wordBits] |= 1 << (b % wordBits) }

// AddRange は first〜last（両端を含む）のバイトを集合に加えます。例: set.AddRange('0', '9')
func (s *ByteSet) AddRange(first, last byte) {
	for value := int(first); value <= int(last); value++ {
		s.Add(byte(value))
	}
}

// Has はバイト b が集合に入っているかを返します。例: digits.Has('5') → true
func (s *ByteSet) Has(b byte) bool { return s[b/wordBits]&(1<<(b%wordBits)) != 0 }

// elementKind は要素の種類です。
type elementKind uint8

const (
	elementBytes elementKind = iota // バイトの集合
	elementRule                     // 別の規則
	elementToken                    // 追加トークン 1 個
)

// Element は選択肢の要素 1 個です。
type Element struct {
	kind  elementKind
	bytes *ByteSet // elementBytes のときの集合
	rule  int      // elementRule のときの規則の番号
	token int      // elementToken のときのトークンの番号
}

// Bytes はバイトの集合の要素を作ります。例: Bytes(digits)
func Bytes(set ByteSet) Element { return Element{kind: elementBytes, bytes: &set} }

// Byte は 1 バイトだけの要素を作ります。例: Byte('{')
func Byte(b byte) Element {
	var set ByteSet
	set.Add(b)
	return Bytes(set)
}

// Ref は規則を参照する要素を作ります。例: Ref(valueRule)
func Ref(rule int) Element { return Element{kind: elementRule, rule: rule} }

// Token は追加トークン 1 個の要素を作ります。例: Token(125019)
func Token(id int) Element { return Element{kind: elementToken, token: id} }

// Literal は、文字列の各バイトを 1 個ずつの要素にした並びを返します。例: Literal("null") → n u l l
func Literal(text string) []Element {
	elements := make([]Element, 0, len(text))
	for index := 0; index < len(text); index++ {
		elements = append(elements, Byte(text[index]))
	}
	return elements
}

// Seq は要素の並びと並びをつなげます。例: Seq(Literal("\"a\""), []Element{Ref(ws)})
func Seq(parts ...[]Element) []Element {
	var joined []Element
	for _, part := range parts {
		joined = append(joined, part...)
	}
	return joined
}

// One は要素 1 個だけの並びを返します。Seq の引数に使います。例: Seq(One(Ref(a)), Literal(","))
func One(element Element) []Element { return []Element{element} }

// rule は規則 1 個です。
type rule struct {
	name         string      // 確認用の名前。例: "string"
	alternatives [][]Element // 選択肢
}

// Grammar は文法です。Builder で作ります。
type Grammar struct {
	rules []rule
	root  int
}

// Builder は文法を組み立てます。規則を先に作ってから選択肢を決められるため、再帰する規則も書けます。
type Builder struct {
	rules []rule
}

// NewRule は選択肢のない規則を作り、番号を返します。後で SetAlternatives で選択肢を決めます。
// 例: value := b.NewRule("value")
func (b *Builder) NewRule(name string) int {
	b.rules = append(b.rules, rule{name: name})
	return len(b.rules) - 1
}

// SetAlternatives は規則の選択肢を決めます。空の並びは「何もない」選択肢です。
// 例: b.SetAlternatives(boolean, Literal("true"), Literal("false"))
func (b *Builder) SetAlternatives(id int, alternatives ...[]Element) {
	b.rules[id].alternatives = alternatives
}

// Rule は選択肢を決めた規則を作ります。例: comma := b.Rule("comma", Literal(","))
func (b *Builder) Rule(name string, alternatives ...[]Element) int {
	id := b.NewRule(name)
	b.SetAlternatives(id, alternatives...)
	return id
}

// Repeat は、element を min 回以上 max 回以下並べる規則を作ります。max が負なら上限なしです。
// 上限ありのときは、残りの回数ごとの規則を鎖のようにつなぎます。
// 例: Repeat("digits", Bytes(d), 1, -1)（1 個以上）、Repeat("ws", Bytes(space), 0, 32)
func (b *Builder) Repeat(name string, element Element, min, max int) int {
	tail := b.NewRule(name + "-tail")
	if max < 0 {
		b.SetAlternatives(tail, Seq(One(element), One(Ref(tail))), nil)
	} else {
		b.SetAlternatives(tail, nil)
		for count := min; count < max; count++ {
			tail = b.Rule(name+"-tail", Seq(One(element), One(Ref(tail))), nil)
		}
	}
	head := tail
	for count := 0; count < min; count++ {
		head = b.Rule(name, Seq(One(element), One(Ref(head))))
	}
	return head
}

// Build は root を最初の規則とする文法を返します。
func (b *Builder) Build(root int) *Grammar {
	return &Grammar{rules: b.rules, root: root}
}

// Either は、a と b のどちらかに合う文字列を受け付ける文法を作ります。b の規則は番号をずらして後ろに置きます。
// 例: Either(JSONObject(), toolCalls) → JSON の object か、ツールの呼び出し
func Either(a, b *Grammar) *Grammar {
	rules := make([]rule, 0, len(a.rules)+len(b.rules)+1)
	rules = append(rules, a.rules...)
	offset := len(a.rules)
	for _, original := range b.rules {
		shifted := rule{name: original.name, alternatives: make([][]Element, len(original.alternatives))}
		for index, alternative := range original.alternatives {
			copied := make([]Element, len(alternative))
			for position, element := range alternative {
				if element.kind == elementRule {
					element.rule += offset
				}
				copied[position] = element
			}
			shifted.alternatives[index] = copied
		}
		rules = append(rules, shifted)
	}
	rules = append(rules, rule{name: "either", alternatives: [][]Element{One(Ref(a.root)), One(Ref(b.root + offset))}})
	return &Grammar{rules: rules, root: len(rules) - 1}
}

// frame はスタックの 1 段です。rule の alternative 番目の選択肢の index 番目の要素を次に照合します。
// parent は戻り先（nil なら一番下）です。
type frame struct {
	rule        int32
	alternative int32
	index       int32
	parent      *frame
}

// State は照合の状態です。変更しないため、そのまま複数の候補の起点に使えます。
type State struct {
	stacks    []*frame // 一番上が終端を指すスタック
	accepting bool     // 文法の最後まで照合し終えた道があるか
}

// maxExpandDepth は、規則をたどる深さの上限です。左再帰の文法で止まらなくなるのを防ぎます。
const maxExpandDepth = 1000

// Initial は、何も照合していない状態を返します。例: state := g.Initial()
func (g *Grammar) Initial() State {
	var state State
	for alternative := range g.rules[g.root].alternatives {
		g.expand(&frame{rule: int32(g.root), alternative: int32(alternative)}, &state, 0)
	}
	return state
}

// expand は、スタックの一番上が終端になるまで規則をたどり、できたスタックを state に加えます。
// 選択肢の最後まで来たら戻り先に進み、戻り先がなければ文法の終わりに着いたことを記録します。
func (g *Grammar) expand(top *frame, state *State, depth int) {
	if depth > maxExpandDepth {
		return
	}
	for top != nil {
		elements := g.rules[top.rule].alternatives[top.alternative]
		if int(top.index) < len(elements) {
			break
		}
		top = top.parent
	}
	if top == nil {
		state.accepting = true
		return
	}
	element := g.rules[top.rule].alternatives[top.alternative][top.index]
	if element.kind != elementRule {
		state.addStack(top)
		return
	}
	// 規則の後に続く位置を戻り先にします。規則が選択肢の最後なら、続く位置は積まずに今の戻り先を使います。
	next := top.parent
	if int(top.index)+1 < len(g.rules[top.rule].alternatives[top.alternative]) {
		next = &frame{rule: top.rule, alternative: top.alternative, index: top.index + 1, parent: top.parent}
	}
	for alternative := range g.rules[element.rule].alternatives {
		g.expand(&frame{rule: int32(element.rule), alternative: int32(alternative), parent: next}, state, depth+1)
	}
}

// addStack は、同じ内容のスタックがなければ加えます。
func (s *State) addStack(top *frame) {
	for _, existing := range s.stacks {
		if sameStack(existing, top) {
			return
		}
	}
	s.stacks = append(s.stacks, top)
}

// sameStack は 2 つのスタックが同じ位置の並びかを返します。
func sameStack(a, b *frame) bool {
	for a != nil && b != nil {
		if a == b {
			return true
		}
		if a.rule != b.rule || a.alternative != b.alternative || a.index != b.index {
			return false
		}
		a, b = a.parent, b.parent
	}
	return a == b
}

// AdvanceByte は、1 バイト照合した後の状態を返します。合う道がなければ Dead な状態です。
// 例: g.Initial().AdvanceByte(g, '{')
func (s State) AdvanceByte(g *Grammar, value byte) State {
	var next State
	for _, top := range s.stacks {
		element := g.rules[top.rule].alternatives[top.alternative][top.index]
		if element.kind == elementBytes && element.bytes.Has(value) {
			g.expand(&frame{rule: top.rule, alternative: top.alternative, index: top.index + 1, parent: top.parent}, &next, 0)
		}
	}
	return next
}

// AdvanceBytes は、バイト列を順に照合した後の状態を返します。続けられる道がなくなったら、残りのバイトは
// 照合できないため、そこで止めます。最後のバイトで文法を終えた場合は、終わった状態（Accepting）を返します。
// 例: state.AdvanceBytes(g, []byte("true"))
func (s State) AdvanceBytes(g *Grammar, data []byte) State {
	for index, value := range data {
		s = s.AdvanceByte(g, value)
		isLast := index == len(data)-1
		if !s.CanContinue() && !isLast {
			return State{}
		}
	}
	return s
}

// AdvanceToken は、追加トークン 1 個を照合した後の状態を返します。例: state.AdvanceToken(g, 125019)
func (s State) AdvanceToken(g *Grammar, id int) State {
	var next State
	for _, top := range s.stacks {
		element := g.rules[top.rule].alternatives[top.alternative][top.index]
		if element.kind == elementToken && element.token == id {
			g.expand(&frame{rule: top.rule, alternative: top.alternative, index: top.index + 1, parent: top.parent}, &next, 0)
		}
	}
	return next
}

// Dead は、続けられる道も、終わった道もないかを返します（その手前の入力は文法に合わない）。
func (s State) Dead() bool { return len(s.stacks) == 0 && !s.accepting }

// Accepting は、ここで終えてよいか（文法の最後まで照合した道があるか）を返します。
func (s State) Accepting() bool { return s.accepting }

// CanContinue は、さらにバイトかトークンを照合できる道があるかを返します。
func (s State) CanContinue() bool { return len(s.stacks) > 0 }
