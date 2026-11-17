package grammar

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/cardy0828/pascallm/internal/pyjson"
)

// この file は、JSON Schema から JSON の文法を作ります。扱う範囲は次のとおりです。
//   - type（文字列または配列）、nullable
//   - object: properties・required・additionalProperties（プロパティは Schema の順に固定。required にないものは省ける）
//   - array: items・minItems・maxItems
//   - enum・const、anyOf・oneOf、allOf（要素 1 個のときだけ）、$ref（同じ文書の #/$defs/… と #/definitions/…）
//
// 数の範囲・文字列の長さ・pattern・format は保証しません（読み飛ばす）。
// properties のある object は、additionalProperties が true か Schema のときだけ、ほかのキーを許します。
// properties のない object は、どのキーでも許します。

// JSON の文法の上限です。モデルが空白や数字を出し続けて止まらなくなるのを防ぎます。
const (
	// maxIndentSpaces・maxIndentTabs は、改行の後の字下げに使えるスペース・タブの数の上限です。
	// 字下げした JSON（深さ 6 × 4 スペース、または深さ 6 のタブ）が入る大きさです。
	maxIndentSpaces = 24
	maxIndentTabs   = 6
	// maxLeadingNewlines は、出力の先頭（"</think>" の直後など）や、ツールの呼び出しの間に置ける改行の数の上限です。
	maxLeadingNewlines = 2
	// maxIntegerDigits・maxFractionDigits・maxExponentDigits は、数の各部分の桁数の上限です。
	maxIntegerDigits  = 20
	maxFractionDigits = 20
	maxExponentDigits = 4
	// maxUnrolledItems は、maxItems を規則に展開して守る上限です。これより大きい maxItems は守りません。
	maxUnrolledItems = 64
)

// jsonCompiler は JSON Schema を文法の規則にします。
type jsonCompiler struct {
	builder    *Builder
	gemma      bool // Gemma ツール DSL（文字列は特殊引用符、既知キーは裸の名前）。
	quoteToken int
	root       pyjson.Value   // $ref を解決する文書全体
	refs       map[string]int // $ref → 規則の番号（再帰する Schema のため、先に番号を決める）
	// 共通の規則です。
	whitespace, anyValue, stringRule, numberRule, integerRule int
}

// newJSONCompiler は、共通の規則（空白・文字列・数・任意の値）を作った jsonCompiler を返します。
func newJSONCompiler(builder *Builder, root pyjson.Value) *jsonCompiler {
	c := &jsonCompiler{builder: builder, root: root, refs: make(map[string]int)}
	c.whitespace = buildWhitespace(builder)
	c.stringRule = c.buildString()
	c.integerRule = c.buildInteger()
	c.numberRule = c.buildNumber()
	c.anyValue = c.buildAnyValue()
	return c
}

// buildWhitespace は、トークンの間の空白の規則を作ります。空白は次のどれかだけです。
//   - なし（例: {"a":1}）
//   - スペース 1 個（例: {"a": 1}）
//   - 改行 1 個と字下げ（スペースだけ maxIndentSpaces 個まで、またはタブだけ maxIndentTabs 個まで）
//
// 空白を広く許すと、モデルが書きたいキーを文法で禁じられたときに、空白やタブを上限まで出し続けることがあります
// （M6 の確認で見つかった）。字下げの形に限ると、すぐに次の文字に進みます。
func buildWhitespace(builder *Builder) int {
	spaces := builder.Repeat("indent-spaces", Byte(' '), 0, maxIndentSpaces)
	tabs := builder.Repeat("indent-tabs", Byte('\t'), 0, maxIndentTabs)
	indent := builder.Rule("indent", One(Ref(spaces)), One(Ref(tabs)))
	return builder.Rule("ws", nil, Literal(" "), Seq(Literal("\n"), One(Ref(indent))))
}

// leadingNewlines は、改行 0〜maxLeadingNewlines 個の規則を作ります（出力の先頭や、ツールの呼び出しの間）。
func leadingNewlines(builder *Builder) int {
	return builder.Repeat("newlines", Byte('\n'), 0, maxLeadingNewlines)
}

// ws は空白の要素です。
func (c *jsonCompiler) ws() []Element { return One(Ref(c.whitespace)) }

// buildString は JSON の文字列の規則を作ります。
// 文字は " と \ と制御文字以外のバイト（UTF-8 の 2 バイト目以降を含む）、または \ で始まるエスケープです。
func (c *jsonCompiler) buildString() int {
	b := c.builder
	var plain ByteSet
	plain.AddRange(' ', 0xff)
	plain[0] &^= 1 << '"' // '"'（0x22）を除く
	plain[1] &^= 1 << ('\\' - wordBits)
	var simpleEscape ByteSet
	for _, value := range []byte(`"\/bfnrt`) {
		simpleEscape.Add(value)
	}
	var hex ByteSet
	hex.AddRange('0', '9')
	hex.AddRange('a', 'f')
	hex.AddRange('A', 'F')
	escape := b.Rule("string-escape",
		One(Bytes(simpleEscape)),
		Seq(Literal("u"), One(Bytes(hex)), One(Bytes(hex)), One(Bytes(hex)), One(Bytes(hex))))
	character := b.Rule("string-char", One(Bytes(plain)), Seq(Literal(`\`), One(Ref(escape))))
	characters := b.Repeat("string-chars", Ref(character), 0, -1)
	return b.Rule("string", Seq(Literal(`"`), One(Ref(characters)), Literal(`"`)))
}

// digitSet は 0〜9 の集合を返します。
func digitSet() ByteSet {
	var digits ByteSet
	digits.AddRange('0', '9')
	return digits
}

// buildInteger は整数の規則を作ります。例: 0、-12、305
func (c *jsonCompiler) buildInteger() int {
	b := c.builder
	var nonZero ByteSet
	nonZero.AddRange('1', '9')
	rest := b.Repeat("integer-digits", Bytes(digitSet()), 0, maxIntegerDigits-1)
	magnitude := b.Rule("integer-magnitude", Literal("0"), Seq(One(Bytes(nonZero)), One(Ref(rest))))
	return b.Rule("integer", One(Ref(magnitude)), Seq(Literal("-"), One(Ref(magnitude))))
}

// buildNumber は数の規則を作ります（整数・小数・指数）。例: -1.5e+3
func (c *jsonCompiler) buildNumber() int {
	b := c.builder
	fractionDigits := b.Repeat("fraction-digits", Bytes(digitSet()), 1, maxFractionDigits)
	fraction := b.Rule("fraction", nil, Seq(Literal("."), One(Ref(fractionDigits))))
	var exponentMark, sign ByteSet
	exponentMark.Add('e')
	exponentMark.Add('E')
	sign.Add('+')
	sign.Add('-')
	exponentDigits := b.Repeat("exponent-digits", Bytes(digitSet()), 1, maxExponentDigits)
	exponentSign := b.Rule("exponent-sign", nil, One(Bytes(sign)))
	exponent := b.Rule("exponent", nil, Seq(One(Bytes(exponentMark)), One(Ref(exponentSign)), One(Ref(exponentDigits))))
	return b.Rule("number", Seq(One(Ref(c.integerRule)), One(Ref(fraction)), One(Ref(exponent))))
}

// buildAnyValue は任意の JSON の値の規則を作ります（object・array は任意の中身）。
func (c *jsonCompiler) buildAnyValue() int {
	b := c.builder
	value := b.NewRule("any-value")
	object := c.objectWithExtras(nil, value)
	array := c.arrayOf(value, 0, -1)
	b.SetAlternatives(value,
		One(Ref(object)), One(Ref(array)), One(Ref(c.stringRule)), One(Ref(c.numberRule)),
		Literal("true"), Literal("false"), Literal("null"))
	return value
}

// property は object の 1 プロパティです。
type property struct {
	key      string
	value    int  // 値の規則
	required bool // required にあるか
}

// objectWithExtras は、properties を Schema の順に並べ、extraValue が 0 以上なら任意のキーの追加も許す
// object の規則を作ります。extraValue が負なら追加のキーを許しません。
func (c *jsonCompiler) objectWithExtras(properties []property, extraValue int) int {
	b := c.builder
	extras := -1
	if extraValue >= 0 {
		extras = c.extraMembers(extraValue)
	}
	members := c.memberChain(properties, extras)
	return b.Rule("object", Seq(Literal("{"), c.ws(), One(Ref(members)), Literal("}")))
}

// memberChain は、プロパティの並びの規則を作ります。後ろから順に、「ここまでに 1 個以上書いたか」の
// 2 通りの規則を作ります（書いたなら次の前に "," が要る）。
func (c *jsonCompiler) memberChain(properties []property, extras int) int {
	b := c.builder
	// after[0] は「まだ 1 個も書いていない」、after[1] は「1 個以上書いた」ときの、残りの規則です。
	var after [2]int
	for written := range after {
		if extras >= 0 {
			after[written] = c.extraStart(written == 1, extras)
		} else {
			after[written] = b.Rule("members-end", nil)
		}
	}
	for index := len(properties) - 1; index >= 0; index-- {
		prop := properties[index]
		member := c.member(prop.key, prop.value)
		var next [2]int
		for written := range next {
			take := Seq(commaPrefix(written == 1, c), One(Ref(member)), c.ws(), One(Ref(after[1])))
			if prop.required {
				next[written] = b.Rule("members", take)
			} else {
				next[written] = b.Rule("members", take, One(Ref(after[written])))
			}
		}
		after = next
	}
	return after[0]
}

// commaPrefix は、すでに 1 個以上書いたなら "," と空白、そうでなければ空の並びを返します。
func commaPrefix(written bool, c *jsonCompiler) []Element {
	if written {
		return Seq(Literal(","), c.ws())
	}
	return nil
}

// extraStart は、追加のキーの並び（0 個以上）の規則を作ります。written はすでに 1 個以上書いたかです。
func (c *jsonCompiler) extraStart(written bool, extras int) int {
	b := c.builder
	if written {
		return b.Rule("extra-after", nil, Seq(Literal(","), c.ws(), One(Ref(extras))))
	}
	return b.Rule("extra-first", nil, One(Ref(extras)))
}

// extraMembers は、任意のキーと値の組を 1 個以上 "," でつないだ規則を作ります（最初の "," は含まない）。
func (c *jsonCompiler) extraMembers(value int) int {
	b := c.builder
	members := b.NewRule("extra-members")
	pair := b.Rule("extra-pair", Seq(One(Ref(c.stringRule)), c.ws(), Literal(":"), c.ws(), One(Ref(value)), c.ws()))
	b.SetAlternatives(members, One(Ref(pair)), Seq(One(Ref(pair)), Literal(","), c.ws(), One(Ref(members))))
	return members
}

// member は `"キー"` ws ":" ws 値 の規則を作ります。キーは json.dumps(ensure_ascii=False) の形で書きます。
func (c *jsonCompiler) member(key string, value int) int {
	quoted := pyjson.NewString(key).Dumps()
	if c.gemma {
		quoted = key
	}
	return c.builder.Rule("member-"+key, Seq(Literal(quoted), c.ws(), Literal(":"), c.ws(), One(Ref(value))))
}

// arrayOf は、item の規則の要素を minItems 個以上 maxItems 個以下（負なら上限なし）並べた配列の規則を作ります。
func (c *jsonCompiler) arrayOf(item, minItems, maxItems int) int {
	b := c.builder
	if maxItems > maxUnrolledItems {
		maxItems = -1
	}
	// rules[count] は、count 個書いた後の残りの規則です。上限なしなら、count が max(minItems, 1) 以上は
	// 同じ規則（lastCount 番目）を使います。
	lastCount := max(minItems, 1)
	if maxItems >= 0 {
		lastCount = maxItems
	}
	rules := make([]int, lastCount+1)
	for count := range rules {
		rules[count] = b.NewRule("array-items")
	}
	for count := lastCount; count >= 0; count-- {
		var alternatives [][]Element
		if count >= minItems {
			alternatives = append(alternatives, nil)
		}
		if maxItems < 0 || count < maxItems {
			next := min(count+1, lastCount)
			alternatives = append(alternatives, Seq(commaPrefix(count > 0, c), One(Ref(item)), c.ws(), One(Ref(rules[next]))))
		}
		b.SetAlternatives(rules[count], alternatives...)
	}
	return b.Rule("array", Seq(Literal("["), c.ws(), One(Ref(rules[0])), Literal("]")))
}

// literalValue は、値をそのまま書いた JSON（json.dumps の形）だけを許す規則の並びを返します。例: "a" → `"a"`
func (c *jsonCompiler) literalValue(value pyjson.Value) []Element {
	if c.gemma {
		return gemmaLiteral(value, c.quoteToken)
	}
	return Literal(value.Dumps())
}

// compile は Schema の規則を作ります。対応しない書き方ならエラーです。
// 例: compile({"type": "integer"}) → 整数の規則
func (c *jsonCompiler) compile(schema pyjson.Value) (int, error) {
	b := c.builder
	switch schema.Kind {
	case pyjson.Bool:
		if schema.Bool {
			return c.anyValue, nil
		}
		return 0, fmt.Errorf("grammar: schema false accepts nothing")
	case pyjson.Object:
	default:
		return 0, fmt.Errorf("grammar: schema must be an object or a boolean, got %s", schema.Dumps())
	}
	if ref, ok := schema.GetString("$ref"); ok {
		return c.compileRef(ref)
	}
	if value, ok := schema.Get("const"); ok {
		return b.Rule("const", c.literalValue(value)), nil
	}
	if values, ok := schema.Get("enum"); ok {
		if values.Kind != pyjson.Array || len(values.Items) == 0 {
			return 0, fmt.Errorf("grammar: enum must be a non-empty array")
		}
		alternatives := make([][]Element, 0, len(values.Items))
		for _, value := range values.Items {
			alternatives = append(alternatives, c.literalValue(value))
		}
		return b.Rule("enum", alternatives...), nil
	}
	for _, key := range []string{"anyOf", "oneOf"} {
		if alternatives, ok := schema.Get(key); ok {
			return c.compileUnion(alternatives)
		}
	}
	if all, ok := schema.Get("allOf"); ok {
		if all.Kind != pyjson.Array || len(all.Items) != 1 {
			return 0, fmt.Errorf("grammar: allOf is supported only with one schema")
		}
		return c.compile(all.Items[0])
	}
	types, err := schemaTypeList(schema)
	if err != nil {
		return 0, err
	}
	if len(types) == 0 {
		return c.anyValue, nil
	}
	alternatives := make([][]Element, 0, len(types))
	for _, typeName := range types {
		rule, err := c.compileType(typeName, schema)
		if err != nil {
			return 0, err
		}
		alternatives = append(alternatives, One(Ref(rule)))
	}
	return b.Rule("typed", alternatives...), nil
}

// schemaTypeList は type（と nullable）から型の名前の並びを返します。type がなく、properties か items が
// あれば object か array とみなします。例: {"type": ["string", "null"]} → [string null]
func schemaTypeList(schema pyjson.Value) ([]string, error) {
	var types []string
	typeValue, ok := schema.Get("type")
	switch {
	case ok && typeValue.Kind == pyjson.String:
		types = []string{typeValue.String}
	case ok && typeValue.Kind == pyjson.Array:
		for _, item := range typeValue.Items {
			if item.Kind != pyjson.String {
				return nil, fmt.Errorf("grammar: type must be a string or an array of strings")
			}
			types = append(types, item.String)
		}
	case ok:
		return nil, fmt.Errorf("grammar: type must be a string or an array of strings")
	default:
		if _, has := schema.Get("properties"); has {
			types = []string{"object"}
		} else if _, has := schema.Get("items"); has {
			types = []string{"array"}
		}
	}
	if nullable, ok := schema.Get("nullable"); ok && nullable.Truthy() && !contains(types, "null") && len(types) > 0 {
		types = append(types, "null")
	}
	return types, nil
}

// contains は、並びに値があるかを返します。例: contains([a b], "b") → true
func contains(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

// compileUnion は anyOf・oneOf の規則（どれか 1 個に合う）を作ります。
func (c *jsonCompiler) compileUnion(alternatives pyjson.Value) (int, error) {
	if alternatives.Kind != pyjson.Array || len(alternatives.Items) == 0 {
		return 0, fmt.Errorf("grammar: anyOf and oneOf must be non-empty arrays")
	}
	elements := make([][]Element, 0, len(alternatives.Items))
	for _, alternative := range alternatives.Items {
		rule, err := c.compile(alternative)
		if err != nil {
			return 0, err
		}
		elements = append(elements, One(Ref(rule)))
	}
	return c.builder.Rule("union", elements...), nil
}

// compileRef は $ref の規則を返します。同じ $ref は同じ規則を使い、再帰する Schema も扱えます。
// 例: "#/$defs/node" → root["$defs"]["node"] の規則
func (c *jsonCompiler) compileRef(ref string) (int, error) {
	if rule, ok := c.refs[ref]; ok {
		return rule, nil
	}
	target, err := resolveRef(c.root, ref)
	if err != nil {
		return 0, err
	}
	rule := c.builder.NewRule("ref " + ref)
	c.refs[ref] = rule
	inner, err := c.compile(target)
	if err != nil {
		return 0, err
	}
	c.builder.SetAlternatives(rule, One(Ref(inner)))
	return rule, nil
}

// resolveRef は、文書の中の "#/a/b" の形の参照をたどります。例: ("#/$defs/x") → root["$defs"]["x"]
func resolveRef(root pyjson.Value, ref string) (pyjson.Value, error) {
	path, ok := strings.CutPrefix(ref, "#")
	if !ok {
		return pyjson.Value{}, fmt.Errorf("grammar: only local $ref is supported, got %q", ref)
	}
	current := root
	for _, part := range strings.Split(strings.TrimPrefix(path, "/"), "/") {
		if part == "" {
			continue
		}
		part = strings.ReplaceAll(strings.ReplaceAll(part, "~1", "/"), "~0", "~")
		next, ok := current.Get(part)
		if !ok {
			return pyjson.Value{}, fmt.Errorf("grammar: $ref %q is not found", ref)
		}
		current = next
	}
	return current, nil
}

// compileType は、型 1 個の規則を作ります。例: ("array", {"items": {"type": "string"}}) → 文字列の配列
func (c *jsonCompiler) compileType(typeName string, schema pyjson.Value) (int, error) {
	b := c.builder
	switch typeName {
	case "string":
		return c.stringRule, nil
	case "integer":
		return c.integerRule, nil
	case "number":
		return c.numberRule, nil
	case "boolean":
		return b.Rule("boolean", Literal("true"), Literal("false")), nil
	case "null":
		return b.Rule("null", Literal("null")), nil
	case "array":
		return c.compileArray(schema)
	case "object":
		return c.compileObject(schema)
	default:
		return 0, fmt.Errorf("grammar: unknown type %q", typeName)
	}
}

// compileArray は array の規則を作ります。items がなければ任意の値の配列です。
func (c *jsonCompiler) compileArray(schema pyjson.Value) (int, error) {
	item := c.anyValue
	if items, ok := schema.Get("items"); ok {
		rule, err := c.compile(items)
		if err != nil {
			return 0, err
		}
		item = rule
	}
	minItems, err := nonNegativeInteger(schema, "minItems", 0)
	if err != nil {
		return 0, err
	}
	maxItems, err := nonNegativeInteger(schema, "maxItems", -1)
	if err != nil {
		return 0, err
	}
	if maxItems >= 0 && maxItems < minItems {
		return 0, fmt.Errorf("grammar: maxItems is smaller than minItems")
	}
	return c.arrayOf(item, minItems, maxItems), nil
}

// nonNegativeInteger は、キーの値（0 以上の整数）を返します。なければ fallback です。
// 例: ({"minItems": 2}, "minItems", 0) → 2
func nonNegativeInteger(schema pyjson.Value, key string, fallback int) (int, error) {
	value, ok := schema.Get(key)
	if !ok {
		return fallback, nil
	}
	if !value.IsInteger() {
		return 0, fmt.Errorf("grammar: %s must be an integer", key)
	}
	number, err := strconv.Atoi(value.Number)
	if err != nil || number < 0 {
		return 0, fmt.Errorf("grammar: %s must be a non-negative integer", key)
	}
	return number, nil
}

// compileObject は object の規則を作ります。
func (c *jsonCompiler) compileObject(schema pyjson.Value) (int, error) {
	properties, hasProperties := schema.Get("properties")
	extraValue := -1
	additional, hasAdditional := schema.Get("additionalProperties")
	switch {
	case !hasProperties && (!hasAdditional || additional.Kind != pyjson.Bool || additional.Bool):
		// properties がなければ、どのキーも許します（additionalProperties: false だけは空の object）。
		extraValue = c.anyValue
		if hasAdditional && additional.Kind == pyjson.Object {
			rule, err := c.compile(additional)
			if err != nil {
				return 0, err
			}
			extraValue = rule
		}
	case hasAdditional && additional.Kind == pyjson.Bool && additional.Bool:
		extraValue = c.anyValue
	case hasAdditional && additional.Kind == pyjson.Object:
		rule, err := c.compile(additional)
		if err != nil {
			return 0, err
		}
		extraValue = rule
	}
	required := make(map[string]bool)
	if list, ok := schema.Get("required"); ok && list.Kind == pyjson.Array {
		for _, item := range list.Items {
			if item.Kind == pyjson.String {
				required[item.String] = true
			}
		}
	}
	var props []property
	if hasProperties {
		if properties.Kind != pyjson.Object {
			return 0, fmt.Errorf("grammar: properties must be an object")
		}
		for _, member := range properties.Members {
			rule, err := c.compile(member.Value)
			if err != nil {
				return 0, fmt.Errorf("property %q: %w", member.Key, err)
			}
			props = append(props, property{key: member.Key, value: rule, required: required[member.Key]})
		}
	}
	return c.objectWithExtras(props, extraValue), nil
}

// JSONSchema は、Schema に合う JSON だけを受け付ける文法を作ります。先頭の改行は許し、JSON の後には何も
// 許しません（終わりのトークンだけを選べるようにして、空白を出し続けないようにする）。
// 例: JSONSchema({"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]})
func JSONSchema(schema pyjson.Value) (*Grammar, error) {
	builder := &Builder{}
	c := newJSONCompiler(builder, schema)
	value, err := c.compile(schema)
	if err != nil {
		return nil, err
	}
	root := builder.Rule("root", Seq(One(Ref(leadingNewlines(builder))), One(Ref(value))))
	return builder.Build(root), nil
}

// JSONObject は、任意の JSON の object だけを受け付ける文法を作ります（response_format の json_object）。
// 先頭の改行は許し、object の後には何も許しません。
func JSONObject() *Grammar {
	builder := &Builder{}
	c := newJSONCompiler(builder, pyjson.Value{})
	object := c.objectWithExtras(nil, c.anyValue)
	root := builder.Rule("root", Seq(One(Ref(leadingNewlines(builder))), One(Ref(object))))
	return builder.Build(root)
}
