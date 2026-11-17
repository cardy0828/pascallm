package tokenizer

import (
	"container/heap"
	"fmt"
)

// noSymbol は、bpeSymbol の前後がないことを表す番号です。
const noSymbol = -1

// bpeSymbol は、BPE の途中の 1 記号（マージで伸びていく文字列）です。前後の記号を番号でつなぎます。
type bpeSymbol struct {
	text    string
	prev    int  // 前の記号の番号（なければ noSymbol）
	next    int  // 次の記号の番号（なければ noSymbol）
	alive   bool // false なら左の記号に吸収済み
	version int  // マージで text が変わるたびに 1 増やす（古い候補を見分けるため）
}

// mergeCandidate は、隣り合う 2 記号のマージの候補です。
type mergeCandidate struct {
	rank         int // マージ規則の優先順位（小さいほど先）
	left, right  int // 記号の番号
	leftVersion  int // 候補を作ったときの左の記号の version
	rightVersion int // 候補を作ったときの右の記号の version
}

// mergeQueue は、優先順位 → 左の位置 の順に小さい候補を取り出す優先度付きキューです
// （Hugging Face の tokenizers と同じ順序。同じ順位なら左から先にマージする）。
type mergeQueue []mergeCandidate

// Len・Less・Swap・Push・Pop は container/heap のための関数です。
func (q mergeQueue) Len() int { return len(q) }

// Less は、i の候補を j より先にマージするかを返します（優先順位、次に左の位置で比べる）。
func (q mergeQueue) Less(i, j int) bool {
	if q[i].rank != q[j].rank {
		return q[i].rank < q[j].rank
	}
	return q[i].left < q[j].left
}

// Swap は i と j の候補を入れ替えます。
func (q mergeQueue) Swap(i, j int) { q[i], q[j] = q[j], q[i] }

// Push は候補を末尾に加えます（heap.Push から呼ばれる）。
func (q *mergeQueue) Push(x any) { *q = append(*q, x.(mergeCandidate)) }

// Pop は末尾の候補を取り出します（heap.Pop から呼ばれる）。
func (q *mergeQueue) Pop() any {
	old := *q
	last := old[len(old)-1]
	*q = old[:len(old)-1]
	return last
}

// bpe は、byte レベルの文字に置き換えた 1 片をトークン列にします。
// ignore_merges のため、片全体が語彙にあればそのまま 1 トークンにします。
// それ以外は 1 文字ずつの記号から始め、優先順位の高いマージ規則から順に隣り合う記号をつなぎます。
// 例: "Ġhello"（語彙にある）→ [id("Ġhello")]、"Ġhelloxyz" → マージした結果の各記号の番号
func (t *Tokenizer) bpe(word string) ([]int, error) {
	if id, ok := t.vocab[word]; ok && !t.gemma {
		return []int{id}, nil
	}
	characters := []rune(word)
	symbols := make([]bpeSymbol, len(characters))
	for index, character := range characters {
		symbols[index] = bpeSymbol{text: string(character), prev: index - 1, next: index + 1, alive: true}
	}
	symbols[len(symbols)-1].next = noSymbol

	queue := &mergeQueue{}
	addCandidate := func(left int) {
		right := symbols[left].next
		if right == noSymbol {
			return
		}
		rank, ok := t.mergeRanks[mergeKey{left: symbols[left].text, right: symbols[right].text}]
		if !ok {
			return
		}
		heap.Push(queue, mergeCandidate{
			rank: rank, left: left, right: right,
			leftVersion: symbols[left].version, rightVersion: symbols[right].version,
		})
	}
	for index := 0; index+1 < len(symbols); index++ {
		addCandidate(index)
	}

	for queue.Len() > 0 {
		candidate := heap.Pop(queue).(mergeCandidate)
		left, right := &symbols[candidate.left], &symbols[candidate.right]
		stale := !left.alive || !right.alive || left.next != candidate.right ||
			left.version != candidate.leftVersion || right.version != candidate.rightVersion
		if stale {
			continue
		}
		left.text += right.text
		left.version++
		left.next = right.next
		right.alive = false
		if right.next != noSymbol {
			symbols[right.next].prev = candidate.left
		}
		if left.prev != noSymbol {
			addCandidate(left.prev)
		}
		addCandidate(candidate.left)
	}

	var ids []int
	for index := 0; index != noSymbol; index = symbols[index].next {
		id, ok := t.vocab[symbols[index].text]
		if !ok && t.gemma {
			for _, value := range []byte(symbols[index].text) {
				fallback, found := t.vocab[fmt.Sprintf("<0x%02X>", value)]
				if !found {
					return nil, fmt.Errorf("tokenizer: missing byte fallback %d", value)
				}
				ids = append(ids, fallback)
			}
			continue
		}
		if !ok {
			return nil, fmt.Errorf("tokenizer: BPE symbol %q is not in the vocabulary", symbols[index].text)
		}
		ids = append(ids, id)
	}
	return ids, nil
}
