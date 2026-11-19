package gemma4gpu

const minSuffix = 4
const longSuffix = 8
const longestSuffix = 32
const searchWindow = 4096

// SetLookupHistory は再利用した prefix も含む現要求の入力を登録する。nil は終了・キャンセル時に履歴を破棄する。
func (m *Model) SetLookupHistory(prefix []int) {
	if m.mtp == nil || m.mtp.lookupLimit == 0 {
		return
	}
	m.mtp.history = nil
	if prefix != nil {
		m.mtp.history = append([]int{}, prefix[max(0, len(prefix)-searchWindow):]...)
	}
}

// historyCandidates は現在の末尾と同じ過去の並びを探し、既知の続きだけを返す。
// 同じ長さなら直近を優先する。例えば4連続一致なら最大3候補、8連続一致なら最大7候補。
func historyCandidates(h []int, limit, remaining int) []int {
	n := len(h)
	best, bestEnd := 0, 0
	first := max(0, n-searchWindow)
	for end := n - 3; end >= first+minSuffix; end-- {
		matched := 0
		for matched < longestSuffix && end-matched > first && h[end-matched-1] == h[n-matched-1] {
			matched++
		}
		if matched >= minSuffix && matched > best {
			best, bestEnd = matched, end
		}
		if best == longestSuffix {
			break
		}
	}
	if best == 0 {
		return nil
	}
	if best < longSuffix {
		limit = 3
	}
	count := min(limit, remaining, n-bestEnd)
	if count < 3 {
		return nil
	}
	return h[bestEnd : bestEnd+count]
}
