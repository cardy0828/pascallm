package gemma4gpu

import (
	"slices"
	"testing"
)

// TestHistoryCandidates は8個一致した履歴から、文脈上限と指定幅に収まる確定候補だけを得ることを確認します。
func TestHistoryCandidates(t *testing.T) {
	prefix := []int{1, 2, 3, 4, 5, 6, 7, 8}
	continuation := []int{11, 12, 13, 14, 15, 16, 17}
	history := append(append(slices.Clone(prefix), continuation...), prefix...)
	for _, tc := range []struct{ limit, remaining, count int }{{3, 20, 3}, {7, 20, 7}, {7, 4, 4}, {7, 2, 0}} {
		got := historyCandidates(history, tc.limit, tc.remaining)
		if !slices.Equal(got, continuation[:tc.count]) {
			t.Fatalf("limit=%d remaining=%d: got %v", tc.limit, tc.remaining, got)
		}
	}
}
