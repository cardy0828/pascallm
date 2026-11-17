package generate

import (
	"slices"
	"testing"
)

// fakeSnapshot はコピー回数と復元内容を観測する容量検査用モデルです。
type fakeSnapshot struct {
	calls    int
	restored int
}

// SnapshotBytes は固定の16 byte payloadを返します。
func (f *fakeSnapshot) SnapshotBytes() int64 { return 16 }

// SnapshotState は呼出番号を不変な保存状態にします。
func (f *fakeSnapshot) SnapshotState() (any, error) { f.calls++; return f.calls, nil }

// RestoreSnapshot は復元した保存番号を記録します。
func (f *fakeSnapshot) RestoreSnapshot(s any) error { f.restored = s.(int); return nil }

// TestPrefixLRU は確保前の容量判定・最長一致・LRU・重複・上限超過をまとめて検査します。
func TestPrefixLRU(t *testing.T) {
	f := &fakeSnapshot{}
	p := &prefixCache{model: f, maxBytes: 80, maxEntries: 2}
	for _, tokens := range [][]int{{1, 2}, {3, 4}} {
		if err := p.save(tokens); err != nil {
			t.Fatal(err)
		}
	}
	i := p.best([]int{1, 2, 9})
	got, err := p.restore(i)
	if err != nil || !slices.Equal(got, []int{1, 2}) || f.restored != 1 {
		t.Fatal(got, err)
	}
	if err := p.save([]int{5, 6}); err != nil {
		t.Fatal(err)
	}
	if p.best([]int{3, 4, 9}) != -1 || p.bytes != 64 || p.evictions != 1 {
		t.Fatal(p.report())
	}
	if err := p.save([]int{1, 2}); err != nil {
		t.Fatal(err)
	}
	if f.calls != 3 {
		t.Fatal("duplicate snapshot")
	}
	if err := p.save(make([]int, 20)); err != nil {
		t.Fatal(err)
	}
	if f.calls != 3 || p.skipped != 1 || p.bytes > p.maxBytes {
		t.Fatal(p.report())
	}
	if p.best([]int{1, 2}) != -1 {
		t.Fatal("last prompt token must be recomputed")
	}
}

// TestPrefixLongestAndEntryLimit は共有prefixの最長選択とbyte上限未満でも働く件数上限を確認します。
func TestPrefixLongestAndEntryLimit(t *testing.T) {
	f := &fakeSnapshot{}
	p := &prefixCache{model: f, maxBytes: 1024, maxEntries: 2}
	for _, tokens := range [][]int{{1}, {1, 2}} {
		if err := p.save(tokens); err != nil {
			t.Fatal(err)
		}
	}
	if i := p.best([]int{1, 2, 3}); i < 0 || len(p.entries[i].tokens) != 2 {
		t.Fatal("longest prefix not selected")
	}
	if err := p.save([]int{9}); err != nil {
		t.Fatal(err)
	}
	if len(p.entries) != 2 || p.best([]int{1, 8}) != -1 || p.evictions != 1 {
		t.Fatal(p.report())
	}
}
