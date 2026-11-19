package gemma4gpu

import "testing"

// TestSnapshotRejectInvalid は不正な保存値をCUDA操作より前に拒否することを確認します。
func TestSnapshotRejectInvalid(t *testing.T) {
	model := &Model{}
	for _, value := range []any{nil, (*kvSnapshot)(nil), 42, &kvSnapshot{owner: &Model{}}} {
		if err := model.RestoreSnapshot(value); err == nil {
			t.Fatal("invalid snapshot accepted")
		}
	}
}

// TestCompactRestoreWindow は小さいリングが上書きされる境界で復元可否を切り替えることを検査する。
func TestCompactRestoreWindow(t *testing.T) {
	m := Model{restoreSlack: 128, savedPosition: 1024, position: 1152}
	if !m.CanRestore() {
		t.Fatal("saved prefix should fit at the boundary")
	}
	m.position++
	if m.CanRestore() {
		t.Fatal("overwritten prefix still restorable")
	}
}
