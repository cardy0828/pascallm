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
