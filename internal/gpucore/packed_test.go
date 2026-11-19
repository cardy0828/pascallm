package gpucore

import (
	"math"
	"testing"

	"github.com/cardy0828/pascallm/internal/cuda"
)

// TestPackedLayout は見積もりと実配置で同じ余白を取り、領域外とオーバーフローを拒むことを確認します。
func TestPackedLayout(t *testing.T) {
	const base cuda.DevicePtr = 4096
	sizes := []int{1, 256, 3}
	expected := []cuda.DevicePtr{base, base + 256, base + 512}
	group := allocationGroup{planning: true}
	for _, n := range sizes {
		if _, err := group.next(n); err != nil {
			t.Fatal(err)
		}
	}
	if group.offset != 515 {
		t.Fatalf("size %d", group.offset)
	}
	group = allocationGroup{size: group.offset, base: base}
	for i, n := range sizes {
		p, err := group.next(n)
		if err != nil || p != expected[i] {
			t.Fatalf("entry %d: %d, %v", i, p, err)
		}
	}
	if _, err := group.next(1); err == nil {
		t.Fatal("accepted allocation past group end")
	}
	group = allocationGroup{planning: true, offset: math.MaxInt}
	if _, err := group.next(1); err == nil {
		t.Fatal("accepted overflowing allocation")
	}
}
