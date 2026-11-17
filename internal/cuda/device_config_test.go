package cuda

import (
	"strings"
	"testing"
)

// TestMissingDeviceSelection はGPUライブラリを呼ぶ前に未指定を検出します。
func TestMissingDeviceSelection(t *testing.T) {
	t.Setenv("PASCA_DEVICE_UUID", "")
	_, err := FindDeviceByUUID("")
	if err == nil || !strings.Contains(err.Error(), "PASCA_DEVICE_UUID") {
		t.Fatalf("expected device selection error, got %v", err)
	}
}
