package gemma4gpu

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// publicTestAsset はPASCA_DATA_DIR/PASCA_MODELS_DIR、またはcheckout直下の資産を使います。
// 例: publicTestAsset(t, "data", "m7/gemma12b-c4096.kld")。GPU検査時だけ呼びます。
func publicTestAsset(t *testing.T, kind, name string) string {
	t.Helper()
	if directory := os.Getenv("PASCA_" + strings.ToUpper(kind) + "_DIR"); directory != "" {
		return filepath.Join(directory, name)
	}
	directory, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := os.Stat(filepath.Join(directory, "go.mod")); err == nil {
			return filepath.Join(directory, kind, name)
		}
		parent := filepath.Dir(directory)
		if parent == directory {
			t.Fatal("cannot locate checkout assets")
			return ""
		}
		directory = parent
	}
}
