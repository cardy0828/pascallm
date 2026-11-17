package gemma4

import (
	"math"
	"testing"
)

// near は、2 つの値の差が tolerance 以下かを返します。
func near(got, want, tolerance float64) bool { return math.Abs(got-want) <= tolerance }

// TestGelu は、GELU の tanh による近似が既知の値になることを確かめます。
func TestGelu(t *testing.T) {
	for _, test := range []struct{ x, want float64 }{{0, 0}, {1, 0.841192}, {-1, -0.158808}, {3, 2.996363}} {
		if got := float64(gelu(float32(test.x))); !near(got, test.want, 1e-5) {
			t.Errorf("gelu(%g) = %g, want %g", test.x, got, test.want)
		}
	}
}

// TestSoftcap は、logits の上限（cap × tanh(x / cap)）と、上限 0 で何もしないことを確かめます。
func TestSoftcap(t *testing.T) {
	values := []float32{0, 30, -60}
	softcapInPlace(values, 30)
	want := []float64{0, 30 * math.Tanh(1), 30 * math.Tanh(-2)}
	for index := range values {
		if !near(float64(values[index]), want[index], 1e-4) {
			t.Errorf("softcap[%d] = %g, want %g", index, values[index], want[index])
		}
	}
	unchanged := []float32{100}
	softcapInPlace(unchanged, 0)
	if unchanged[0] != 100 {
		t.Errorf("softcap with limit 0 changed the value to %g", unchanged[0])
	}
}

// TestFirstVisiblePosition は、窓のある層とない層で見える最初の位置を確かめます。
func TestFirstVisiblePosition(t *testing.T) {
	for _, test := range []struct{ position, window, want int }{
		{0, 1024, 0}, {1023, 1024, 0}, {1024, 1024, 1}, {1500, 1024, 477}, {5000, noWindowLimit, 0},
	} {
		if got := firstVisiblePosition(test.position, test.window); got != test.want {
			t.Errorf("firstVisiblePosition(%d, %d) = %d, want %d", test.position, test.window, got, test.want)
		}
	}
}

// TestRopeInverseFrequencies は、逆周波数の式（底^(−2i / head_dim) ÷ 係数）を確かめます。
func TestRopeInverseFrequencies(t *testing.T) {
	plain := ropeInverseFrequencies(10000, 256, nil)
	if len(plain) != 128 || plain[0] != 1 || !near(float64(plain[64]), math.Pow(10000, -0.5), 1e-6) {
		t.Errorf("plain frequencies: len %d, [0] %g, [64] %g", len(plain), plain[0], plain[64])
	}
	factors := make([]float32, 256)
	for index := range factors {
		factors[index] = 1
	}
	factors[200] = 1e30
	scaled := ropeInverseFrequencies(1e6, 512, factors)
	if scaled[200] > 1e-25 || !near(float64(scaled[128]), math.Pow(1e6, -0.5), 1e-7) {
		t.Errorf("scaled frequencies: [200] %g, [128] %g", scaled[200], scaled[128])
	}
}

// TestRmsNormAndRope は、正規化（重みあり・なし）と RoPE の式を確かめます。
func TestRmsNormAndRope(t *testing.T) {
	values := []float32{3, 4}
	normed := make([]float32, 2)
	rmsNorm(normed, values, []float32{1, 2}, 0)
	if !near(float64(normed[0]), 0.848528, 1e-5) || !near(float64(normed[1]), 2.262742, 1e-5) {
		t.Errorf("rmsNorm = %v", normed)
	}
	rmsNormPlain(values, 0)
	if !near(float64(values[0]), 0.848528, 1e-5) || !near(float64(values[1]), 1.131371, 1e-5) {
		t.Errorf("rmsNormPlain = %v", values)
	}
	const angle = 0.5
	vector := []float32{1, 2, 3, 4}
	applyRope(vector, []float32{float32(math.Cos(angle)), 1}, []float32{float32(math.Sin(angle)), 0})
	// 次元 0 と 2 が角度 0.5 で回転し、次元 1 と 3 は変わらない。
	wantFirst := math.Cos(angle) - 3*math.Sin(angle)
	wantThird := 3*math.Cos(angle) + math.Sin(angle)
	if !near(float64(vector[0]), wantFirst, 1e-5) || !near(float64(vector[2]), wantThird, 1e-5) || vector[1] != 2 || vector[3] != 4 {
		t.Errorf("applyRope = %v", vector)
	}
}

// TestConfigValidate は、構成の確認（層のヘッド数など）を確かめます。
func TestConfigValidate(t *testing.T) {
	config := Config{Architecture: Architecture, HiddenSize: 3840, FeedForwardSize: 15360, NumHeads: 16, VocabSize: 262144,
		SlidingWindow: 1024, Layers: []LayerConfig{{Sliding: true, KVHeads: 8, HeadDim: 256}, {KVHeads: 1, HeadDim: 512}}}
	if err := config.Validate(); err != nil {
		t.Fatal(err)
	}
	if config.QueryWidth(1) != 8192 || config.KVWidth(0) != 2048 || !near(float64(config.EmbeddingScale()), 61.967734, 1e-4) {
		t.Errorf("widths %d %d, scale %g", config.QueryWidth(1), config.KVWidth(0), config.EmbeddingScale())
	}
	config.Layers[0].KVHeads = 3
	if err := config.Validate(); err == nil {
		t.Error("Validate accepted 3 kv heads for 16 heads")
	}
}
