package gemma4gpu

import (
	"math"
	"os"
	"runtime"
	"testing"

	"github.com/cardy0828/pascallm/internal/cuda"
	"github.com/cardy0828/pascallm/internal/gpucore"
	"github.com/cardy0828/pascallm/internal/numeric"
)

// TestKVAttentionGPU は固定 K/V に対する attention を独立した float64 の密な softmax と比較します。
// PASCA_KV_GPU_TEST=1 のとき GTX 1080 Ti だけを使用。局所の輪状折り返し、端数タイル、全文脈を検査します。
func TestKVAttentionGPU(t *testing.T) {
	if os.Getenv("PASCA_KV_GPU_TEST") != "1" {
		t.Skip("GPU validation is opt-in")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := cuda.Init(); err != nil {
		t.Fatal(err)
	}
	device, err := cuda.FindDeviceByUUID("")
	if err != nil {
		t.Fatal(err)
	}
	context, err := cuda.NewPrimaryContext(device)
	if err != nil {
		t.Fatal(err)
	}
	defer context.Close()
	stream, err := cuda.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	for _, format := range []string{"fp16", "k-fp32", "fp32"} {
		storage, err := resolveKVStorage(format)
		if err != nil {
			t.Fatal(err)
		}
		module, err := cuda.LoadModule(storage.cubin)
		if err != nil {
			t.Fatal(err)
		}
		for _, c := range []struct {
			name                     string
			dim, heads, ring, window int
		}{{"local", 256, 2, 32, 19}, {"global", 512, 4, 64, 0}} {
			t.Logf("format=%s attention=%s", format, c.name)
			// t.Run は別 goroutine のため使わず、CUDA context を持つ同じ OS thread で検査する。
			func() {
				const position = 36
				query := make([]float32, c.heads*c.dim)
				keys := make([]float32, c.ring*c.dim)
				values := make([]float32, len(keys))
				for i := range query {
					query[i] = float32(math.Sin(float64(i)*0.17) * 0.3)
				}
				for i := range keys {
					keys[i] = float32(math.Cos(float64(i)*0.13) * 0.7)
					values[i] = float32(math.Sin(float64(i) * 0.19))
				}
				// uploadKVTestArray は CPU 参照側にも保存精度を反映するため、keys/values を丸めた値へ置き換えます。
				q := uploadKVTestArray(t, stream, query, float32Bytes)
				defer cuda.Free(q)
				k := uploadKVTestArray(t, stream, keys, storage.keyBytes)
				defer cuda.Free(k)
				v := uploadKVTestArray(t, stream, values, storage.valueBytes)
				defer cuda.Free(v)
				out, err := cuda.Alloc(len(query) * float32Bytes)
				if err != nil {
					t.Fatal(err)
				}
				defer cuda.Free(out)
				function, err := module.Function("g4_attention_" + c.name + "_rows")
				if err != nil {
					t.Fatal(err)
				}
				args := new(cuda.KernelArgs).Pointer(q).Pointer(k).Pointer(v).Int32(position).Int32(int32(c.window)).Int32(int32(c.ring)).Int32(1).Int32(int32(c.heads)).Int32(int32(c.heads)).Float32(1).Pointer(out)
				if err = function.Launch(gpucore.Dim2(1, 1), gpucore.Dim1(256), 0, stream, args); err != nil {
					t.Fatal(err)
				}
				got := make([]float32, len(query))
				if err = cuda.CopyFromDevice(stream, got, out); err != nil {
					t.Fatal(err)
				}
				first := 0
				if c.window > 0 {
					first = position - c.window + 1
				}
				maximumError := 0.0
				for h := 0; h < c.heads; h++ {
					scores := make([]float64, position-first+1)
					maximum := math.Inf(-1)
					for p := first; p <= position; p++ {
						for d := 0; d < c.dim; d++ {
							scores[p-first] += float64(query[h*c.dim+d]) * float64(keys[(p%c.ring)*c.dim+d])
						}
						maximum = math.Max(maximum, scores[p-first])
					}
					total := 0.0
					for i := range scores {
						scores[i] = math.Exp(scores[i] - maximum)
						total += scores[i]
					}
					for d := 0; d < c.dim; d++ {
						want := 0.0
						for p := first; p <= position; p++ {
							want += scores[p-first] / total * float64(values[(p%c.ring)*c.dim+d])
						}
						maximumError = math.Max(maximumError, math.Abs(float64(got[h*c.dim+d])-want))
					}
				}
				t.Logf("max absolute error %.9g", maximumError)
				if math.IsNaN(maximumError) || math.IsInf(maximumError, 0) || maximumError > 0.00003 {
					t.Fatalf("attention differs from dense float64 reference: %g", maximumError)
				}
			}()
		}
		module.Close()
	}
}

// uploadKVTestArray は指定精度で GPU へ転送し、CPU 参照に使う値も同じ精度に丸めます。例: bytes=2 は FP16。
func uploadKVTestArray(t *testing.T, stream *cuda.Stream, values []float32, bytes int) cuda.DevicePtr {
	t.Helper()
	pointer, err := cuda.Alloc(len(values) * bytes)
	if err != nil {
		t.Fatal(err)
	}
	if bytes == float16Bytes {
		half := make([]uint16, len(values))
		for i, v := range values {
			half[i] = numeric.Float32ToFloat16(v)
			values[i] = numeric.Float16ToFloat32(half[i])
		}
		err = cuda.CopyToDevice(stream, pointer, half)
	} else {
		err = cuda.CopyToDevice(stream, pointer, values)
	}
	if err != nil {
		t.Fatal(err)
	}
	return pointer
}
