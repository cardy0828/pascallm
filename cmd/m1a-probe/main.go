// Command m1a-probe は、M1a（cgo を使わない Go から CUDA を呼ぶ試作）の動作を確かめます。
//
// 確認すること:
//  1. purego で libcuda.so.1 を読み込み、UUID で指定した GPU を選べる
//  2. go:embed で埋め込んだ sm_61 の cubin を読み込み、カーネルを起動して正しい結果が返る
//  3. ページ固定のホストメモリへ結果をコピーできる
//  4. CUDA Graph を記録・再生でき、再生回数どおりに処理が実行される
//  5. 通常起動と Graph 再生の、カーネル 1 個あたりの時間
//
// 結果は 1 行 1 件の JSON で標準出力に書きます。いずれかの確認に失敗すると終了コード 1 で終わります。
//
// ビルド例: CGO_ENABLED=0 go build -o bin/m1a-probe ./cmd/m1a-probe
// 実行例:   ./bin/m1a-probe -device-uuid GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math/rand/v2"
	"os"
	"runtime"
	"time"

	"github.com/cardy0828/pascallm/internal/cuda"
	"github.com/cardy0828/pascallm/internal/kernels"
)

// defaultDeviceUUID は空なら PASCA_DEVICE_UUID を使用します。GPUは明示的に選択します。
const defaultDeviceUUID = ""

// threadsPerBlock は、確認用カーネルの 1 ブロックあたりのスレッド数です。
const threadsPerBlock = 256

// options はコマンドライン引数です。
type options struct {
	deviceUUID      string
	kernelsPerToken int
	measuredTokens  int
	warmupTokens    int
}

// parseOptions はコマンドライン引数を読みます。
// 例: -kernels-per-token 600 -tokens 200 -warmup-tokens 20
func parseOptions() options {
	var opts options
	flag.StringVar(&opts.deviceUUID, "device-uuid", defaultDeviceUUID, "使う GPU の UUID（nvidia-smi -L の表記）")
	flag.IntVar(&opts.kernelsPerToken, "kernels-per-token", 600, "1 トークン分として起動する小さなカーネルの数")
	flag.IntVar(&opts.measuredTokens, "tokens", 200, "計時するトークン数")
	flag.IntVar(&opts.warmupTokens, "warmup-tokens", 20, "計時前に捨てるトークン数（クロック上昇のため）")
	flag.Parse()
	return opts
}

// emit は 1 件の結果を JSON の 1 行として標準出力に書きます。
func emit(record map[string]any) {
	line, err := json.Marshal(record)
	if err != nil {
		fmt.Fprintf(os.Stderr, "json encode error: %v\n", err)
		return
	}
	fmt.Println(string(line))
}

// blocksFor は、n 要素を threadsPerBlock ずつ処理するのに必要なブロック数を返します。
// 例: n = 1000 -> 4（256 × 4 = 1024 ≥ 1000）
func blocksFor(n int) uint32 {
	return uint32((n + threadsPerBlock - 1) / threadsPerBlock)
}

// probeKernels は、確認に使うカーネル一式です。
type probeKernels struct {
	axpy      *cuda.Function
	dot4      *cuda.Function
	increment *cuda.Function
}

// loadProbeKernels は、埋め込んだ cubin からカーネルを取り出します。
func loadProbeKernels(module *cuda.Module) (*probeKernels, error) {
	axpy, err := module.Function("axpy_f32")
	if err != nil {
		return nil, err
	}
	dot4, err := module.Function("dot4_i8")
	if err != nil {
		return nil, err
	}
	increment, err := module.Function("increment_f32")
	if err != nil {
		return nil, err
	}
	return &probeKernels{axpy: axpy, dot4: dot4, increment: increment}, nil
}

// reportDevice は選んだ GPU の名前・UUID・compute capability を出力し、sm_61 であることを確かめます。
func reportDevice(device cuda.Device) error {
	name, err := device.Name()
	if err != nil {
		return err
	}
	uuid, err := device.UUID()
	if err != nil {
		return err
	}
	major, minor, err := device.ComputeCapability()
	if err != nil {
		return err
	}
	version, err := cuda.DriverVersion()
	if err != nil {
		return err
	}
	count, err := cuda.DeviceCount()
	if err != nil {
		return err
	}
	emit(map[string]any{"kind": "device", "name": name, "uuid": uuid, "cc": fmt.Sprintf("%d.%d", major, minor),
		"driver_cuda_version": version, "visible_devices": count, "cgo": cgoEnabled()})
	if major != 6 || minor != 1 {
		return fmt.Errorf("device %s is compute capability %d.%d, but the cubin is built for 6.1", uuid, major, minor)
	}
	return nil
}

// checkAxpy は axpy_f32 を 1,048,576 要素で実行し、全要素が CPU の計算と完全に一致するか確かめます。
// x は 0〜999 の整数、y は 1、a は 2 なので、結果は float32 で誤差なく表せます。
func checkAxpy(stream *cuda.Stream, kernels *probeKernels) error {
	const count = 1 << 20
	const scale float32 = 2
	x := make([]float32, count)
	y := make([]float32, count)
	for index := range x {
		x[index] = float32(index % 1000)
		y[index] = 1
	}
	deviceX, err := cuda.Alloc(count * 4)
	if err != nil {
		return err
	}
	defer cuda.Free(deviceX)
	deviceY, err := cuda.Alloc(count * 4)
	if err != nil {
		return err
	}
	defer cuda.Free(deviceY)
	if err := cuda.CopyToDevice(stream, deviceX, x); err != nil {
		return err
	}
	if err := cuda.CopyToDevice(stream, deviceY, y); err != nil {
		return err
	}
	args := new(cuda.KernelArgs).Pointer(deviceY).Pointer(deviceX).Float32(scale).Int32(count)
	grid := cuda.Dim3{X: blocksFor(count), Y: 1, Z: 1}
	block := cuda.Dim3{X: threadsPerBlock, Y: 1, Z: 1}
	if err := kernels.axpy.Launch(grid, block, 0, stream, args); err != nil {
		return err
	}
	if err := stream.Synchronize(); err != nil {
		return err
	}
	result := make([]float32, count)
	if err := cuda.CopyFromDevice(stream, result, deviceY); err != nil {
		return err
	}
	mismatches := 0
	for index := range result {
		expected := y[index] + scale*x[index]
		if result[index] != expected {
			mismatches++
		}
	}
	emit(map[string]any{"kind": "check", "name": "axpy_f32", "elements": count, "mismatches": mismatches,
		"ok": mismatches == 0})
	if mismatches != 0 {
		return fmt.Errorf("axpy_f32: %d of %d elements differ", mismatches, count)
	}
	return nil
}

// dot4Reference は、int8 を 4 個詰めた a と b の積和を CPU で計算します（DP4A と同じ定義）。
// 例: dot4Reference(0x01020304, 0x01010101) = 10
func dot4Reference(a, b int32) int32 {
	var sum int32
	for lane := 0; lane < 4; lane++ {
		shift := uint(8 * lane)
		sum += int32(int8(a>>shift)) * int32(int8(b>>shift))
	}
	return sum
}

// checkDot4 は dot4_i8 を 65,536 要素で実行し、結果をページ固定メモリへコピーして CPU と比べます。
func checkDot4(stream *cuda.Stream, kernels *probeKernels) error {
	const count = 1 << 16
	random := rand.New(rand.NewPCG(20261114, 1))
	a := make([]int32, count)
	b := make([]int32, count)
	for index := range a {
		a[index] = int32(random.Uint32())
		b[index] = int32(random.Uint32())
	}
	deviceA, err := cuda.Alloc(count * 4)
	if err != nil {
		return err
	}
	defer cuda.Free(deviceA)
	deviceB, err := cuda.Alloc(count * 4)
	if err != nil {
		return err
	}
	defer cuda.Free(deviceB)
	deviceOut, err := cuda.Alloc(count * 4)
	if err != nil {
		return err
	}
	defer cuda.Free(deviceOut)
	if err := cuda.CopyToDevice(stream, deviceA, a); err != nil {
		return err
	}
	if err := cuda.CopyToDevice(stream, deviceB, b); err != nil {
		return err
	}
	args := new(cuda.KernelArgs).Pointer(deviceA).Pointer(deviceB).Pointer(deviceOut).Int32(count)
	grid := cuda.Dim3{X: blocksFor(count), Y: 1, Z: 1}
	block := cuda.Dim3{X: threadsPerBlock, Y: 1, Z: 1}
	if err := kernels.dot4.Launch(grid, block, 0, stream, args); err != nil {
		return err
	}
	if err := stream.Synchronize(); err != nil {
		return err
	}
	pinned, err := cuda.AllocHost(count * 4)
	if err != nil {
		return err
	}
	defer pinned.Close()
	if err := pinned.CopyFromDevice(stream, deviceOut); err != nil {
		return err
	}
	mismatches := 0
	for index, value := range pinned.Int32s() {
		if value != dot4Reference(a[index], b[index]) {
			mismatches++
		}
	}
	emit(map[string]any{"kind": "check", "name": "dot4_i8_pinned", "elements": count, "mismatches": mismatches,
		"ok": mismatches == 0})
	if mismatches != 0 {
		return fmt.Errorf("dot4_i8: %d of %d elements differ", mismatches, count)
	}
	return nil
}

// launchTimer は、ストリーム上の 2 つのイベントで GPU 側の経過時間を測る道具です。
type launchTimer struct {
	start *cuda.Event
	stop  *cuda.Event
}

// newLaunchTimer は計時用のイベントを 2 個作ります。
func newLaunchTimer() (*launchTimer, error) {
	start, err := cuda.NewEvent()
	if err != nil {
		return nil, err
	}
	stop, err := cuda.NewEvent()
	if err != nil {
		start.Close()
		return nil, err
	}
	return &launchTimer{start: start, stop: stop}, nil
}

// close は計時用のイベントを破棄します。
func (t *launchTimer) close() {
	t.start.Close()
	t.stop.Close()
}

// measure は、runToken を tokens 回呼んだときの GPU 側の経過ミリ秒と、ホスト側の経過時間を返します。
// runToken は 1 トークン分の処理を積んで、完了まで待つ関数です。
func (t *launchTimer) measure(stream *cuda.Stream, tokens int, runToken func() error) (float64, time.Duration, error) {
	if err := t.start.Record(stream); err != nil {
		return 0, 0, err
	}
	wallStart := time.Now()
	for token := 0; token < tokens; token++ {
		if err := runToken(); err != nil {
			return 0, 0, err
		}
	}
	wall := time.Since(wallStart)
	if err := t.stop.Record(stream); err != nil {
		return 0, 0, err
	}
	if err := t.stop.Synchronize(); err != nil {
		return 0, 0, err
	}
	gpuMilliseconds, err := cuda.ElapsedMilliseconds(t.start, t.stop)
	return gpuMilliseconds, wall, err
}

// checkGraph は、小さなカーネルを 1 トークンあたり kernelsPerToken 個起動する処理を、
// 通常起動と CUDA Graph 再生の両方で実行し、時間と結果の正しさを確かめます。
// data の各要素は、起動 1 回につき 1 増えるので、最後に総起動回数と一致するはずです。
func checkGraph(stream *cuda.Stream, kernels *probeKernels, opts options) error {
	const count = 2048 // LFM3 の hidden 次元と同じ大きさ
	data, err := cuda.Alloc(count * 4)
	if err != nil {
		return err
	}
	defer cuda.Free(data)
	if err := cuda.MemsetBytes(stream, data, 0, count*4); err != nil {
		return err
	}
	args := new(cuda.KernelArgs).Pointer(data).Int32(count)
	grid := cuda.Dim3{X: blocksFor(count), Y: 1, Z: 1}
	block := cuda.Dim3{X: threadsPerBlock, Y: 1, Z: 1}

	// 1 トークン分のカーネルを stream に積む（完了は待たない）。
	enqueueToken := func() error {
		for kernel := 0; kernel < opts.kernelsPerToken; kernel++ {
			if err := kernels.increment.Launch(grid, block, 0, stream, args); err != nil {
				return err
			}
		}
		return nil
	}
	// 通常起動で 1 トークン分を積み、完了を待つ（生成ではサンプリング結果を待つため毎トークン同期する）。
	streamToken := func() error {
		if err := enqueueToken(); err != nil {
			return err
		}
		return stream.Synchronize()
	}

	timer, err := newLaunchTimer()
	if err != nil {
		return err
	}
	defer timer.close()

	for token := 0; token < opts.warmupTokens; token++ {
		if err := streamToken(); err != nil {
			return err
		}
	}
	streamGPU, streamWall, err := timer.measure(stream, opts.measuredTokens, streamToken)
	if err != nil {
		return err
	}

	// ホスト側で 1 トークン分を積むだけの時間（purego 経由の cuLaunchKernel の CPU コスト）。
	enqueueStart := time.Now()
	if err := enqueueToken(); err != nil {
		return err
	}
	enqueueWall := time.Since(enqueueStart)
	if err := stream.Synchronize(); err != nil {
		return err
	}

	// 1 トークン分を CUDA Graph に記録する。記録中のカーネルは実行されない。
	if err := stream.BeginCapture(); err != nil {
		return err
	}
	if err := enqueueToken(); err != nil {
		stream.EndCapture()
		return err
	}
	graph, err := stream.EndCapture()
	if err != nil {
		return err
	}
	defer graph.Close()
	executable, err := graph.Instantiate()
	if err != nil {
		return err
	}
	defer executable.Close()

	graphToken := func() error {
		if err := executable.Launch(stream); err != nil {
			return err
		}
		return stream.Synchronize()
	}
	for token := 0; token < opts.warmupTokens; token++ {
		if err := graphToken(); err != nil {
			return err
		}
	}
	graphGPU, graphWall, err := timer.measure(stream, opts.measuredTokens, graphToken)
	if err != nil {
		return err
	}

	// 結果の確認: 通常起動（捨て + 計時 + 積むだけの 1 回）と Graph 再生（捨て + 計時）の合計回数。
	tokensRun := (opts.warmupTokens + opts.measuredTokens + 1) + (opts.warmupTokens + opts.measuredTokens)
	expected := float32(tokensRun * opts.kernelsPerToken)
	result := make([]float32, count)
	if err := cuda.CopyFromDevice(stream, result, data); err != nil {
		return err
	}
	mismatches := 0
	for _, value := range result {
		if value != expected {
			mismatches++
		}
	}
	kernelsMeasured := float64(opts.measuredTokens * opts.kernelsPerToken)
	emit(map[string]any{
		"kind":                         "launch",
		"kernels_per_token":            opts.kernelsPerToken,
		"tokens":                       opts.measuredTokens,
		"stream_gpu_us_per_kernel":     streamGPU * 1e3 / kernelsMeasured,
		"stream_wall_ms_per_token":     float64(streamWall.Microseconds()) / 1e3 / float64(opts.measuredTokens),
		"host_enqueue_us_per_kernel":   float64(enqueueWall.Nanoseconds()) / 1e3 / float64(opts.kernelsPerToken),
		"graph_gpu_us_per_kernel":      graphGPU * 1e3 / kernelsMeasured,
		"graph_wall_ms_per_token":      float64(graphWall.Microseconds()) / 1e3 / float64(opts.measuredTokens),
		"expected_increment_per_value": expected,
		"mismatches":                   mismatches,
		"ok":                           mismatches == 0,
	})
	if mismatches != 0 {
		return fmt.Errorf("graph replay: %d of %d values differ from %.0f", mismatches, count, expected)
	}
	return nil
}

// run は確認を順番に実行します。最初に失敗した確認のエラーを返します。
func run(opts options) error {
	// CUDA のコンテキストは OS スレッドに結び付くため、このゴルーチンを 1 つのスレッドに固定する。
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if err := cuda.Init(); err != nil {
		return err
	}
	device, err := cuda.FindDeviceByUUID(opts.deviceUUID)
	if err != nil {
		return err
	}
	if err := reportDevice(device); err != nil {
		return err
	}
	context, err := cuda.NewPrimaryContext(device)
	if err != nil {
		return err
	}
	defer context.Close()

	module, err := cuda.LoadModule(kernels.ProbeSM61)
	if err != nil {
		return err
	}
	defer module.Close()
	probe, err := loadProbeKernels(module)
	if err != nil {
		return err
	}
	stream, err := cuda.NewStream()
	if err != nil {
		return err
	}
	defer stream.Close()

	return errors.Join(
		checkAxpy(stream, probe),
		checkDot4(stream, probe),
		checkGraph(stream, probe, opts),
	)
}

func main() {
	opts := parseOptions()
	if err := run(opts); err != nil {
		emit(map[string]any{"kind": "result", "ok": false, "error": err.Error()})
		os.Exit(1)
	}
	emit(map[string]any{"kind": "result", "ok": true})
}
