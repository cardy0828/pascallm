// Package gpucore は、GPU 実装（モデルごとのパッケージ）が共通に使う部品です（M7）。
//
//   - Memory: GPU メモリの確保の記録と、ホストからの転送
//   - Launcher: カーネルを順に積む間の、最初のエラーの記録
//   - Matrix・UploadMatrices: GPU 上の PQ4 行列（kernels の gemv_pq4・gemm_pq4 が読む配置）
//   - CandidateParams・Candidates: 選び方の候補を GPU で絞る処理の設定と結果
//
// 最初の利用者は internal/gemma4gpu です。internal/lfm3gpu は同じ内容の部品を自分の中に持っており（M2〜M4 の実装）、
// このパッケージへの載せ替えは後の整理とします（生成部 internal/generate を両方のモデルで使える形にするときに行う）。
package gpucore

import (
	"fmt"
	"unsafe"

	"github.com/cardy0828/pascallm/internal/cuda"
	"github.com/cardy0828/pascallm/internal/pq4"
)

// カーネルの起動の大きさに関する定数です（kernels/lfm3.cu の gemv_pq4・gemm_pq4 の前提と合わせる）。
const (
	WarpSize          = 32
	GemvWarpsPerBlock = 4  // gemv_pq4 の 1 スレッドブロックのワープ数
	GemvRowsPerWarp   = 4  // gemv_pq4 の 1 ワープが担当する行数（行列の行数はこの倍数であること）
	GemmTileRows      = 64 // gemm_pq4 の 1 ブロックが受け持つ重みの行数（行列の行数はこの倍数であること）
	GemmTileTokens    = 128
	GemmThreads       = 256
	GemmStageBlocks   = 4 // gemm_pq4 が K 方向に 1 回で読むブロック数（1 行のブロック数はこの倍数であること）
	// ActivationInfoBytes は、int8 化の付帯情報 ActBlockInfo（float32 × 2）のバイト数です。
	ActivationInfoBytes = 8
)

// stagingBytes は、ホストから GPU へ重みを送るときに使う固定メモリの大きさです。
// 重みは .pq4m を mmap した領域にあり、GC の管理外なので、固定メモリに少しずつ写してから転送します。
const stagingBytes = 64 << 20

// Memory は、GPU メモリの確保をまとめて記録し、Close で一括して解放します。
type Memory struct {
	stream      *cuda.Stream
	staging     *cuda.HostBuffer
	allocations []cuda.DevicePtr
	totalBytes  int64
}

// NewMemory は、転送用の固定メモリを確保して Memory を作ります。
func NewMemory(stream *cuda.Stream) (*Memory, error) {
	staging, err := cuda.AllocHost(stagingBytes)
	if err != nil {
		return nil, err
	}
	return &Memory{stream: stream, staging: staging}, nil
}

// TotalBytes は、確保した GPU メモリの合計を返します（報告用）。
func (m *Memory) TotalBytes() int64 { return m.totalBytes }

// Alloc は GPU メモリを size バイト確保し、記録します。例: Alloc(4 * 3840) は float32 3840 個分
func (m *Memory) Alloc(size int) (cuda.DevicePtr, error) {
	if size <= 0 {
		return 0, fmt.Errorf("gpucore: cannot allocate %d bytes", size)
	}
	pointer, err := cuda.Alloc(size)
	if err != nil {
		return 0, fmt.Errorf("gpucore: allocating %d bytes: %w", size, err)
	}
	m.allocations = append(m.allocations, pointer)
	m.totalBytes += int64(size)
	return pointer, nil
}

// AllocZeroed は size バイトを確保し、0 で埋めます（状態・キャッシュ用）。
func (m *Memory) AllocZeroed(size int) (cuda.DevicePtr, error) {
	pointer, err := m.Alloc(size)
	if err != nil {
		return 0, err
	}
	if err := cuda.MemsetBytes(m.stream, pointer, 0, size); err != nil {
		return 0, err
	}
	return pointer, nil
}

// CopyBytes は、ホストのバイト列 data を GPU の destination へコピーし、完了まで待ちます。
func (m *Memory) CopyBytes(destination cuda.DevicePtr, data []byte) error {
	buffer := m.staging.Bytes()
	for offset := 0; offset < len(data); offset += len(buffer) {
		chunk := copy(buffer, data[offset:])
		if err := m.staging.CopyPrefixToDeviceAsync(m.stream, destination+cuda.DevicePtr(offset), chunk); err != nil {
			return err
		}
		// 次の塊で固定メモリを上書きする前に、転送の完了を待つ。
		if err := m.stream.Synchronize(); err != nil {
			return err
		}
	}
	return nil
}

// Close は、確保したすべての GPU メモリと転送用の固定メモリを解放します。最初のエラーを返します。
func (m *Memory) Close() error {
	var first error
	for _, pointer := range m.allocations {
		if err := cuda.Free(pointer); err != nil && first == nil {
			first = err
		}
	}
	m.allocations = nil
	if m.staging != nil {
		if err := m.staging.Close(); err != nil && first == nil {
			first = err
		}
		m.staging = nil
	}
	return first
}

// AsBytes は、ポインターを含まない型のスライスを、同じメモリのバイト列として見せます（コピーしない）。
// 例: AsBytes([]float32{1, 2}) は 8 バイト
func AsBytes[T any](values []T) []byte {
	if len(values) == 0 {
		return nil
	}
	var zero T
	return unsafe.Slice((*byte)(unsafe.Pointer(&values[0])), len(values)*int(unsafe.Sizeof(zero)))
}

// Upload は、スライスと同じ大きさの GPU メモリを確保して内容をコピーし、そのアドレスを返します。
// 例: Upload(memory, normWeights) は float32 の正規化の重みを GPU に置く
func Upload[T any](m *Memory, values []T) (cuda.DevicePtr, error) {
	data := AsBytes(values)
	pointer, err := m.Alloc(len(data))
	if err != nil {
		return 0, err
	}
	if err := m.CopyBytes(pointer, data); err != nil {
		return 0, err
	}
	return pointer, nil
}

// Launcher は、カーネルを順に積む間の最初のエラーだけを覚えておく道具です。
// エラーの後の起動は何もしないので、呼び出し側は最後に Err を 1 回確かめれば済みます。
// Timer が nil でなければ、起動ごとに前後のイベントを置いて、名前ごとの GPU 時間を測ります（計測用）。
type Launcher struct {
	Stream *cuda.Stream
	Timer  *KernelTimer
	Err    error
}

// Launch は function を grid × block で積みます。計測の名前はカーネル名です。
// 例: l.Launch(kernels.embed, Dim1(15), Dim1(256), args)
func (l *Launcher) Launch(function *cuda.Function, grid, block cuda.Dim3, args *cuda.KernelArgs) {
	l.LaunchLabeled(function.Name(), function, grid, block, args)
}

// LaunchLabeled は Launch と同じですが、計測の名前を label にします（GEMV を形ごとに分けるため）。
func (l *Launcher) LaunchLabeled(label string, function *cuda.Function, grid, block cuda.Dim3, args *cuda.KernelArgs) {
	if l.Err != nil {
		return
	}
	if l.Timer != nil {
		l.Err = l.Timer.measure(label, l.Stream, func() error { return function.Launch(grid, block, 0, l.Stream, args) })
		return
	}
	l.Err = function.Launch(grid, block, 0, l.Stream, args)
}

// KernelTimer は、カーネルの起動ごとに前後のイベントを置き、名前ごとの GPU 時間を集めます（計測用）。
type KernelTimer struct {
	labels []string
	events []*cuda.Event // 起動 i の前が events[2i]、後が events[2i+1]
}

// measure は、launch（カーネル 1 個の起動）の前後にイベントを置きます。
func (t *KernelTimer) measure(label string, stream *cuda.Stream, launch func() error) error {
	pair := make([]*cuda.Event, 2)
	for index := range pair {
		event, err := cuda.NewEvent()
		if err != nil {
			return err
		}
		t.events = append(t.events, event)
		pair[index] = event
	}
	t.labels = append(t.labels, label)
	if err := pair[0].Record(stream); err != nil {
		return err
	}
	if err := launch(); err != nil {
		return err
	}
	return pair[1].Record(stream)
}

// Totals は、名前ごとの GPU 時間の合計（マイクロ秒）と起動回数を返します。ストリームの完了を待った後に呼びます。
func (t *KernelTimer) Totals() (microseconds map[string]float64, launches map[string]int, err error) {
	const microsecondsPerMillisecond = 1000
	microseconds = make(map[string]float64)
	launches = make(map[string]int)
	for index, label := range t.labels {
		milliseconds, err := cuda.ElapsedMilliseconds(t.events[2*index], t.events[2*index+1])
		if err != nil {
			return nil, nil, err
		}
		microseconds[label] += milliseconds * microsecondsPerMillisecond
		launches[label]++
	}
	return microseconds, launches, nil
}

// Close はイベントをすべて破棄します。
func (t *KernelTimer) Close() {
	for _, event := range t.events {
		event.Close()
	}
	t.events = nil
	t.labels = nil
}

// Dim1 は 1 次元の大きさ（x だけ）を返します。例: Dim1(256) = {256, 1, 1}
func Dim1(x int) cuda.Dim3 {
	return cuda.Dim3{X: uint32(x), Y: 1, Z: 1}
}

// Dim2 は 2 次元の大きさ（x と y）を返します。例: Dim2(8, 16) = {8, 16, 1}
func Dim2(x, y int) cuda.Dim3 {
	return cuda.Dim3{X: uint32(x), Y: uint32(y), Z: 1}
}

// CeilDiv は value / divisor の切り上げです。例: CeilDiv(10, 4) = 3
func CeilDiv(value, divisor int) int {
	return (value + divisor - 1) / divisor
}

// Matrix は GPU 上の PQ4・PQ8 行列です（4bit 本体と FP16 係数の分離配置、CPU 版と同じ並び）。
type Matrix struct {
	Packed cuda.DevicePtr // [rows][重みのブロック][16 バイト]
	Scales cuda.DevicePtr // [rows][重みのブロック] の FP16
	Rows   int
	Cols   int
	// WeightBlocks は 1 行あたりの重みのブロック数（PQ4 は cols/32、PQ8 は 2 × cols/32）です。
	WeightBlocks int
	// InputBlockShift は、カーネルで活性値のブロック番号を「重みのブロック番号 >> shift」とする値（PQ4 は 0、PQ8 は 1）です。
	InputBlockShift int
}

// Label は、計測用の名前を返します。例: "gemv_pq4 30720x3840"
func (m Matrix) Label(kernel string) string {
	return fmt.Sprintf("%s %dx%d", kernel, m.Rows, m.Cols)
}

// UploadMatrices は、入力次元が同じ PQ4 行列を、行方向につないだ 1 つの行列として GPU に置きます。
// 1 回の GEMV で複数の行列を計算するためです（出力は渡した順に並ぶ）。
// つないだ行列が gemv_pq4（行数が 4 の倍数）と gemm_pq4（行数が 64 の倍数、1 行のブロック数が 4 の倍数）の
// 前提を満たすかも確かめます。
// 例: UploadMatrices(memory, "ffn gate_up", gate, up) -> 出力の前半が Wgate·x、後半が Wup·x
func UploadMatrices(memory *Memory, name string, parts ...*pq4.Matrix) (Matrix, error) {
	cols := parts[0].Cols()
	shift := parts[0].InputBlockShift()
	var rows, packedBytes, scaleCount int
	for _, part := range parts {
		if part.Cols() != cols {
			return Matrix{}, fmt.Errorf("gpucore: %s: cannot join matrices with %d and %d columns", name, cols, part.Cols())
		}
		if part.InputBlockShift() != shift {
			return Matrix{}, fmt.Errorf("gpucore: %s: cannot join PQ4 and PQ8 matrices", name)
		}
		rows += part.Rows()
		packedBytes += len(part.Packed())
		scaleCount += len(part.Scales())
	}
	blocks := parts[0].BlocksPerRow()
	if rows%GemmTileRows != 0 || rows%GemvRowsPerWarp != 0 || blocks%GemmStageBlocks != 0 {
		return Matrix{}, fmt.Errorf("gpucore: %s (%d × %d) does not fit the kernels (rows multiple of %d, blocks per row multiple of %d)",
			name, rows, cols, GemmTileRows, GemmStageBlocks)
	}
	packed, err := memory.Alloc(packedBytes)
	if err != nil {
		return Matrix{}, err
	}
	const scaleBytes = 2 // FP16
	scales, err := memory.Alloc(scaleBytes * scaleCount)
	if err != nil {
		return Matrix{}, err
	}
	packedOffset, scaleOffset := 0, 0
	for _, part := range parts {
		if err := memory.CopyBytes(packed+cuda.DevicePtr(packedOffset), part.Packed()); err != nil {
			return Matrix{}, err
		}
		if err := memory.CopyBytes(scales+cuda.DevicePtr(scaleOffset), AsBytes(part.Scales())); err != nil {
			return Matrix{}, err
		}
		packedOffset += len(part.Packed())
		scaleOffset += scaleBytes * len(part.Scales())
	}
	return Matrix{Packed: packed, Scales: scales, Rows: rows, Cols: cols, WeightBlocks: blocks, InputBlockShift: shift}, nil
}

// Gemv は out = matrix · (int8 化済みの活性値 actQ・actInfo) を積みます（kernels の gemv_pq4）。
func Gemv(l *Launcher, kernel *cuda.Function, matrix Matrix, actQ, actInfo, out cuda.DevicePtr) {
	args := new(cuda.KernelArgs).
		Pointer(matrix.Packed).Pointer(matrix.Scales).Pointer(actQ).Pointer(actInfo).
		Pointer(out).Int32(int32(matrix.Rows)).Int32(int32(matrix.WeightBlocks)).Int32(int32(matrix.InputBlockShift))
	grid := Dim1(CeilDiv(matrix.Rows, GemvWarpsPerBlock*GemvRowsPerWarp))
	block := cuda.Dim3{X: WarpSize, Y: GemvWarpsPerBlock, Z: 1}
	l.LaunchLabeled(matrix.Label("gemv_pq4"), kernel, grid, block, args)
}

// Gemm は、out[t] = matrix · (トークン t の int8 化済みの活性値) を count トークン分積みます（kernels の gemm_pq4）。
func Gemm(l *Launcher, kernel *cuda.Function, matrix Matrix, actQ, actInfo, out cuda.DevicePtr, count int) {
	args := new(cuda.KernelArgs).
		Pointer(matrix.Packed).Pointer(matrix.Scales).Pointer(actQ).Pointer(actInfo).
		Pointer(out).Int32(int32(matrix.Rows)).Int32(int32(matrix.WeightBlocks)).Int32(int32(count)).
		Int32(int32(matrix.InputBlockShift))
	grid := Dim2(matrix.Rows/GemmTileRows, CeilDiv(count, GemmTileTokens))
	l.LaunchLabeled(matrix.Label("gemm_pq4"), kernel, grid, Dim1(GemmThreads), args)
}

// 選び方の候補を GPU で絞る処理の寸法です（kernels/lfm3.cu の kCandidate* と合わせる）。
const (
	// CandidateCount は、候補を絞る処理が返す候補の数です。
	CandidateCount = 128
	// CandidateParts は、1 段目のスレッドブロック数（語彙を分ける区間の数）です。
	CandidateParts = 64
	// CandidateFinalThreads は、2 段目（candidates_final）のスレッド数です。
	CandidateFinalThreads = 1024
	candidateStatCount    = 2 // 2 段目が書く統計の数（全体の最大値、exp の合計）
	// CandidateResultFloats は、2 段目の結果 [値 × 128][番号 × 128][統計 × 2] の 4 バイト単位の数です。
	CandidateResultFloats = 2*CandidateCount + candidateStatCount
)

// CandidateParams は、候補を絞るときに GPU で適用する設定です。
type CandidateParams struct {
	ValidTokens        int     // 選んでよい番号の上限（含まない）。トークナイザーの番号の数。例: 262144
	InverseTemperature float32 // 1 / temperature（exp の合計に使う）。temperature 0 のときは 1。例: 1
	RepetitionPenalty  float32 // 1 ならなし。例: 1.1
	PresencePenalty    float32 // 0 ならなし
	FrequencyPenalty   float32 // 0 ならなし
}

// Candidates は、GPU で絞った候補です。値はペナルティを適用した logits です（temperature は掛けていない）。
// スライスは固定メモリを指し、次の呼び出しで上書きされます。
type Candidates struct {
	IDs      []int32   // 長さ CandidateCount。値の大きい順（同じ値なら番号の小さい順）
	Logits   []float32 // IDs と同じ順の値
	MaxLogit float32   // 番号 [0, ValidTokens) 全体の最大値
	// ExpSum は、全体の Σ exp((値 - MaxLogit) × InverseTemperature) です（確率の分母）。
	ExpSum float32
}

// CandidatesFromResult は、2 段目の結果を受け取った固定メモリを Candidates として見せます（コピーしない）。
func CandidatesFromResult(host *cuda.HostBuffer) Candidates {
	values := host.Float32s()
	return Candidates{
		Logits:   values[:CandidateCount],
		IDs:      host.Int32s()[CandidateCount : 2*CandidateCount],
		MaxLogit: values[2*CandidateCount],
		ExpSum:   values[2*CandidateCount+1],
	}
}
