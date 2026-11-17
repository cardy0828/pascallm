package cuda

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"strings"
	"unsafe"
)

// resultSuccess は CUDA_SUCCESS（成功）を表す CUresult の値です。
const resultSuccess int32 = 0

// Error は、CUDA Driver API が成功以外の CUresult を返したことを表します。
// 例: Error{Call: "cuMemAlloc_v2", Code: 2, Name: "CUDA_ERROR_OUT_OF_MEMORY"}
type Error struct {
	Call string
	Code int32
	Name string
}

// Error は、呼び出した関数名と CUDA のエラー名をまとめた文字列を返します。
// 例: "cuda: cuMemAlloc_v2 failed: CUDA_ERROR_OUT_OF_MEMORY (2)"
func (e *Error) Error() string {
	return fmt.Sprintf("cuda: %s failed: %s (%d)", e.Call, e.Name, e.Code)
}

// check は CUresult を Go の error に変換します。成功なら nil を返します。
// call: 呼び出した Driver API の名前（例: "cuInit"）, code: 戻り値
func check(call string, code int32) error {
	if code == resultSuccess {
		return nil
	}
	return &Error{Call: call, Code: code, Name: errorName(code)}
}

// errorName は CUresult の名前（例: "CUDA_ERROR_INVALID_VALUE"）を返します。
// 名前を取得できない値の場合は "CUDA_ERROR_UNKNOWN_CODE" を返します。
func errorName(code int32) string {
	var name *byte
	if driver.cuGetErrorName(code, &name) != resultSuccess || name == nil {
		return "CUDA_ERROR_UNKNOWN_CODE"
	}
	return goStringFromC(name)
}

// goStringFromC は、NUL 終端の C 文字列を Go の文字列にコピーします。
// 長さの上限 limit を超えた場合はそこで打ち切ります（壊れた文字列で無限に読まないため）。
func goStringFromC(text *byte) string {
	const limit = 4096
	var builder strings.Builder
	for offset := 0; offset < limit; offset++ {
		value := *(*byte)(unsafe.Add(unsafe.Pointer(text), offset))
		if value == 0 {
			break
		}
		builder.WriteByte(value)
	}
	return builder.String()
}

// cString は、Go の文字列を NUL 終端のバイト列に変換します。
// 例: cString("axpy") -> []byte{'a','x','p','y',0}
func cString(text string) []byte {
	buffer := make([]byte, len(text)+1)
	copy(buffer, text)
	return buffer
}

// Init は libcuda.so.1 を読み込み、cuInit(0) を呼びます。最初に 1 回呼んでください。
func Init() error {
	if err := loadDriver(); err != nil {
		return err
	}
	return check("cuInit", driver.cuInit(0))
}

// DriverVersion は、ドライバーが対応する CUDA の版を返します。
// 例: 13000 は CUDA 13.0 を表す（1000 × 主版 + 10 × 副版）。
func DriverVersion() (int, error) {
	var version int32
	if err := check("cuDriverGetVersion", driver.cuDriverGetVersion(&version)); err != nil {
		return 0, err
	}
	return int(version), nil
}

// Device は CUDA のデバイス番号（CUdevice）です。番号の順序は nvidia-smi と一致しない
// ことがあるため、特定の GPU を選ぶときは UUID で探してください（FindDeviceByUUID）。
type Device int32

// DeviceCount は、このプロセスから見える GPU の数を返します。
func DeviceCount() (int, error) {
	var count int32
	if err := check("cuDeviceGetCount", driver.cuDeviceGetCount(&count)); err != nil {
		return 0, err
	}
	return int(count), nil
}

// DeviceByOrdinal は、0 から始まる番号でデバイスを取得します。
func DeviceByOrdinal(ordinal int) (Device, error) {
	var device int32
	if err := check("cuDeviceGet", driver.cuDeviceGet(&device, int32(ordinal))); err != nil {
		return 0, err
	}
	return Device(device), nil
}

// Name は GPU の製品名を返します。例: "NVIDIA GeForce GTX 1080 Ti"
func (d Device) Name() (string, error) {
	const bufferLength = 256
	buffer := make([]byte, bufferLength)
	if err := check("cuDeviceGetName", driver.cuDeviceGetName(&buffer[0], bufferLength, int32(d))); err != nil {
		return "", err
	}
	return goStringFromC(&buffer[0]), nil
}

// UUID は、nvidia-smi と同じ書式の UUID を返します。
// 例: "GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
func (d Device) UUID() (string, error) {
	var raw [16]byte
	if err := check("cuDeviceGetUuid_v2", driver.cuDeviceGetUuid(&raw[0], int32(d))); err != nil {
		return "", err
	}
	return fmt.Sprintf("GPU-%x-%x-%x-%x-%x", raw[0:4], raw[4:6], raw[6:8], raw[8:10], raw[10:16]), nil
}

// CUdevice_attribute のうち、このパッケージで使う値です。
const (
	attributeComputeCapabilityMajor int32 = 75
	attributeComputeCapabilityMinor int32 = 76
)

// ComputeCapability は compute capability を (主版, 副版) で返します。例: GTX 1080 Ti は (6, 1)
func (d Device) ComputeCapability() (int, int, error) {
	var major, minor int32
	if err := check("cuDeviceGetAttribute", driver.cuDeviceGetAttribute(&major, attributeComputeCapabilityMajor, int32(d))); err != nil {
		return 0, 0, err
	}
	if err := check("cuDeviceGetAttribute", driver.cuDeviceGetAttribute(&minor, attributeComputeCapabilityMinor, int32(d))); err != nil {
		return 0, 0, err
	}
	return int(major), int(minor), nil
}

// FindDeviceByUUID は、見えている GPU の中から UUID が一致するものを返します。
// uuid の例: "GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"（大文字・小文字は区別しない）。空ならPASCA_DEVICE_UUIDを使います。
func FindDeviceByUUID(uuid string) (Device, error) {
	if strings.TrimSpace(uuid) == "" {
		uuid = strings.TrimSpace(os.Getenv("PASCA_DEVICE_UUID"))
	}
	if uuid == "" {
		return 0, fmt.Errorf("cuda: specify -device-uuid or PASCA_DEVICE_UUID (see nvidia-smi -L)")
	}

	count, err := DeviceCount()
	if err != nil {
		return 0, err
	}
	for ordinal := 0; ordinal < count; ordinal++ {
		device, err := DeviceByOrdinal(ordinal)
		if err != nil {
			return 0, err
		}
		candidate, err := device.UUID()
		if err != nil {
			return 0, err
		}
		if strings.EqualFold(candidate, uuid) {
			return device, nil
		}
	}
	return 0, fmt.Errorf("cuda: no device with UUID %s among %d visible devices", uuid, count)
}

// Context は、デバイスの primary context です。Close で参照を解放します。
type Context struct {
	device Device
	handle uintptr
}

// NewPrimaryContext は device の primary context を取得し、呼び出したスレッドの
// 現在のコンテキストに設定します。呼び出し前に runtime.LockOSThread を呼んでください。
func NewPrimaryContext(device Device) (*Context, error) {
	var handle uintptr
	if err := check("cuDevicePrimaryCtxRetain", driver.cuDevicePrimaryCtxRetain(&handle, int32(device))); err != nil {
		return nil, err
	}
	if err := check("cuCtxSetCurrent", driver.cuCtxSetCurrent(handle)); err != nil {
		driver.cuDevicePrimaryCtxRelease(int32(device))
		return nil, err
	}
	return &Context{device: device, handle: handle}, nil
}

// Close は primary context の参照を解放します。
func (c *Context) Close() error {
	return check("cuDevicePrimaryCtxRelease_v2", driver.cuDevicePrimaryCtxRelease(int32(c.device)))
}

// Module は、読み込んだ cubin（カーネルの集まり）です。
type Module struct {
	handle uintptr
}

// LoadModule は、メモリ上の cubin を読み込みます。image は go:embed で埋め込んだ cubin です。
func LoadModule(image []byte) (*Module, error) {
	if len(image) == 0 {
		return nil, fmt.Errorf("cuda: empty module image")
	}
	var handle uintptr
	if err := check("cuModuleLoadData", driver.cuModuleLoadData(&handle, &image[0])); err != nil {
		return nil, err
	}
	return &Module{handle: handle}, nil
}

// Function は、モジュール内の 1 つのカーネルです。
type Function struct {
	name   string
	handle uintptr
}

// Function は、extern "C" で公開したカーネルを名前で取得します。例: module.Function("axpy_f32")
func (m *Module) Function(name string) (*Function, error) {
	var handle uintptr
	nameBytes := cString(name)
	if err := check("cuModuleGetFunction", driver.cuModuleGetFunction(&handle, m.handle, &nameBytes[0])); err != nil {
		return nil, fmt.Errorf("%w (kernel %s)", err, name)
	}
	return &Function{name: name, handle: handle}, nil
}

// Close はモジュールを解放します。
func (m *Module) Close() error {
	return check("cuModuleUnload", driver.cuModuleUnload(m.handle))
}

// DevicePtr は GPU メモリのアドレスです。ホスト側のポインターと取り違えないよう別の型にしています。
type DevicePtr uintptr

// Alloc は GPU メモリを size バイト確保します。例: Alloc(4 * 1024) は float32 1024 個分
func Alloc(size int) (DevicePtr, error) {
	var pointer uintptr
	if err := check("cuMemAlloc_v2", driver.cuMemAlloc(&pointer, uintptr(size))); err != nil {
		return 0, err
	}
	return DevicePtr(pointer), nil
}

// Free は Alloc で確保した GPU メモリを解放します。
func Free(pointer DevicePtr) error {
	return check("cuMemFree_v2", driver.cuMemFree(uintptr(pointer)))
}

// コピーと初期化について:
// 同期版の cuMemcpyHtoD や cuMemsetD8 は既定ストリームで動き、ホスト側のメモリからのコピーでは
// GPU への転送完了前に戻ることがあります。このパッケージのストリームは既定ストリームと同期しない
// （streamNonBlocking）ため、同期版を使うと後続のカーネルが転送完了前に走ることがあります
// （M1a の最初の実行で、axpy_f32 の約 39% の要素が食い違った）。そのため、コピーと初期化は
// 必ずカーネルと同じストリームに積み、完了まで待ってから戻る形にしています。

// MemsetBytes は、GPU メモリの先頭 size バイトを value で埋める処理を stream に積み、完了まで待ちます。
func MemsetBytes(stream *Stream, destination DevicePtr, value uint8, size int) error {
	if err := check("cuMemsetD8Async", driver.cuMemsetD8Async(uintptr(destination), value, uintptr(size), stream.handle)); err != nil {
		return err
	}
	return stream.Synchronize()
}

// sliceBytes は、スライスの先頭アドレスとバイト数を返します。空のスライスは (nil, 0) です。
func sliceBytes[T any](values []T) (unsafe.Pointer, int) {
	if len(values) == 0 {
		return nil, 0
	}
	var zero T
	return unsafe.Pointer(&values[0]), len(values) * int(unsafe.Sizeof(zero))
}

// CopyToDevice は、ホストのスライスの内容を GPU メモリへコピーする処理を stream に積み、完了まで待ちます。
// T には float32・int32・uint8 など、Go のポインターを含まない型だけを使ってください。
func CopyToDevice[T any](stream *Stream, destination DevicePtr, source []T) error {
	pointer, size := sliceBytes(source)
	if size == 0 {
		return nil
	}
	var pinner runtime.Pinner
	defer pinner.Unpin()
	pinner.Pin(&source[0])
	if err := check("cuMemcpyHtoDAsync_v2", driver.cuMemcpyHtoDAsync(uintptr(destination), pointer, uintptr(size), stream.handle)); err != nil {
		return err
	}
	return stream.Synchronize()
}

// CopyFromDevice は、GPU メモリの内容をホストのスライスへコピーする処理を stream に積み、完了まで待ちます
// （len(destination) 要素分）。それまでに stream に積んだカーネルの結果が反映された値が返ります。
func CopyFromDevice[T any](stream *Stream, destination []T, source DevicePtr) error {
	pointer, size := sliceBytes(destination)
	if size == 0 {
		return nil
	}
	var pinner runtime.Pinner
	defer pinner.Unpin()
	pinner.Pin(&destination[0])
	if err := check("cuMemcpyDtoHAsync_v2", driver.cuMemcpyDtoHAsync(pointer, uintptr(source), uintptr(size), stream.handle)); err != nil {
		return err
	}
	return stream.Synchronize()
}

// CopyDeviceAsync は、GPU メモリの source から destination へ size バイトをコピーする処理を stream に積みます。
// 完了は待ちません。後に同じ stream に積んだカーネルは、コピーの完了後に動きます。
// 例: CopyDeviceAsync(stream, saved, history, 49152)
func CopyDeviceAsync(stream *Stream, destination, source DevicePtr, size int) error {
	if size <= 0 {
		return nil
	}
	return check("cuMemcpyDtoDAsync_v2", driver.cuMemcpyDtoDAsync(uintptr(destination), uintptr(source), uintptr(size), stream.handle))
}

// HostBuffer は、cuMemAllocHost で確保したページ固定（pinned）のホストメモリです。
// Go の GC の管理外なので、Close するまで有効です。
type HostBuffer struct {
	pointer unsafe.Pointer
	size    int
}

// AllocHost は、ページ固定のホストメモリを size バイト確保します。
func AllocHost(size int) (*HostBuffer, error) {
	var pointer unsafe.Pointer
	if err := check("cuMemAllocHost_v2", driver.cuMemAllocHost(&pointer, uintptr(size))); err != nil {
		return nil, err
	}
	return &HostBuffer{pointer: pointer, size: size}, nil
}

// Bytes は、固定メモリ全体をバイト列として見せます（コピーはしない）。Close 後は使えません。
func (h *HostBuffer) Bytes() []byte {
	return unsafe.Slice((*byte)(h.pointer), h.size)
}

// Int32s は、固定メモリを int32 の列として見せます。例: 64 バイトなら 16 要素
func (h *HostBuffer) Int32s() []int32 {
	return unsafe.Slice((*int32)(h.pointer), h.size/4)
}

// CopyFromDevice は、GPU メモリの先頭から、この固定メモリの大きさ分をコピーする処理を stream に積み、
// 完了まで待ちます。
func (h *HostBuffer) CopyFromDevice(stream *Stream, source DevicePtr) error {
	if err := check("cuMemcpyDtoHAsync_v2", driver.cuMemcpyDtoHAsync(h.pointer, uintptr(source), uintptr(h.size), stream.handle)); err != nil {
		return err
	}
	return stream.Synchronize()
}

// Close は固定メモリを解放します。
func (h *HostBuffer) Close() error {
	return check("cuMemFreeHost", driver.cuMemFreeHost(h.pointer))
}

// Stream は CUDA のストリーム（GPU への命令の列）です。
type Stream struct {
	handle uintptr
}

// streamNonBlocking は CU_STREAM_NON_BLOCKING（既定ストリームと暗黙に同期しない）です。
// CUDA Graph の記録には既定ストリーム以外が必要なため、このフラグで作ります。
const streamNonBlocking uint32 = 1

// NewStream は、既定ストリームと同期しないストリームを作ります。
func NewStream() (*Stream, error) {
	var handle uintptr
	if err := check("cuStreamCreate", driver.cuStreamCreate(&handle, streamNonBlocking)); err != nil {
		return nil, err
	}
	return &Stream{handle: handle}, nil
}

// Synchronize は、ストリームに積んだ処理がすべて終わるまで待ちます。
func (s *Stream) Synchronize() error {
	return check("cuStreamSynchronize", driver.cuStreamSynchronize(s.handle))
}

// Close はストリームを破棄します。
func (s *Stream) Close() error {
	return check("cuStreamDestroy_v2", driver.cuStreamDestroy(s.handle))
}

// captureModeThreadLocal は CU_STREAM_CAPTURE_MODE_THREAD_LOCAL です。
// 記録中に、同じスレッドからの危険な API 呼び出しだけを禁止します（他スレッドの CUDA 呼び出しは妨げない）。
const captureModeThreadLocal int32 = 1

// BeginCapture は、このストリームに積む処理の CUDA Graph への記録を始めます。
// 記録中に積んだカーネルは実行されず、EndCapture で Graph として返ります。
func (s *Stream) BeginCapture() error {
	return check("cuStreamBeginCapture_v2", driver.cuStreamBeginCapture(s.handle, captureModeThreadLocal))
}

// EndCapture は記録を終え、記録した処理を Graph として返します。
func (s *Stream) EndCapture() (*Graph, error) {
	var handle uintptr
	if err := check("cuStreamEndCapture", driver.cuStreamEndCapture(s.handle, &handle)); err != nil {
		return nil, err
	}
	return &Graph{handle: handle}, nil
}

// Graph は、記録した処理の並び（まだ実行可能な形ではない）です。
type Graph struct {
	handle uintptr
}

// Instantiate は、Graph を繰り返し実行できる GraphExec に変換します。
func (g *Graph) Instantiate() (*GraphExec, error) {
	var handle uintptr
	if err := check("cuGraphInstantiateWithFlags", driver.cuGraphInstantiateWithFlags(&handle, g.handle, 0)); err != nil {
		return nil, err
	}
	return &GraphExec{handle: handle}, nil
}

// Close は Graph を破棄します。Instantiate 済みの GraphExec には影響しません。
func (g *Graph) Close() error {
	return check("cuGraphDestroy", driver.cuGraphDestroy(g.handle))
}

// GraphExec は実行可能な Graph です。Launch 1 回で、記録したすべての処理をストリームに積みます。
type GraphExec struct {
	handle uintptr
}

// Launch は、記録した処理一式を stream に積みます（完了は待たない）。
func (g *GraphExec) Launch(stream *Stream) error {
	return check("cuGraphLaunch", driver.cuGraphLaunch(g.handle, stream.handle))
}

// Close は GraphExec を破棄します。
func (g *GraphExec) Close() error {
	return check("cuGraphExecDestroy", driver.cuGraphExecDestroy(g.handle))
}

// Event は、ストリーム上の時点を記録する印です。GPU 側の経過時間の計測に使います。
type Event struct {
	handle uintptr
}

// NewEvent は、時間計測ができるイベントを作ります（フラグ 0 = CU_EVENT_DEFAULT）。
func NewEvent() (*Event, error) {
	var handle uintptr
	if err := check("cuEventCreate", driver.cuEventCreate(&handle, 0)); err != nil {
		return nil, err
	}
	return &Event{handle: handle}, nil
}

// Record は、stream の現在の位置にこのイベントを置きます。
func (e *Event) Record(stream *Stream) error {
	return check("cuEventRecord", driver.cuEventRecord(e.handle, stream.handle))
}

// Synchronize は、このイベントの位置まで GPU の処理が進むのを待ちます。
func (e *Event) Synchronize() error {
	return check("cuEventSynchronize", driver.cuEventSynchronize(e.handle))
}

// ElapsedMilliseconds は、start から end までの GPU 上の経過時間をミリ秒で返します。
// 両方のイベントが完了している必要があります（end.Synchronize の後に呼ぶ）。
func ElapsedMilliseconds(start, end *Event) (float64, error) {
	var milliseconds float32
	if err := check("cuEventElapsedTime", driver.cuEventElapsedTime(&milliseconds, start.handle, end.handle)); err != nil {
		return 0, err
	}
	return float64(milliseconds), nil
}

// Close はイベントを破棄します。
func (e *Event) Close() error {
	return check("cuEventDestroy_v2", driver.cuEventDestroy(e.handle))
}

// Dim3 は、グリッドまたはブロックの大きさ（x, y, z）です。例: Dim3{X: 256, Y: 1, Z: 1}
type Dim3 struct {
	X, Y, Z uint32
}

// KernelArgs は、カーネルに渡す引数を順番に並べたものです。
//
// cuLaunchKernel は「各引数の値が入った場所を指すポインターの配列」を受け取ります。
// 値は 1 個 8 バイトの領域に格納し、そのアドレスを別の配列に並べます。4 バイトの int32・float32 は
// 先頭 4 バイトに入り、x86-64（リトルエンディアン）ではドライバーが正しく読めます。
// 例: args.Pointer(y).Pointer(x).Float32(2.0).Int32(1024) は
// カーネル axpy_f32(float* y, const float* x, float a, int n) の引数に対応します。
type KernelArgs struct {
	values []uint64
}

// Pointer は GPU メモリのアドレスを 1 個追加します。
func (a *KernelArgs) Pointer(value DevicePtr) *KernelArgs {
	a.values = append(a.values, uint64(value))
	return a
}

// Int32 は 32 ビット整数を 1 個追加します。
func (a *KernelArgs) Int32(value int32) *KernelArgs {
	a.values = append(a.values, uint64(uint32(value)))
	return a
}

// Float32 は 32 ビット浮動小数点数を 1 個追加します。
func (a *KernelArgs) Float32(value float32) *KernelArgs {
	a.values = append(a.values, uint64(math.Float32bits(value)))
	return a
}

// Count は追加した引数の数を返します。
func (a *KernelArgs) Count() int {
	return len(a.values)
}

// Launch は function を grid × block の大きさで stream に積みます（完了は待たない）。
// sharedBytes は動的共有メモリのバイト数で、使わない場合は 0 です。
// 記録中（BeginCapture 後）のストリームに積んだ場合は、Graph に記録されます。
func (f *Function) Launch(grid, block Dim3, sharedBytes uint32, stream *Stream, args *KernelArgs) error {
	var parameterArray unsafe.Pointer
	var pinner runtime.Pinner
	defer pinner.Unpin()
	if args != nil && len(args.values) > 0 {
		// 値の格納場所と、そのアドレスの配列を、呼び出しが終わるまで GC から固定する。
		pinner.Pin(&args.values[0])
		addresses := make([]uintptr, len(args.values))
		for index := range args.values {
			addresses[index] = uintptr(unsafe.Pointer(&args.values[index]))
		}
		pinner.Pin(&addresses[0])
		parameterArray = unsafe.Pointer(&addresses[0])
		defer runtime.KeepAlive(addresses)
	}
	code := driver.cuLaunchKernel(f.handle, grid.X, grid.Y, grid.Z, block.X, block.Y, block.Z, sharedBytes,
		stream.handle, parameterArray, nil)
	if args != nil {
		runtime.KeepAlive(args.values)
	}
	if err := check("cuLaunchKernel", code); err != nil {
		return fmt.Errorf("%w (kernel %s)", err, f.name)
	}
	return nil
}
