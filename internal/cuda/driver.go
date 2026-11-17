// Package cuda は、cgo を使わずに CUDA Driver API（libcuda.so）を呼ぶ薄いラッパーです。
//
// ライブラリは実行時に purego で dlopen します。そのため、このパッケージを使うプログラムは
// CGO_ENABLED=0 でビルドできます。CUDA のコンテキストは OS スレッドごとに「現在のもの」が
// 決まるため、CUDA を呼ぶゴルーチンは最初に runtime.LockOSThread を呼んでください。
package cuda

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// driverLibraryName は、NVIDIA ドライバーが提供する Driver API のライブラリ名です。
// 開発用の libcuda.so（シンボリックリンク）ではなく、実行時に必ず存在する .so.1 を読みます。
const driverLibraryName = "libcuda.so.1"

// driverFunctions は、libcuda.so.1 から取り出した関数の一覧です。
// 各フィールドの型は C の宣言に対応します（CUresult は int32、ハンドル類は uintptr）。
type driverFunctions struct {
	cuInit                      func(flags uint32) int32
	cuDriverGetVersion          func(version *int32) int32
	cuGetErrorName              func(code int32, name **byte) int32
	cuDeviceGetCount            func(count *int32) int32
	cuDeviceGet                 func(device *int32, ordinal int32) int32
	cuDeviceGetName             func(name *byte, length int32, device int32) int32
	cuDeviceGetUuid             func(uuid *byte, device int32) int32
	cuDeviceGetAttribute        func(value *int32, attribute int32, device int32) int32
	cuDevicePrimaryCtxRetain    func(context *uintptr, device int32) int32
	cuDevicePrimaryCtxRelease   func(device int32) int32
	cuCtxSetCurrent             func(context uintptr) int32
	cuModuleLoadData            func(module *uintptr, image *byte) int32
	cuModuleUnload              func(module uintptr) int32
	cuModuleGetFunction         func(function *uintptr, module uintptr, name *byte) int32
	cuMemAlloc                  func(pointer *uintptr, size uintptr) int32
	cuMemFree                   func(pointer uintptr) int32
	cuMemAllocHost              func(pointer *unsafe.Pointer, size uintptr) int32
	cuMemFreeHost               func(pointer unsafe.Pointer) int32
	cuMemcpyHtoDAsync           func(destination uintptr, source unsafe.Pointer, size uintptr, stream uintptr) int32
	cuMemcpyDtoHAsync           func(destination unsafe.Pointer, source uintptr, size uintptr, stream uintptr) int32
	cuMemcpyDtoDAsync           func(destination uintptr, source uintptr, size uintptr, stream uintptr) int32
	cuMemsetD8Async             func(destination uintptr, value uint8, count uintptr, stream uintptr) int32
	cuStreamCreate              func(stream *uintptr, flags uint32) int32
	cuStreamDestroy             func(stream uintptr) int32
	cuStreamSynchronize         func(stream uintptr) int32
	cuStreamBeginCapture        func(stream uintptr, mode int32) int32
	cuStreamEndCapture          func(stream uintptr, graph *uintptr) int32
	cuGraphInstantiateWithFlags func(exec *uintptr, graph uintptr, flags uint64) int32
	cuGraphLaunch               func(exec uintptr, stream uintptr) int32
	cuGraphExecDestroy          func(exec uintptr) int32
	cuGraphDestroy              func(graph uintptr) int32
	cuEventCreate               func(event *uintptr, flags uint32) int32
	cuEventDestroy              func(event uintptr) int32
	cuEventRecord               func(event uintptr, stream uintptr) int32
	cuEventSynchronize          func(event uintptr) int32
	cuEventElapsedTime          func(milliseconds *float32, start uintptr, end uintptr) int32
	cuLaunchKernel              func(function uintptr, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedBytes uint32,
		stream uintptr, kernelParams unsafe.Pointer, extra unsafe.Pointer) int32
}

// driverSymbol は、Go のフィールドと、libcuda.so.1 内の実際のシンボル名の対応です。
// 64 ビット版の API は「_v2」付きの名前で公開されているものがあるため、明示的に書きます。
type driverSymbol struct {
	target any
	name   string
}

var (
	loadOnce  sync.Once
	loadError error
	driver    driverFunctions
)

// symbolTable は、driver の各フィールドと読み込むシンボル名を並べたものです。
func symbolTable(functions *driverFunctions) []driverSymbol {
	return []driverSymbol{
		{&functions.cuInit, "cuInit"},
		{&functions.cuDriverGetVersion, "cuDriverGetVersion"},
		{&functions.cuGetErrorName, "cuGetErrorName"},
		{&functions.cuDeviceGetCount, "cuDeviceGetCount"},
		{&functions.cuDeviceGet, "cuDeviceGet"},
		{&functions.cuDeviceGetName, "cuDeviceGetName"},
		{&functions.cuDeviceGetUuid, "cuDeviceGetUuid_v2"},
		{&functions.cuDeviceGetAttribute, "cuDeviceGetAttribute"},
		{&functions.cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain"},
		{&functions.cuDevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease_v2"},
		{&functions.cuCtxSetCurrent, "cuCtxSetCurrent"},
		{&functions.cuModuleLoadData, "cuModuleLoadData"},
		{&functions.cuModuleUnload, "cuModuleUnload"},
		{&functions.cuModuleGetFunction, "cuModuleGetFunction"},
		{&functions.cuMemAlloc, "cuMemAlloc_v2"},
		{&functions.cuMemFree, "cuMemFree_v2"},
		{&functions.cuMemAllocHost, "cuMemAllocHost_v2"},
		{&functions.cuMemFreeHost, "cuMemFreeHost"},
		{&functions.cuMemcpyHtoDAsync, "cuMemcpyHtoDAsync_v2"},
		{&functions.cuMemcpyDtoHAsync, "cuMemcpyDtoHAsync_v2"},
		{&functions.cuMemcpyDtoDAsync, "cuMemcpyDtoDAsync_v2"},
		{&functions.cuMemsetD8Async, "cuMemsetD8Async"},
		{&functions.cuStreamCreate, "cuStreamCreate"},
		{&functions.cuStreamDestroy, "cuStreamDestroy_v2"},
		{&functions.cuStreamSynchronize, "cuStreamSynchronize"},
		{&functions.cuStreamBeginCapture, "cuStreamBeginCapture_v2"},
		{&functions.cuStreamEndCapture, "cuStreamEndCapture"},
		{&functions.cuGraphInstantiateWithFlags, "cuGraphInstantiateWithFlags"},
		{&functions.cuGraphLaunch, "cuGraphLaunch"},
		{&functions.cuGraphExecDestroy, "cuGraphExecDestroy"},
		{&functions.cuGraphDestroy, "cuGraphDestroy"},
		{&functions.cuEventCreate, "cuEventCreate"},
		{&functions.cuEventDestroy, "cuEventDestroy_v2"},
		{&functions.cuEventRecord, "cuEventRecord"},
		{&functions.cuEventSynchronize, "cuEventSynchronize"},
		{&functions.cuEventElapsedTime, "cuEventElapsedTime"},
		{&functions.cuLaunchKernel, "cuLaunchKernel"},
	}
}

// loadDriver は libcuda.so.1 を読み込み、全シンボルを driver に登録します。
// 2 回目以降の呼び出しは、1 回目の結果（成功または同じエラー）をそのまま返します。
// 例: シンボルが欠けていれば "cuda: symbol cuGraphLaunch not found: ..." を返す。
func loadDriver() error {
	loadOnce.Do(func() {
		handle, err := purego.Dlopen(driverLibraryName, purego.RTLD_NOW|purego.RTLD_GLOBAL)
		if err != nil {
			loadError = fmt.Errorf("cuda: cannot open %s: %w", driverLibraryName, err)
			return
		}
		for _, symbol := range symbolTable(&driver) {
			address, err := purego.Dlsym(handle, symbol.name)
			if err != nil {
				loadError = fmt.Errorf("cuda: symbol %s not found: %w", symbol.name, err)
				return
			}
			purego.RegisterFunc(symbol.target, address)
		}
	})
	return loadError
}
