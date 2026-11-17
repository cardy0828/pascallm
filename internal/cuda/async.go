package cuda

import (
	"fmt"
	"unsafe"
)

// 待たないコピーについて:
// 1 トークンごとの処理では、「入力をコピー → Graph を再生 → 結果をコピー」をストリームに積み、
// 最後に 1 回だけ待ちます。途中で待つと、その都度 GPU が止まるためです。
// 待たないコピーは、GC に動かされないページ固定のメモリ（HostBuffer）との間でだけ提供します。

// Name は、カーネルの名前を返します。例: "gemv_pq4"
func (f *Function) Name() string { return f.name }

// Float32s は、固定メモリを float32 の列として見せます（コピーしない）。例: 512 KB なら 131072 要素
func (h *HostBuffer) Float32s() []float32 {
	return unsafe.Slice((*float32)(h.pointer), h.size/4)
}

// CopyToDeviceAsync は、この固定メモリ全体を GPU メモリの destination へコピーする処理を stream に積みます。
// 完了は待ちません。コピーが終わるまで、固定メモリの内容を書き換えないでください。
func (h *HostBuffer) CopyToDeviceAsync(stream *Stream, destination DevicePtr) error {
	return check("cuMemcpyHtoDAsync_v2",
		driver.cuMemcpyHtoDAsync(uintptr(destination), h.pointer, uintptr(h.size), stream.handle))
}

// CopyPrefixToDeviceAsync は、この固定メモリの先頭 size バイトを destination へコピーする処理を stream に積みます。
// 完了は待ちません。大きなデータを固定メモリ経由で少しずつ転送するときに使います。
// 例: 64 MB の固定メモリに 10 MB だけ詰めたときは size = 10 << 20
func (h *HostBuffer) CopyPrefixToDeviceAsync(stream *Stream, destination DevicePtr, size int) error {
	if size < 0 || size > h.size {
		return fmt.Errorf("cuda: copy of %d bytes from a %d-byte host buffer", size, h.size)
	}
	return check("cuMemcpyHtoDAsync_v2",
		driver.cuMemcpyHtoDAsync(uintptr(destination), h.pointer, uintptr(size), stream.handle))
}

// CopyFromDeviceAsync は、GPU メモリの source から、この固定メモリの大きさ分をコピーする処理を stream に積みます。
// 完了は待ちません。stream.Synchronize の後で内容を読んでください。
func (h *HostBuffer) CopyFromDeviceAsync(stream *Stream, source DevicePtr) error {
	return check("cuMemcpyDtoHAsync_v2",
		driver.cuMemcpyDtoHAsync(h.pointer, uintptr(source), uintptr(h.size), stream.handle))
}
