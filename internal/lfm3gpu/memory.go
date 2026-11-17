package lfm3gpu

import (
	"fmt"
	"unsafe"

	"github.com/cardy0828/pascallm/internal/cuda"
)

// stagingBytes は、ホストから GPU へ重みを送るときに使う固定メモリの大きさです。
// 重みは .pq4m ファイルを mmap した領域にあり、GC の管理外なので runtime.Pinner では固定できません。
// そこで固定メモリに少しずつ写してから転送します。
const stagingBytes = 64 << 20

// deviceMemory は、GPU メモリの確保をまとめて記録し、Close で一括して解放します。
type deviceMemory struct {
	stream      *cuda.Stream
	staging     *cuda.HostBuffer
	allocations []cuda.DevicePtr
	totalBytes  int64 // 確保した合計（報告用）
}

// newDeviceMemory は、転送用の固定メモリを確保して deviceMemory を作ります。
func newDeviceMemory(stream *cuda.Stream) (*deviceMemory, error) {
	staging, err := cuda.AllocHost(stagingBytes)
	if err != nil {
		return nil, err
	}
	return &deviceMemory{stream: stream, staging: staging}, nil
}

// alloc は GPU メモリを size バイト確保し、記録します。例: alloc(4 * 2048) は float32 2048 個分
func (m *deviceMemory) alloc(size int) (cuda.DevicePtr, error) {
	if size <= 0 {
		return 0, fmt.Errorf("lfm3gpu: cannot allocate %d bytes", size)
	}
	pointer, err := cuda.Alloc(size)
	if err != nil {
		return 0, fmt.Errorf("lfm3gpu: allocating %d bytes: %w", size, err)
	}
	m.allocations = append(m.allocations, pointer)
	m.totalBytes += int64(size)
	return pointer, nil
}

// copyBytes は、ホストのバイト列 data を GPU の destination へコピーし、完了まで待ちます。
// 固定メモリに stagingBytes ずつ写して転送します。
func (m *deviceMemory) copyBytes(destination cuda.DevicePtr, data []byte) error {
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

// close は、確保したすべての GPU メモリと転送用の固定メモリを解放します。最初のエラーを返します。
func (m *deviceMemory) close() error {
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

// asBytes は、ポインターを含まない型のスライスを、同じメモリのバイト列として見せます（コピーしない）。
// 例: asBytes([]float32{1, 2}) は 8 バイト
func asBytes[T any](values []T) []byte {
	if len(values) == 0 {
		return nil
	}
	var zero T
	return unsafe.Slice((*byte)(unsafe.Pointer(&values[0])), len(values)*int(unsafe.Sizeof(zero)))
}

// upload は、スライスと同じ大きさの GPU メモリを確保して内容をコピーし、そのアドレスを返します。
// 例: upload(memory, normWeights) は float32 の正規化の重みを GPU に置く
func upload[T any](m *deviceMemory, values []T) (cuda.DevicePtr, error) {
	data := asBytes(values)
	pointer, err := m.alloc(len(data))
	if err != nil {
		return 0, err
	}
	if err := m.copyBytes(pointer, data); err != nil {
		return 0, err
	}
	return pointer, nil
}
