// Package safetensors は、Hugging Face の safetensors ファイルを読み取り専用で開き、
// テンソルを float32 に変換して取り出します。
//
// ファイルは mmap するため、3 GB を超える重みでも読み込み時に全体をメモリへコピーしません。
package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"syscall"
	"unsafe"

	"github.com/cardy0828/pascallm/internal/numeric"
)

// headerLengthBytes は、ファイル先頭にある「ヘッダー JSON の長さ」（uint64）のバイト数です。
const headerLengthBytes = 8

// maxHeaderBytes は、壊れたファイルで巨大な確保をしないためのヘッダー長の上限（100 MiB）です。
const maxHeaderBytes = 100 << 20

// TensorInfo は、ヘッダーに書かれた 1 テンソルの情報です。
// 例: {DType: "BF16", Shape: [6144, 2048], Begin: 0, End: 25165824}
type TensorInfo struct {
	Name  string
	DType string
	Shape []int
	Begin int64 // データ領域の先頭からのオフセット（バイト）
	End   int64
}

// Elements はテンソルの要素数を返します。例: Shape [6144, 2048] -> 12,582,912
func (t TensorInfo) Elements() int {
	count := 1
	for _, dimension := range t.Shape {
		count *= dimension
	}
	return count
}

// File は開いた safetensors ファイルです。Close で mmap を解除します。
type File struct {
	path    string
	mapping []byte
	data    []byte // ヘッダーの後ろのデータ領域
	tensors map[string]TensorInfo
}

// rawTensorEntry は、ヘッダー JSON の 1 項目の形です。
type rawTensorEntry struct {
	DType       string  `json:"dtype"`
	Shape       []int   `json:"shape"`
	DataOffsets []int64 `json:"data_offsets"`
}

// Open は safetensors ファイルを mmap して、ヘッダーを読みます。
// 例: Open("./models/lfm3-1.7b-training/model.safetensors")
func Open(path string) (*File, error) {
	handle, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer handle.Close()
	stat, err := handle.Stat()
	if err != nil {
		return nil, err
	}
	if stat.Size() < headerLengthBytes {
		return nil, fmt.Errorf("safetensors: %s is too small (%d bytes)", path, stat.Size())
	}
	mapping, err := syscall.Mmap(int(handle.Fd()), 0, int(stat.Size()), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("safetensors: mmap %s: %w", path, err)
	}
	file, err := parse(path, mapping)
	if err != nil {
		syscall.Munmap(mapping)
		return nil, err
	}
	return file, nil
}

// parse は mmap したバイト列からヘッダーを解釈し、各テンソルの範囲を確かめます。
func parse(path string, mapping []byte) (*File, error) {
	headerLength := binary.LittleEndian.Uint64(mapping[:headerLengthBytes])
	if headerLength > maxHeaderBytes || headerLengthBytes+headerLength > uint64(len(mapping)) {
		return nil, fmt.Errorf("safetensors: %s has an invalid header length %d", path, headerLength)
	}
	headerEnd := headerLengthBytes + int(headerLength)
	var entries map[string]json.RawMessage
	if err := json.Unmarshal(mapping[headerLengthBytes:headerEnd], &entries); err != nil {
		return nil, fmt.Errorf("safetensors: %s header: %w", path, err)
	}
	data := mapping[headerEnd:]
	tensors := make(map[string]TensorInfo, len(entries))
	for name, raw := range entries {
		if name == "__metadata__" {
			continue
		}
		var entry rawTensorEntry
		if err := json.Unmarshal(raw, &entry); err != nil {
			return nil, fmt.Errorf("safetensors: %s tensor %s: %w", path, name, err)
		}
		if len(entry.DataOffsets) != 2 || entry.DataOffsets[0] < 0 || entry.DataOffsets[1] < entry.DataOffsets[0] ||
			entry.DataOffsets[1] > int64(len(data)) {
			return nil, fmt.Errorf("safetensors: %s tensor %s has invalid offsets %v", path, name, entry.DataOffsets)
		}
		info := TensorInfo{Name: name, DType: entry.DType, Shape: entry.Shape,
			Begin: entry.DataOffsets[0], End: entry.DataOffsets[1]}
		elementBytes, err := dtypeBytes(entry.DType)
		if err != nil {
			return nil, fmt.Errorf("safetensors: %s tensor %s: %w", path, name, err)
		}
		if int64(info.Elements()*elementBytes) != info.End-info.Begin {
			return nil, fmt.Errorf("safetensors: %s tensor %s size does not match shape %v", path, name, entry.Shape)
		}
		tensors[name] = info
	}
	return &File{path: path, mapping: mapping, data: data, tensors: tensors}, nil
}

// dtypeBytes は、safetensors の型名ごとの 1 要素のバイト数を返します。対応するのは BF16・F16・F32 だけです。
func dtypeBytes(dtype string) (int, error) {
	switch dtype {
	case "BF16", "F16":
		return 2, nil
	case "F32":
		return 4, nil
	default:
		return 0, fmt.Errorf("unsupported dtype %s", dtype)
	}
}

// Close は mmap を解除します。Close の後は、このファイルから得たスライスを使ってはいけません。
func (f *File) Close() error {
	return syscall.Munmap(f.mapping)
}

// Names は、含まれるテンソル名を辞書順で返します。
func (f *File) Names() []string {
	names := make([]string, 0, len(f.tensors))
	for name := range f.tensors {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// Info は、指定した名前のテンソル情報を返します。
func (f *File) Info(name string) (TensorInfo, error) {
	info, ok := f.tensors[name]
	if !ok {
		return TensorInfo{}, fmt.Errorf("safetensors: tensor %s not found in %s", name, f.path)
	}
	return info, nil
}

// Float32 は、テンソルを float32 の新しいスライスに変換して返します。
// wantShape を渡すと形状が一致するかも確かめます（nil なら確かめない）。
// 例: Float32("model.layers.0.ffn_norm.weight", []int{2048})
func (f *File) Float32(name string, wantShape []int) ([]float32, error) {
	info, err := f.Info(name)
	if err != nil {
		return nil, err
	}
	if wantShape != nil && !sameShape(info.Shape, wantShape) {
		return nil, fmt.Errorf("safetensors: tensor %s has shape %v, want %v", name, info.Shape, wantShape)
	}
	raw := f.data[info.Begin:info.End]
	result := make([]float32, info.Elements())
	switch info.DType {
	case "BF16":
		for index := range result {
			result[index] = numeric.BF16ToFloat32(binary.LittleEndian.Uint16(raw[2*index:]))
		}
	case "F16":
		for index := range result {
			result[index] = numeric.Float16ToFloat32(binary.LittleEndian.Uint16(raw[2*index:]))
		}
	case "F32":
		copy(result, unsafe.Slice((*float32)(unsafe.Pointer(&raw[0])), len(result)))
	}
	return result, nil
}

// sameShape は、2 つの形状が同じかを返します。例: sameShape([2048, 1, 7], [2048, 1, 7]) = true
func sameShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for index := range a {
		if a[index] != b[index] {
			return false
		}
	}
	return true
}
