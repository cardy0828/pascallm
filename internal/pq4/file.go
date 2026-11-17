package pq4

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/crc32"
	"io"
	"os"
	"syscall"
	"unsafe"
)

// モデルファイル（拡張子 .pq4m）の配置:
//
//	[0, 8)        マジック "PQ4MODEL"
//	[8, 16)       ヘッダー JSON のバイト数（uint64、リトルエンディアン）
//	[16, 16+H)    ヘッダー JSON（FileHeader）
//	[D, ...)      データ領域。D は 16+H を Alignment に切り上げた位置
//
// 各テンソルの各部分（PQ4 なら本体と係数）は、データ領域の先頭からのオフセットで指し、
// オフセットは Alignment の倍数です。mmap した後、そのまま型付きの配列として読めます。
const (
	fileMagic       = "PQ4MODEL"
	FormatVersion   = 1
	Alignment       = 4096
	prefixBytes     = 16
	maxHeaderLength = 64 << 20
)

// TensorKind はテンソルの保存形式です。
type TensorKind string

const (
	KindPQ4 TensorKind = "pq4" // 部分 "packed"（4bit 本体）と "scales"（FP16 係数）
	KindF16 TensorKind = "f16" // 部分 "data"（FP16）
	KindF32 TensorKind = "f32" // 部分 "data"（FP32）
)

// Part は、テンソルを構成する 1 つの連続領域です。CRC32C はデータの破損を検出するためのものです。
type Part struct {
	Name   string `json:"name"`
	Offset int64  `json:"offset"`
	Size   int64  `json:"size"`
	CRC32C uint32 `json:"crc32c"`
}

// TensorRecord は、ヘッダーに書く 1 テンソルの情報です。
// 例: {Name: "model.layers.0.feed_forward.w1.weight", Kind: "pq4", Shape: [7680, 2048], Parts: [...]}
type TensorRecord struct {
	Name  string     `json:"name"`
	Kind  TensorKind `json:"kind"`
	Shape []int      `json:"shape"`
	Parts []Part     `json:"parts"`
}

// FileHeader はヘッダー JSON の全体です。
type FileHeader struct {
	FormatVersion int               `json:"format_version"`
	Alignment     int               `json:"alignment"`
	ModelConfig   json.RawMessage   `json:"model_config"` // 変換元の config.json をそのまま入れる
	Source        map[string]string `json:"source"`       // 変換元のパス・SHA-256 など
	Converter     map[string]string `json:"converter"`    // 変換ツールの名前・量子化方法など
	Tensors       []TensorRecord    `json:"tensors"`
}

// TensorData は、書き込むテンソル 1 個の中身です。Kind に応じて PQ4・F16・F32 のどれか 1 つを入れます。
type TensorData struct {
	Name  string
	Kind  TensorKind
	Shape []int
	PQ4   *Matrix
	F16   []uint16
	F32   []float32
}

// rawPart は、書き込む 1 部分の名前とバイト列です。
type rawPart struct {
	name  string
	bytes []byte
}

// parts は、テンソルを部分ごとのバイト列に分けます（コピーしない）。
func (t TensorData) parts() ([]rawPart, error) {
	switch t.Kind {
	case KindPQ4, KindPQ8:
		if t.PQ4 == nil || t.PQ4.IsPQ8() != (t.Kind == KindPQ8) {
			return nil, fmt.Errorf("pq4: tensor %s has kind %s but no matching matrix", t.Name, t.Kind)
		}
		return []rawPart{{"packed", t.PQ4.packed}, {"scales", uint16Bytes(t.PQ4.scales)}}, nil
	case KindF16:
		return []rawPart{{"data", uint16Bytes(t.F16)}}, nil
	case KindF32:
		return []rawPart{{"data", float32Bytes(t.F32)}}, nil
	default:
		return nil, fmt.Errorf("pq4: tensor %s has unknown kind %q", t.Name, t.Kind)
	}
}

// uint16Bytes・float32Bytes は、配列をそのままのメモリ表現（リトルエンディアン）のバイト列として見せます。
// x86-64 はリトルエンディアンなので、ファイルに書いたものを mmap してそのまま読み戻せます。
func uint16Bytes(values []uint16) []byte {
	if len(values) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&values[0])), 2*len(values))
}

func float32Bytes(values []float32) []byte {
	if len(values) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&values[0])), 4*len(values))
}

// alignUp は value を Alignment の倍数に切り上げます。例: 5000 -> 8192
func alignUp(value int64) int64 {
	return (value + Alignment - 1) / Alignment * Alignment
}

var crcTable = crc32.MakeTable(crc32.Castagnoli)

// WriteFile は、header（Tensors 以外を埋めたもの）と tensors を path に書きます。
// オフセットと CRC32C はこの関数が計算して header.Tensors に入れます。
// 書き込みは一時ファイルに行い、成功した場合だけ path へ名前を変えます（途中で失敗しても壊れたファイルを残さない）。
func WriteFile(path string, header FileHeader, tensors []TensorData) error {
	header.FormatVersion = FormatVersion
	header.Alignment = Alignment
	header.Tensors = header.Tensors[:0]
	var allParts [][]rawPart
	var offset int64
	for _, tensor := range tensors {
		parts, err := tensor.parts()
		if err != nil {
			return err
		}
		record := TensorRecord{Name: tensor.Name, Kind: tensor.Kind, Shape: tensor.Shape}
		for _, part := range parts {
			record.Parts = append(record.Parts, Part{Name: part.name, Offset: offset, Size: int64(len(part.bytes)),
				CRC32C: crc32.Checksum(part.bytes, crcTable)})
			offset = alignUp(offset + int64(len(part.bytes)))
		}
		header.Tensors = append(header.Tensors, record)
		allParts = append(allParts, parts)
	}
	headerJSON, err := json.Marshal(header)
	if err != nil {
		return err
	}

	temporary := path + ".tmp"
	file, err := os.Create(temporary)
	if err != nil {
		return err
	}
	defer os.Remove(temporary) // 名前の変更に成功した後は、何もしない
	writer := bufio.NewWriterSize(file, 16<<20)
	if err := writeAll(writer, headerJSON, header.Tensors, allParts); err != nil {
		file.Close()
		return err
	}
	if err := writer.Flush(); err != nil {
		file.Close()
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}
	return os.Rename(temporary, path)
}

// writeAll は、先頭・ヘッダー・各部分を、パディングを挟みながら順に書きます。
func writeAll(writer io.Writer, headerJSON []byte, records []TensorRecord, allParts [][]rawPart) error {
	var prefix [prefixBytes]byte
	copy(prefix[:8], fileMagic)
	binary.LittleEndian.PutUint64(prefix[8:], uint64(len(headerJSON)))
	if _, err := writer.Write(prefix[:]); err != nil {
		return err
	}
	if _, err := writer.Write(headerJSON); err != nil {
		return err
	}
	written := int64(prefixBytes + len(headerJSON))
	dataStart := alignUp(written)
	if err := writePadding(writer, dataStart-written); err != nil {
		return err
	}
	position := int64(0) // データ領域の先頭からの位置
	for tensorIndex, record := range records {
		for partIndex, part := range record.Parts {
			if err := writePadding(writer, part.Offset-position); err != nil {
				return err
			}
			if _, err := writer.Write(allParts[tensorIndex][partIndex].bytes); err != nil {
				return err
			}
			position = part.Offset + part.Size
		}
	}
	return nil
}

// writePadding は count バイトの 0 を書きます。
func writePadding(writer io.Writer, count int64) error {
	if count < 0 {
		return fmt.Errorf("pq4: internal error: negative padding %d", count)
	}
	_, err := writer.Write(make([]byte, count))
	return err
}

// File は、読み取り用に開いた .pq4m ファイルです。Close で mmap を解除します。
type File struct {
	Header  FileHeader
	path    string
	mapping []byte
	data    []byte
	records map[string]TensorRecord
}

// OpenFile は .pq4m ファイルを mmap して開き、全部分の範囲と CRC32C を確かめます。
// 例: OpenFile("./data/lfm3-1.7b-pq4-search-imatrix.pq4m")
func OpenFile(path string) (*File, error) {
	handle, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer handle.Close()
	stat, err := handle.Stat()
	if err != nil {
		return nil, err
	}
	if stat.Size() < prefixBytes {
		return nil, fmt.Errorf("pq4: %s is too small", path)
	}
	mapping, err := syscall.Mmap(int(handle.Fd()), 0, int(stat.Size()), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("pq4: mmap %s: %w", path, err)
	}
	file, err := parseFile(path, mapping)
	if err != nil {
		syscall.Munmap(mapping)
		return nil, err
	}
	return file, nil
}

// parseFile はヘッダーを読み、各部分がデータ領域に収まり CRC32C が一致するかを確かめます。
func parseFile(path string, mapping []byte) (*File, error) {
	if string(mapping[:8]) != fileMagic {
		return nil, fmt.Errorf("pq4: %s is not a PQ4 model file", path)
	}
	headerLength := binary.LittleEndian.Uint64(mapping[8:prefixBytes])
	if headerLength > maxHeaderLength || prefixBytes+headerLength > uint64(len(mapping)) {
		return nil, fmt.Errorf("pq4: %s has an invalid header length %d", path, headerLength)
	}
	var header FileHeader
	if err := json.Unmarshal(mapping[prefixBytes:prefixBytes+headerLength], &header); err != nil {
		return nil, fmt.Errorf("pq4: %s header: %w", path, err)
	}
	if header.FormatVersion != FormatVersion || header.Alignment != Alignment {
		return nil, fmt.Errorf("pq4: %s has format version %d / alignment %d, want %d / %d",
			path, header.FormatVersion, header.Alignment, FormatVersion, Alignment)
	}
	dataStart := alignUp(int64(prefixBytes + headerLength))
	if dataStart > int64(len(mapping)) {
		return nil, fmt.Errorf("pq4: %s is truncated before the data section", path)
	}
	data := mapping[dataStart:]
	records := make(map[string]TensorRecord, len(header.Tensors))
	for _, record := range header.Tensors {
		for _, part := range record.Parts {
			if part.Offset < 0 || part.Size < 0 || part.Offset%Alignment != 0 || part.Offset+part.Size > int64(len(data)) {
				return nil, fmt.Errorf("pq4: %s tensor %s part %s is outside the file", path, record.Name, part.Name)
			}
			if crc32.Checksum(data[part.Offset:part.Offset+part.Size], crcTable) != part.CRC32C {
				return nil, fmt.Errorf("pq4: %s tensor %s part %s failed the CRC32C check", path, record.Name, part.Name)
			}
		}
		records[record.Name] = record
	}
	return &File{Header: header, path: path, mapping: mapping, data: data, records: records}, nil
}

// Close は mmap を解除します。Close の後は、このファイルから得た行列や配列を使ってはいけません。
func (f *File) Close() error {
	return syscall.Munmap(f.mapping)
}

// Record は、名前を指定してテンソルの情報を返します。
func (f *File) Record(name string) (TensorRecord, error) {
	record, ok := f.records[name]
	if !ok {
		return TensorRecord{}, fmt.Errorf("pq4: tensor %s not found in %s", name, f.path)
	}
	return record, nil
}

// partBytes は、テンソルの指定した部分のバイト列を返します（コピーしない）。
func (f *File) partBytes(record TensorRecord, name string) ([]byte, error) {
	for _, part := range record.Parts {
		if part.Name == name {
			return f.data[part.Offset : part.Offset+part.Size], nil
		}
	}
	return nil, fmt.Errorf("pq4: tensor %s has no part %s", record.Name, name)
}

// Matrix は、PQ4・PQ8 のテンソルを行列として返します（mmap した領域をそのまま使い、コピーしない）。
func (f *File) Matrix(name string, rows, cols int) (*Matrix, error) {
	record, err := f.Record(name)
	if err != nil {
		return nil, err
	}
	if (record.Kind != KindPQ4 && record.Kind != KindPQ8) || len(record.Shape) != 2 || record.Shape[0] != rows ||
		record.Shape[1] != cols {
		return nil, fmt.Errorf("pq4: tensor %s is %s %v, want pq4 or pq8 [%d %d]", name, record.Kind, record.Shape, rows, cols)
	}
	packed, err := f.partBytes(record, "packed")
	if err != nil {
		return nil, err
	}
	scaleBytes, err := f.partBytes(record, "scales")
	if err != nil {
		return nil, err
	}
	scales := unsafe.Slice((*uint16)(unsafe.Pointer(unsafe.SliceData(scaleBytes))), len(scaleBytes)/2)
	if record.Kind == KindPQ8 {
		return NewPQ8Matrix(rows, cols, packed, scales)
	}
	return NewMatrix(rows, cols, packed, scales)
}

// Float16 は、F16 のテンソルを FP16 のビット列として返します（コピーしない）。
func (f *File) Float16(name string) ([]uint16, []int, error) {
	record, err := f.Record(name)
	if err != nil {
		return nil, nil, err
	}
	if record.Kind != KindF16 {
		return nil, nil, fmt.Errorf("pq4: tensor %s is %s, want f16", name, record.Kind)
	}
	raw, err := f.partBytes(record, "data")
	if err != nil {
		return nil, nil, err
	}
	return unsafe.Slice((*uint16)(unsafe.Pointer(unsafe.SliceData(raw))), len(raw)/2), record.Shape, nil
}

// Float32 は、F32 のテンソルを返します（コピーしない）。
func (f *File) Float32(name string) ([]float32, []int, error) {
	record, err := f.Record(name)
	if err != nil {
		return nil, nil, err
	}
	if record.Kind != KindF32 {
		return nil, nil, fmt.Errorf("pq4: tensor %s is %s, want f32", name, record.Kind)
	}
	raw, err := f.partBytes(record, "data")
	if err != nil {
		return nil, nil, err
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(unsafe.SliceData(raw))), len(raw)/4), record.Shape, nil
}
