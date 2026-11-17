// Package gguf は、llama.cpp の GGUF ファイル（版 2・3）を読み取り専用で開きます。
//
// 使い道は、GGUF の Q4_0 の重みを PQ4（.pq4m）に詰め替える変換（cmd/gguf-to-pq4）です。
// メタデータ（キーと値）とテンソルの一覧を読み、テンソルの中身は mmap した領域をそのまま返します。
//
// ファイルの配置（すべてリトルエンディアン）:
//
//	"GGUF"、版（uint32）、テンソル数（uint64）、メタデータ数（uint64）
//	メタデータ: キー（文字列）、型（uint32）、値 … を個数分
//	テンソル情報: 名前（文字列）、次元数（uint32）、各次元（uint64）、型（uint32）、オフセット（uint64）… を個数分
//	データ領域: general.alignment（既定 32）の倍数の位置から。テンソルのオフセットはこの先頭から数える
//
// 文字列は「長さ（uint64）+ バイト列」です。次元は ggml の順（最初が最も内側）で、
// 行列（nn.Linear の重み）は [入力次元, 出力次元] と書かれます。
package gguf

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"syscall"
)

// ファイルの形式に関する定数です。
const (
	magic              = "GGUF"
	minimumVersion     = 2
	maximumVersion     = 3
	defaultAlignment   = 32
	alignmentKey       = "general.alignment"
	maximumStringBytes = 1 << 30 // 文字列 1 個の上限（壊れたファイルで巨大な確保をしないため）
	maximumDimensions  = 8       // テンソルの次元数の上限
)

// メタデータの値の型の番号です（GGUF の仕様の順）。
const (
	valueUint8   = 0
	valueInt8    = 1
	valueUint16  = 2
	valueInt16   = 3
	valueUint32  = 4
	valueInt32   = 5
	valueFloat32 = 6
	valueBool    = 7
	valueString  = 8
	valueArray   = 9
	valueUint64  = 10
	valueInt64   = 11
	valueFloat64 = 12
)

// TensorType はテンソルの保存形式の番号です（ggml_type）。このパッケージが大きさを計算できるのは下の 3 つだけです。
type TensorType uint32

const (
	TypeF32  TensorType = 0 // FP32
	TypeF16  TensorType = 1 // FP16
	TypeQ4_0 TensorType = 2 // 32 要素ごとに FP16 の係数 1 個と 4bit の値 32 個（18 バイト）
)

// Q4_0 の 1 ブロックの寸法です。
const (
	Q4BlockElements = 32 // 1 ブロックの要素数
	Q4BlockBytes    = 18 // 1 ブロックのバイト数（FP16 の係数 2 バイト + 4bit × 32 個 = 16 バイト）
	Q4ScaleBytes    = 2  // ブロックの先頭にある係数のバイト数
)

// String は形式の名前を返します。例: TypeQ4_0 -> "Q4_0"
func (t TensorType) String() string {
	switch t {
	case TypeF32:
		return "F32"
	case TypeF16:
		return "F16"
	case TypeQ4_0:
		return "Q4_0"
	default:
		return fmt.Sprintf("type%d", uint32(t))
	}
}

// TensorInfo はテンソル 1 個の情報です。
// 例: {Name: "blk.0.attn_q.weight", Dims: [3840, 4096], Type: TypeQ4_0}（入力 3840 次元、出力 4096 次元）
type TensorInfo struct {
	Name   string
	Dims   []int // ggml の順（最初が最も内側）
	Type   TensorType
	Offset int64 // データ領域の先頭からの位置
}

// Elements は要素の数（全次元の積）を返します。例: Dims [3840, 4096] -> 15,728,640
func (t TensorInfo) Elements() int64 {
	elements := int64(1)
	for _, dim := range t.Dims {
		elements *= int64(dim)
	}
	return elements
}

// ByteSize は中身のバイト数を返します。対応していない形式ならエラーです。
// 例: Q4_0 の 3840 × 4096 -> 15,728,640 / 32 × 18 = 8,847,360
func (t TensorInfo) ByteSize() (int64, error) {
	elements := t.Elements()
	switch t.Type {
	case TypeF32:
		return 4 * elements, nil
	case TypeF16:
		return 2 * elements, nil
	case TypeQ4_0:
		if len(t.Dims) == 0 || t.Dims[0]%Q4BlockElements != 0 {
			return 0, fmt.Errorf("gguf: tensor %s has Q4_0 with inner dimension %v, which is not a multiple of %d", t.Name, t.Dims, Q4BlockElements)
		}
		return elements / Q4BlockElements * Q4BlockBytes, nil
	default:
		return 0, fmt.Errorf("gguf: tensor %s has unsupported type %s", t.Name, t.Type)
	}
}

// File は読み取り用に開いた GGUF ファイルです。Close で mmap を解除します。
type File struct {
	Version  uint32
	Metadata map[string]any // 値は Go の型（uint8〜uint64、int8〜int64、float32、float64、bool、string、[]any）
	Tensors  []TensorInfo   // ファイルに書かれた順

	path    string
	mapping []byte
	data    []byte // データ領域
	byName  map[string]int
}

// reader は、mmap した領域を先頭から順に読む道具です。範囲を超えたら err に記録し、以後は 0 を返します。
type reader struct {
	bytes  []byte
	offset int
	err    error
}

// take は次の count バイトを返して進みます。足りなければエラーを記録して nil を返します。
func (r *reader) take(count int) []byte {
	if r.err != nil {
		return nil
	}
	if count < 0 || r.offset+count > len(r.bytes) {
		r.err = fmt.Errorf("gguf: unexpected end of file at offset %d (need %d bytes)", r.offset, count)
		return nil
	}
	part := r.bytes[r.offset : r.offset+count]
	r.offset += count
	return part
}

// uint32 は 4 バイトの符号なし整数を読みます。
func (r *reader) uint32() uint32 {
	part := r.take(4)
	if part == nil {
		return 0
	}
	return binary.LittleEndian.Uint32(part)
}

// uint64 は 8 バイトの符号なし整数を読みます。
func (r *reader) uint64() uint64 {
	part := r.take(8)
	if part == nil {
		return 0
	}
	return binary.LittleEndian.Uint64(part)
}

// text は「長さ（uint64）+ バイト列」の文字列を読みます。
func (r *reader) text() string {
	length := r.uint64()
	if r.err == nil && length > maximumStringBytes {
		r.err = fmt.Errorf("gguf: string of %d bytes at offset %d is too long", length, r.offset)
	}
	return string(r.take(int(length)))
}

// scalar は、配列でない型の値を 1 個読みます。
func (r *reader) scalar(kind uint32) any {
	switch kind {
	case valueUint8:
		if part := r.take(1); part != nil {
			return part[0]
		}
	case valueInt8:
		if part := r.take(1); part != nil {
			return int8(part[0])
		}
	case valueUint16:
		if part := r.take(2); part != nil {
			return binary.LittleEndian.Uint16(part)
		}
	case valueInt16:
		if part := r.take(2); part != nil {
			return int16(binary.LittleEndian.Uint16(part))
		}
	case valueUint32:
		return r.uint32()
	case valueInt32:
		return int32(r.uint32())
	case valueFloat32:
		return math.Float32frombits(r.uint32())
	case valueBool:
		if part := r.take(1); part != nil {
			return part[0] != 0
		}
	case valueString:
		return r.text()
	case valueUint64:
		return r.uint64()
	case valueInt64:
		return int64(r.uint64())
	case valueFloat64:
		return math.Float64frombits(r.uint64())
	default:
		if r.err == nil {
			r.err = fmt.Errorf("gguf: unknown value type %d at offset %d", kind, r.offset)
		}
	}
	return nil
}

// value は、型 kind の値を 1 個読みます。配列は []any（要素は scalar と同じ Go の型）にします。
func (r *reader) value(kind uint32) any {
	if kind != valueArray {
		return r.scalar(kind)
	}
	elementKind := r.uint32()
	count := r.uint64()
	if r.err != nil {
		return nil
	}
	if elementKind == valueArray {
		r.err = fmt.Errorf("gguf: nested arrays are not supported (offset %d)", r.offset)
		return nil
	}
	// 1 要素は最低 1 バイトなので、残りのバイト数より多い個数は壊れたファイルです。
	if count > uint64(len(r.bytes)-r.offset) {
		r.err = fmt.Errorf("gguf: array of %d elements at offset %d is longer than the file", count, r.offset)
		return nil
	}
	items := make([]any, 0, count)
	for index := uint64(0); index < count && r.err == nil; index++ {
		items = append(items, r.scalar(elementKind))
	}
	return items
}

// Open は GGUF ファイルを mmap して開き、メタデータとテンソルの一覧を読みます。
// 例: Open("./models/gemma-4-12b-it-qat/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf")
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
	mapping, err := syscall.Mmap(int(handle.Fd()), 0, int(stat.Size()), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("gguf: mmap %s: %w", path, err)
	}
	file, err := parse(path, mapping)
	if err != nil {
		syscall.Munmap(mapping)
		return nil, err
	}
	return file, nil
}

// parse は、mmap した領域からヘッダー・メタデータ・テンソル情報を読み、各テンソルがファイルに収まるかを確かめます。
func parse(path string, mapping []byte) (*File, error) {
	r := &reader{bytes: mapping}
	if string(r.take(len(magic))) != magic {
		return nil, fmt.Errorf("gguf: %s is not a GGUF file", path)
	}
	version := r.uint32()
	if r.err == nil && (version < minimumVersion || version > maximumVersion) {
		return nil, fmt.Errorf("gguf: %s has version %d (supported: %d to %d)", path, version, minimumVersion, maximumVersion)
	}
	tensorCount := r.uint64()
	metadataCount := r.uint64()
	file := &File{Version: version, Metadata: make(map[string]any), path: path, mapping: mapping, byName: make(map[string]int)}
	for index := uint64(0); index < metadataCount && r.err == nil; index++ {
		key := r.text()
		kind := r.uint32()
		file.Metadata[key] = r.value(kind)
	}
	for index := uint64(0); index < tensorCount && r.err == nil; index++ {
		info := TensorInfo{Name: r.text()}
		dimensions := r.uint32()
		if r.err == nil && dimensions > maximumDimensions {
			return nil, fmt.Errorf("gguf: tensor %s has %d dimensions", info.Name, dimensions)
		}
		for dimension := uint32(0); dimension < dimensions; dimension++ {
			info.Dims = append(info.Dims, int(r.uint64()))
		}
		info.Type = TensorType(r.uint32())
		info.Offset = int64(r.uint64())
		file.byName[info.Name] = len(file.Tensors)
		file.Tensors = append(file.Tensors, info)
	}
	if r.err != nil {
		return nil, fmt.Errorf("%w (%s)", r.err, path)
	}
	alignment := int64(defaultAlignment)
	if _, ok := file.Metadata[alignmentKey]; ok {
		value, err := file.Int(alignmentKey)
		if err != nil || value <= 0 {
			return nil, fmt.Errorf("gguf: %s has an invalid %s", path, alignmentKey)
		}
		alignment = value
	}
	dataStart := (int64(r.offset) + alignment - 1) / alignment * alignment
	if dataStart > int64(len(mapping)) {
		return nil, fmt.Errorf("gguf: %s is truncated before the data section", path)
	}
	file.data = mapping[dataStart:]
	for _, info := range file.Tensors {
		size, err := info.ByteSize()
		if err != nil {
			continue // 対応していない形式は、Bytes で読もうとしたときにエラーにする
		}
		if info.Offset < 0 || info.Offset+size > int64(len(file.data)) {
			return nil, fmt.Errorf("gguf: tensor %s is outside the file %s", info.Name, path)
		}
	}
	return file, nil
}

// Close は mmap を解除します。Close の後は、このファイルから得たバイト列を使ってはいけません。
func (f *File) Close() error {
	return syscall.Munmap(f.mapping)
}

// Tensor は名前を指定してテンソルの情報を返します。
func (f *File) Tensor(name string) (TensorInfo, error) {
	index, ok := f.byName[name]
	if !ok {
		return TensorInfo{}, fmt.Errorf("gguf: tensor %s not found in %s", name, f.path)
	}
	return f.Tensors[index], nil
}

// Bytes はテンソルの中身のバイト列を返します（mmap した領域をそのまま返し、コピーしない）。
func (f *File) Bytes(info TensorInfo) ([]byte, error) {
	size, err := info.ByteSize()
	if err != nil {
		return nil, err
	}
	return f.data[info.Offset : info.Offset+size], nil
}

// lookup はメタデータの値を返します。なければエラーです。
func (f *File) lookup(key string) (any, error) {
	value, ok := f.Metadata[key]
	if !ok {
		return nil, fmt.Errorf("gguf: metadata %s not found in %s", key, f.path)
	}
	return value, nil
}

// toInt は、整数の型の値を int64 にします。整数でなければ ok = false です。
func toInt(value any) (int64, bool) {
	switch typed := value.(type) {
	case uint8:
		return int64(typed), true
	case int8:
		return int64(typed), true
	case uint16:
		return int64(typed), true
	case int16:
		return int64(typed), true
	case uint32:
		return int64(typed), true
	case int32:
		return int64(typed), true
	case uint64:
		return int64(typed), true
	case int64:
		return typed, true
	default:
		return 0, false
	}
}

// Int は整数のメタデータを返します。例: Int("gemma4.block_count") -> 48
func (f *File) Int(key string) (int64, error) {
	value, err := f.lookup(key)
	if err != nil {
		return 0, err
	}
	number, ok := toInt(value)
	if !ok {
		return 0, fmt.Errorf("gguf: metadata %s is %T, want an integer", key, value)
	}
	return number, nil
}

// Float は小数のメタデータを返します。例: Float("gemma4.rope.freq_base") -> 1e6
func (f *File) Float(key string) (float64, error) {
	value, err := f.lookup(key)
	if err != nil {
		return 0, err
	}
	switch typed := value.(type) {
	case float32:
		return float64(typed), nil
	case float64:
		return typed, nil
	default:
		return 0, fmt.Errorf("gguf: metadata %s is %T, want a float", key, value)
	}
}

// String は文字列のメタデータを返します。例: String("general.architecture") -> "gemma4"
func (f *File) String(key string) (string, error) {
	value, err := f.lookup(key)
	if err != nil {
		return "", err
	}
	text, ok := value.(string)
	if !ok {
		return "", fmt.Errorf("gguf: metadata %s is %T, want a string", key, value)
	}
	return text, nil
}

// array は配列のメタデータを返します。
func (f *File) array(key string) ([]any, error) {
	value, err := f.lookup(key)
	if err != nil {
		return nil, err
	}
	items, ok := value.([]any)
	if !ok {
		return nil, fmt.Errorf("gguf: metadata %s is %T, want an array", key, value)
	}
	return items, nil
}

// IntArray は整数の配列のメタデータを返します。例: IntArray("gemma4.attention.head_count_kv") -> [8, 8, 8, 8, 8, 1, …]
func (f *File) IntArray(key string) ([]int64, error) {
	items, err := f.array(key)
	if err != nil {
		return nil, err
	}
	numbers := make([]int64, len(items))
	for index, item := range items {
		number, ok := toInt(item)
		if !ok {
			return nil, fmt.Errorf("gguf: metadata %s[%d] is %T, want an integer", key, index, item)
		}
		numbers[index] = number
	}
	return numbers, nil
}

// BoolArray は真偽値の配列のメタデータを返します。
// 例: BoolArray("gemma4.attention.sliding_window_pattern") -> [true, true, true, true, true, false, …]
func (f *File) BoolArray(key string) ([]bool, error) {
	items, err := f.array(key)
	if err != nil {
		return nil, err
	}
	flags := make([]bool, len(items))
	for index, item := range items {
		flag, ok := item.(bool)
		if !ok {
			return nil, fmt.Errorf("gguf: metadata %s[%d] is %T, want a bool", key, index, item)
		}
		flags[index] = flag
	}
	return flags, nil
}

// StringArray は文字列の配列のメタデータを返します。例: StringArray("tokenizer.ggml.tokens")
func (f *File) StringArray(key string) ([]string, error) {
	items, err := f.array(key)
	if err != nil {
		return nil, err
	}
	texts := make([]string, len(items))
	for index, item := range items {
		text, ok := item.(string)
		if !ok {
			return nil, fmt.Errorf("gguf: metadata %s[%d] is %T, want a string", key, index, item)
		}
		texts[index] = text
	}
	return texts, nil
}
