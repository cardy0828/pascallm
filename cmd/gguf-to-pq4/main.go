// Command gguf-to-pq4 は、GGUF の Q4_0 の重みを、値を変えずに PQ4（.pq4m）へ詰め替えます（M7）。
//
// Q4_0 と PQ4 は同じ量子化（32 要素ごとに FP16 の係数、4bit の値から 8 を引く）で、配置だけが違います。
// そのため、元の FP16・BF16 の重みも imatrix も使わずに変換でき、品質は元の GGUF と同じです。
// 対応するのは Gemma 4（general.architecture = gemma4）で、行列がすべて Q4_0、ほかが F32 の GGUF です。
//
//   - Q4_0 の 2 次元テンソル [入力, 出力] → PQ4 の行列 [出力, 入力]
//   - F32 のテンソル → そのまま F32
//   - テンソル名は GGUF のまま。モデルの構成は GGUF のメタデータから作り、ヘッダーに保存する
//
// 書いた後で .pq4m を開き直し、全ブロックの本体と係数が GGUF と同じことを確かめます（-verify=false で省略）。
//
// 実行例:
//
//	gguf-to-pq4 -gguf ./models/gemma-4-12b-it-qat/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf \
//	    -out ./data/gemma4-12b-qat.pq4m
package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/cardy0828/pascallm/internal/gemma4"
	"github.com/cardy0828/pascallm/internal/gguf"
	"github.com/cardy0828/pascallm/internal/linear"
	"github.com/cardy0828/pascallm/internal/pq4"
)

// matrixDimensions は、行列（nn.Linear の重み）の次元数です。
const matrixDimensions = 2

// options はコマンドライン引数です。
type options struct {
	ggufPath string
	outPath  string
	verify   bool
}

// parseOptions はコマンドライン引数を読みます。
func parseOptions() (options, error) {
	var opts options
	flag.StringVar(&opts.ggufPath, "gguf", "", "変換元の GGUF（行列がすべて Q4_0 の Gemma 4）")
	flag.StringVar(&opts.outPath, "out", "", "書き出す .pq4m のパス")
	flag.BoolVar(&opts.verify, "verify", true, "書いた後で開き直し、全ブロックが GGUF と同じことを確かめる")
	flag.Parse()
	if opts.ggufPath == "" || opts.outPath == "" {
		return opts, fmt.Errorf("-gguf と -out は必須です")
	}
	return opts, nil
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

// sha256File はファイル全体の SHA-256 を 16 進で返します。
func sha256File(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()
	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return "", err
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
}

// reversed は、次元の並びを逆にした新しいスライスを返します（ggml の順 → 行優先の形）。
// 例: [3840, 4096] -> [4096, 3840]
func reversed(dims []int) []int {
	result := make([]int, len(dims))
	for index, dim := range dims {
		result[len(dims)-1-index] = dim
	}
	return result
}

// float32View は、4 の倍数の長さのバイト列を float32 の配列として見せます（コピーしない）。
// GGUF のテンソルは 32 バイト境界に置かれるため、そのまま読めます。
func float32View(data []byte) []float32 {
	if len(data) == 0 {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(unsafe.SliceData(data))), len(data)/4)
}

// convertTensor は、GGUF のテンソル 1 個を .pq4m に書く形にします。
func convertTensor(source *gguf.File, info gguf.TensorInfo) (pq4.TensorData, error) {
	data, err := source.Bytes(info)
	if err != nil {
		return pq4.TensorData{}, err
	}
	shape := reversed(info.Dims)
	switch info.Type {
	case gguf.TypeQ4_0:
		if len(info.Dims) != matrixDimensions {
			return pq4.TensorData{}, fmt.Errorf("tensor %s is Q4_0 with shape %v (only matrices are supported)", info.Name, info.Dims)
		}
		rows, cols := shape[0], shape[1]
		matrix, err := pq4.FromQ4_0(rows, cols, data)
		if err != nil {
			return pq4.TensorData{}, fmt.Errorf("tensor %s: %w", info.Name, err)
		}
		return pq4.TensorData{Name: info.Name, Kind: pq4.KindPQ4, Shape: shape, PQ4: matrix}, nil
	case gguf.TypeF32:
		return pq4.TensorData{Name: info.Name, Kind: pq4.KindF32, Shape: shape, F32: float32View(data)}, nil
	default:
		return pq4.TensorData{}, fmt.Errorf("tensor %s has type %s (only Q4_0 and F32 are supported)", info.Name, info.Type)
	}
}

// verifyMatrix は、.pq4m の行列 name の全ブロックが、GGUF の Q4_0 のバイト列 data と同じかを確かめます。
// 違うブロックの数を返します。
func verifyMatrix(written *pq4.File, name string, rows, cols int, data []byte) (int64, error) {
	matrix, err := written.Matrix(name, rows, cols)
	if err != nil {
		return 0, err
	}
	packed := matrix.Packed()
	scales := matrix.Scales()
	var mismatches atomic.Int64
	linear.ParallelRows(len(scales), func(begin, end int) {
		for block := begin; block < end; block++ {
			source := data[block*gguf.Q4BlockBytes : (block+1)*gguf.Q4BlockBytes]
			sameScale := binary.LittleEndian.Uint16(source[:gguf.Q4ScaleBytes]) == scales[block]
			sameBody := bytes.Equal(source[gguf.Q4ScaleBytes:], packed[block*pq4.BlockBytes:(block+1)*pq4.BlockBytes])
			if !sameScale || !sameBody {
				mismatches.Add(1)
			}
		}
	})
	return mismatches.Load(), nil
}

// verifyFile は、書いた .pq4m を開き直し（CRC32C の確認を含む）、全テンソルを GGUF と比べます。
// 違いがあればエラーを返します。
func verifyFile(source *gguf.File, outPath string) error {
	written, err := pq4.OpenFile(outPath)
	if err != nil {
		return err
	}
	defer written.Close()
	var blocks, mismatches int64
	for _, info := range source.Tensors {
		data, err := source.Bytes(info)
		if err != nil {
			return err
		}
		shape := reversed(info.Dims)
		switch info.Type {
		case gguf.TypeQ4_0:
			different, err := verifyMatrix(written, info.Name, shape[0], shape[1], data)
			if err != nil {
				return err
			}
			blocks += int64(len(data) / gguf.Q4BlockBytes)
			mismatches += different
		case gguf.TypeF32:
			values, _, err := written.Float32(info.Name)
			if err != nil {
				return err
			}
			if !bytes.Equal(data, unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(values))), 4*len(values))) {
				mismatches++
			}
		}
	}
	emit(map[string]any{"event": "verify", "q4_blocks": blocks, "mismatches": mismatches})
	if mismatches != 0 {
		return fmt.Errorf("%d blocks or tensors differ from the GGUF", mismatches)
	}
	return nil
}

// run は変換の全体です。
func run(opts options) error {
	started := time.Now()
	source, err := gguf.Open(opts.ggufPath)
	if err != nil {
		return err
	}
	defer source.Close()
	config, err := gemma4.ConfigFromGGUF(source)
	if err != nil {
		return err
	}
	configJSON, err := json.Marshal(config)
	if err != nil {
		return err
	}
	digest, err := sha256File(opts.ggufPath)
	if err != nil {
		return err
	}
	modelName, _ := source.String("general.name") // 名前は記録用なので、なくてもよい

	tensors := make([]pq4.TensorData, 0, len(source.Tensors))
	var matrices, vectors int
	for _, info := range source.Tensors {
		tensor, err := convertTensor(source, info)
		if err != nil {
			return err
		}
		if tensor.Kind == pq4.KindPQ4 {
			matrices++
		} else {
			vectors++
		}
		tensors = append(tensors, tensor)
	}
	header := pq4.FileHeader{
		ModelConfig: configJSON,
		Source:      map[string]string{"path": opts.ggufPath, "sha256": digest, "format": "gguf", "name": modelName},
		Converter:   map[string]string{"tool": "gguf-to-pq4", "method": "repack-q4_0"},
	}
	if err := pq4.WriteFile(opts.outPath, header, tensors); err != nil {
		return err
	}
	stat, err := os.Stat(opts.outPath)
	if err != nil {
		return err
	}
	emit(map[string]any{"event": "written", "out": opts.outPath, "bytes": stat.Size(), "pq4_matrices": matrices,
		"f32_tensors": vectors, "layers": config.NumLayers(), "vocab": config.VocabSize,
		"seconds": time.Since(started).Seconds()})
	if !opts.verify {
		return nil
	}
	return verifyFile(source, opts.outPath)
}

func main() {
	opts, err := parseOptions()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	if err := run(opts); err != nil {
		fmt.Fprintln(os.Stderr, "gguf-to-pq4:", err)
		os.Exit(1)
	}
}
