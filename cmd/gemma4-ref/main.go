// Command gemma4-ref は、Gemma 4 の CPU 参照実装（internal/gemma4）を、llama.cpp の結果と突き合わせます（M7）。
//
// 基準は、同じ GGUF を llama-perplexity --kl-divergence-base で評価したファイルです。同じトークン列を
// 参照実装で処理し、基準が保存している位置（各区切りの後半）で次トークンの分布を比べます。
// 重みは同じ（Q4_0 を詰め替えただけ）なので、-activations exact では KL がほぼ 0 になるはずです。
// -activations int8 にすると、GPU と同じ「活性値の int8 化」による差が加わります。
//
// 実行例:
//
//	gemma4-ref -pq4 ./data/gemma4-12b-qat.pq4m \
//	    -kld ./data/m7/gemma12b-c128.kld -chunks 1 -activations exact
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"
	"unsafe"

	"github.com/cardy0828/pascallm/internal/compare"
	"github.com/cardy0828/pascallm/internal/gemma4"
	"github.com/cardy0828/pascallm/internal/llamakld"
)

// options はコマンドライン引数です。
type options struct {
	pq4Path     string
	kldPath     string
	chunks      int
	activations string
	saveLogits  string
	halfCache   bool
}

// parseOptions はコマンドライン引数を読みます。
func parseOptions() (options, error) {
	var opts options
	flag.StringVar(&opts.pq4Path, "pq4", "", "Gemma 4 の .pq4m")
	flag.StringVar(&opts.kldPath, "kld", "", "llama-perplexity --kl-divergence-base のファイル")
	flag.IntVar(&opts.chunks, "chunks", 1, "比べる区切りの数（ファイルの区切りの数まで）")
	flag.StringVar(&opts.activations, "activations", "exact", "活性値の扱い: exact（FP32）または int8（GPU と同じ）")
	flag.StringVar(&opts.saveLogits, "save-logits", "", "比べた位置の logits を float32 のまま書くファイル（省略可。GPU 実装との比較の基準に使う）")
	flag.BoolVar(&opts.halfCache, "half-cache", false, "k・v を FP16 に丸めて KV キャッシュに入れる（GPU 実装と同じ）")
	flag.Parse()
	if opts.pq4Path == "" || opts.kldPath == "" {
		return opts, fmt.Errorf("-pq4 と -kld は必須です")
	}
	if opts.activations != "exact" && opts.activations != "int8" {
		return opts, fmt.Errorf("-activations は exact か int8 です: %q", opts.activations)
	}
	if opts.chunks < 1 {
		return opts, fmt.Errorf("-chunks は 1 以上です: %d", opts.chunks)
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

// logitsWriter は、比べた位置の logits を順に 1 つのファイルへ書きます（float32、位置ごとに語彙の数）。
// パスが空なら何も書きません。
type logitsWriter struct {
	file   *os.File
	writer *bufio.Writer
}

// newLogitsWriter は書き出し先を開きます。path が空なら、何も書かない logitsWriter を返します。
func newLogitsWriter(path string) (*logitsWriter, error) {
	if path == "" {
		return &logitsWriter{}, nil
	}
	file, err := os.Create(path)
	if err != nil {
		return nil, err
	}
	const bufferBytes = 4 << 20
	return &logitsWriter{file: file, writer: bufio.NewWriterSize(file, bufferBytes)}, nil
}

// write は 1 位置の logits を書きます。
func (w *logitsWriter) write(logits []float32) error {
	if w.writer == nil {
		return nil
	}
	const float32Bytes = 4
	_, err := w.writer.Write(unsafe.Slice((*byte)(unsafe.Pointer(&logits[0])), float32Bytes*len(logits)))
	return err
}

// close はバッファを書き出してファイルを閉じます。
func (w *logitsWriter) close() error {
	if w.writer == nil {
		return nil
	}
	if err := w.writer.Flush(); err != nil {
		w.file.Close()
		return err
	}
	return w.file.Close()
}

// run は突き合わせの全体です。
func run(opts options) error {
	base, err := llamakld.Open(opts.kldPath)
	if err != nil {
		return err
	}
	defer base.Close()
	weights, err := gemma4.OpenWeights(opts.pq4Path)
	if err != nil {
		return err
	}
	defer weights.Close()
	config := weights.Config
	if base.VocabSize != config.VocabSize {
		return fmt.Errorf("the base file has %d tokens in the vocabulary, the model has %d", base.VocabSize, config.VocabSize)
	}
	mode := gemma4.ActivationsExact
	if opts.activations == "int8" {
		mode = gemma4.ActivationsInt8
	}
	model := gemma4.NewModel(weights, mode)
	model.HalfPrecisionCache = opts.halfCache
	chunks := min(opts.chunks, base.Chunks)
	baseLogProbs := make([]float32, base.VocabSize)
	var total compare.Accumulator
	saved, err := newLogitsWriter(opts.saveLogits)
	if err != nil {
		return err
	}
	started := time.Now()
	processed := 0

	for chunk := 0; chunk < chunks; chunk++ {
		tokens := base.ChunkTokens(chunk)
		model.Reset()
		var perChunk compare.Accumulator
		// 最後のトークンは正解としてだけ使い、入力しません（基準も位置 n_ctx − 2 まで）。
		for position := 0; position < base.ContextSize-1; position++ {
			token := tokens[position]
			if position == 0 {
				token = config.BOSTokenID // llama-perplexity と同じく、区切りの先頭は BOS
			}
			logits, err := model.Forward(token)
			if err != nil {
				return err
			}
			processed++
			if position < base.FirstScored() {
				continue
			}
			floor, err := base.LogProbs(chunk, position, baseLogProbs)
			if err != nil {
				return err
			}
			perChunk.Add(baseLogProbs, floor, logits, tokens[position+1])
			total.Add(baseLogProbs, floor, logits, tokens[position+1])
			if err := saved.write(logits); err != nil {
				return err
			}
		}
		emit(map[string]any{"event": "chunk", "chunk": chunk, "summary": perChunk.Summary(),
			"seconds_per_token": time.Since(started).Seconds() / float64(processed)})
	}
	emit(map[string]any{"event": "total", "activations": opts.activations, "chunks": chunks,
		"context": base.ContextSize, "summary": total.Summary(), "seconds": time.Since(started).Seconds()})
	return saved.close()
}

func main() {
	opts, err := parseOptions()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	if err := run(opts); err != nil {
		fmt.Fprintln(os.Stderr, "gemma4-ref:", err)
		os.Exit(1)
	}
}
