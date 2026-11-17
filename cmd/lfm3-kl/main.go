// Command lfm3-kl は、同じトークン列を FP32 の参照実装（基準）と PQ4 の重み（評価対象）で処理し、
// 全位置で次トークン分布の差を測ります。
//
// 集計は llama.cpp の llama-perplexity --kl-divergence に合わせ、位置 -score-from 以降
// （既定は列の後半）の各位置で、次のトークンを予測する分布を比べます。
//   - KL(基準 || 評価対象) の平均・中央値・99 パーセンタイル・最大
//   - 上位 1 位の一致率（llama.cpp の「Same top p」）
//   - 両者の perplexity と、その比
//
// 基準の logits は計算に時間がかかるため、-base-logits のファイルに保存し、2 回目以降は読み込みます。
//
// 実行例:
//
//	go run ./cmd/lfm3-kl -tokens tokens.json -base-model-dir ./models/lfm3-1.7b-training \
//	    -base-logits base.f32 -pq4 model.pq4m
package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"runtime"
	"sort"
	"time"
	"unsafe"

	"github.com/cardy0828/pascallm/internal/lfm3"
	"github.com/cardy0828/pascallm/internal/lfm3gpu"
)

// defaultDeviceUUID は空なら PASCA_DEVICE_UUID を使用します。GPUは明示的に選択します。
const defaultDeviceUUID = ""

// forwarder は、トークンを 1 個ずつ処理して logits を返す計算器です（CPU の参照実装と GPU 実装の共通の形）。
type forwarder interface {
	Forward(token int) ([]float32, error)
}

// options はコマンドライン引数です。
type options struct {
	tokensPath   string
	baseModelDir string
	baseLogits   string
	pq4Path      string
	dequantize   bool
	scoreFrom    int
	label        string
	engine       string
	deviceUUID   string
}

// parseOptions はコマンドライン引数を読みます。
func parseOptions() (options, error) {
	var opts options
	flag.StringVar(&opts.tokensPath, "tokens", "", "トークン列の JSON（{\"token_ids\": [...]}）")
	flag.StringVar(&opts.baseModelDir, "base-model-dir", "", "基準にする Hugging Face 形式のモデル（-base-logits が無いときに計算する）")
	flag.StringVar(&opts.baseLogits, "base-logits", "", "基準の logits の保存先（float32、位置ごとに vocab 個）")
	flag.StringVar(&opts.pq4Path, "pq4", "", "評価する PQ4 のモデルファイル（.pq4m）")
	flag.BoolVar(&opts.dequantize, "dequantize", false, "PQ4 の重みを FP32 に戻して計算する（活性値の int8 化をしない）")
	flag.IntVar(&opts.scoreFrom, "score-from", -1, "集計を始める位置（-1 なら列の長さの半分。llama-perplexity と同じ）")
	flag.StringVar(&opts.label, "label", "", "結果に付ける名前")
	flag.StringVar(&opts.engine, "engine", "cpu", "評価対象の計算器: cpu（参照実装）または gpu（internal/lfm3gpu）")
	flag.StringVar(&opts.deviceUUID, "device-uuid", defaultDeviceUUID, "-engine gpu で使う GPU の UUID")
	flag.Parse()
	if opts.tokensPath == "" || opts.baseLogits == "" || opts.pq4Path == "" {
		return options{}, errors.New("-tokens, -base-logits and -pq4 are required")
	}
	if opts.engine != "cpu" && opts.engine != "gpu" {
		return options{}, fmt.Errorf("unknown -engine %q (want cpu or gpu)", opts.engine)
	}
	if opts.engine == "gpu" && opts.dequantize {
		return options{}, errors.New("-dequantize cannot be used with -engine gpu")
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

// loadTokens はトークン列の JSON を読みます。
func loadTokens(path string) ([]int, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var content struct {
		TokenIDs []int `json:"token_ids"`
	}
	if err := json.Unmarshal(raw, &content); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if len(content.TokenIDs) < 2 {
		return nil, fmt.Errorf("%s has fewer than 2 tokens", path)
	}
	return content.TokenIDs, nil
}

// float32Bytes は、float32 の配列をそのままのメモリ表現のバイト列として見せます（コピーしない）。
func float32Bytes(values []float32) []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(&values[0])), 4*len(values))
}

// ensureBaseLogits は、基準の logits のファイルが無ければ FP32 の参照実装で計算して書き、
// ファイルの大きさが「トークン数 × vocab × 4 バイト」と一致することを確かめます。
func ensureBaseLogits(opts options, tokens []int, vocab int) error {
	want := int64(len(tokens)) * int64(vocab) * 4
	if stat, err := os.Stat(opts.baseLogits); err == nil {
		if stat.Size() != want {
			return fmt.Errorf("%s has %d bytes, want %d", opts.baseLogits, stat.Size(), want)
		}
		return nil
	}
	if opts.baseModelDir == "" {
		return fmt.Errorf("%s does not exist and -base-model-dir is not given", opts.baseLogits)
	}
	config, source, err := lfm3.OpenSafetensorsModel(opts.baseModelDir)
	if err != nil {
		return err
	}
	defer source.File.Close()
	weights, err := lfm3.LoadWeights(config, source)
	if err != nil {
		return err
	}
	model := lfm3.NewModel(weights)
	temporary := opts.baseLogits + ".tmp"
	file, err := os.Create(temporary)
	if err != nil {
		return err
	}
	defer os.Remove(temporary)
	writer := bufio.NewWriterSize(file, 16<<20)
	started := time.Now()
	for _, token := range tokens {
		logits, err := model.Forward(token)
		if err != nil {
			file.Close()
			return err
		}
		if _, err := writer.Write(float32Bytes(logits)); err != nil {
			file.Close()
			return err
		}
	}
	if err := writer.Flush(); err != nil {
		file.Close()
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}
	emit(map[string]any{"kind": "base_logits", "path": opts.baseLogits, "tokens": len(tokens),
		"seconds": time.Since(started).Seconds()})
	return os.Rename(temporary, opts.baseLogits)
}

// logSoftmax は log(softmax(values)) を float64 で dst に書きます。
func logSoftmax(values []float32, dst []float64) {
	maximum := math.Inf(-1)
	for _, value := range values {
		maximum = math.Max(maximum, float64(value))
	}
	var sum float64
	for _, value := range values {
		sum += math.Exp(float64(value) - maximum)
	}
	logSum := maximum + math.Log(sum)
	for index, value := range values {
		dst[index] = float64(value) - logSum
	}
}

// argmax は最大値の添字を返します。例: [0.1, 0.9, 0.5] -> 1
func argmax(values []float32) int {
	best := 0
	for index, value := range values {
		if value > values[best] {
			best = index
		}
	}
	return best
}

// percentile は、昇順に並べた values の p（0〜1）分位点を返します（最も近い順位の値）。
// 例: percentile([1, 2, 3, 4], 0.5) = 2
func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	rank := int(math.Ceil(p*float64(len(sorted)))) - 1
	return sorted[min(max(rank, 0), len(sorted)-1)]
}

// openTestModel は、評価対象の計算器（CPU の参照実装または GPU 実装）を用意します。
// 戻り値の関数は後片付け（GPU のメモリの解放など）で、必ず呼んでください。
// tokens は処理するトークン数で、GPU の KV キャッシュの大きさに使います。
func openTestModel(opts options, config *lfm3.Config, source *lfm3.PQ4Source, tokens int) (forwarder, func(), error) {
	if opts.engine == "gpu" {
		model, err := lfm3gpu.Open(opts.pq4Path, lfm3gpu.Options{DeviceUUID: opts.deviceUUID, MaxContext: tokens})
		if err != nil {
			return nil, nil, err
		}
		return model, func() { model.Close() }, nil
	}
	weights, err := lfm3.LoadWeights(config, source)
	if err != nil {
		return nil, nil, err
	}
	if opts.dequantize {
		lfm3.DequantizeMatrices(weights)
	}
	return lfm3.NewModel(weights), func() {}, nil
}

// run は基準の用意・評価対象の計算・集計を行います。
func run(opts options) error {
	tokens, err := loadTokens(opts.tokensPath)
	if err != nil {
		return err
	}
	config, source, err := lfm3.OpenPQ4Model(opts.pq4Path)
	if err != nil {
		return err
	}
	defer source.File.Close()
	if err := ensureBaseLogits(opts, tokens, config.VocabSize); err != nil {
		return err
	}
	model, closeModel, err := openTestModel(opts, config, source, len(tokens))
	if err != nil {
		return err
	}
	defer closeModel()

	scoreFrom := opts.scoreFrom
	if scoreFrom < 0 {
		scoreFrom = len(tokens) / 2
	}
	baseFile, err := os.Open(opts.baseLogits)
	if err != nil {
		return err
	}
	defer baseFile.Close()
	baseReader := bufio.NewReaderSize(baseFile, 16<<20)

	vocab := config.VocabSize
	baseRow := make([]float32, vocab)
	baseLog := make([]float64, vocab)
	testLog := make([]float64, vocab)
	var klValues []float64
	var sameTop int
	var baseNLL, testNLL float64
	started := time.Now()
	// 位置 p の logits は、トークン p+1 を予測する分布です。最後の位置は予測先が無いので集計しません。
	for position := 0; position < len(tokens); position++ {
		logits, err := model.Forward(tokens[position])
		if err != nil {
			return err
		}
		if _, err := io.ReadFull(baseReader, float32Bytes(baseRow)); err != nil {
			return fmt.Errorf("reading base logits at position %d: %w", position, err)
		}
		if position < scoreFrom || position == len(tokens)-1 {
			continue
		}
		logSoftmax(baseRow, baseLog)
		logSoftmax(logits, testLog)
		var kl float64
		for index, value := range baseLog {
			kl += math.Exp(value) * (value - testLog[index])
		}
		klValues = append(klValues, kl)
		if argmax(baseRow) == argmax(logits) {
			sameTop++
		}
		next := tokens[position+1]
		baseNLL -= baseLog[next]
		testNLL -= testLog[next]
	}

	scored := len(klValues)
	if scored == 0 {
		return fmt.Errorf("no positions were scored (score-from %d, tokens %d)", scoreFrom, len(tokens))
	}
	var klSum float64
	for _, value := range klValues {
		klSum += value
	}
	sort.Float64s(klValues)
	emit(map[string]any{
		"kind":        "summary",
		"label":       opts.label,
		"pq4":         opts.pq4Path,
		"method":      source.File.Header.Converter["method"],
		"lm_head":     source.File.Header.Converter["lm_head"],
		"dequantized": opts.dequantize,
		"engine":      opts.engine,
		"tokens":      len(tokens),
		"score_from":  scoreFrom,
		"scored":      scored,
		"mean_kl":     klSum / float64(scored),
		"median_kl":   percentile(klValues, 0.5),
		"p99_kl":      percentile(klValues, 0.99),
		"max_kl":      klValues[scored-1],
		"same_top_p":  float64(sameTop) / float64(scored),
		"ppl_base":    math.Exp(baseNLL / float64(scored)),
		"ppl_test":    math.Exp(testNLL / float64(scored)),
		"ppl_ratio":   math.Exp((testNLL - baseNLL) / float64(scored)),
		"seconds":     time.Since(started).Seconds(),
	})
	return nil
}

func main() {
	// -engine gpu のとき、CUDA のコンテキストは OS スレッドごとに決まるため、main のゴルーチンを固定する。
	runtime.LockOSThread()
	opts, err := parseOptions()
	if err == nil {
		err = run(opts)
	}
	if err != nil {
		emit(map[string]any{"kind": "result", "ok": false, "error": err.Error()})
		os.Exit(1)
	}
	emit(map[string]any{"kind": "result", "ok": true})
}
