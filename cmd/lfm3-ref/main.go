// Command lfm3-ref は、Go の参照実装（internal/lfm3）で基準のトークン列を 1 トークンずつ処理し、
// Hugging Face 実装で作った基準 logits（tools/reference/lfm3_reference.py の出力）と比べます。
//
// 重みは次のどちらかから読みます。
//   - -model-dir: Hugging Face 形式（BF16 を FP32 に変換）。参照実装そのものの正しさを確かめる
//   - -pq4: PQ4 のモデルファイル（.pq4m）。量子化による誤差を測る
//
// 実行例:
//
//	go run ./cmd/lfm3-ref -model-dir ./models/lfm3-1.7b-training \
//	    -reference ./data/reference-lfm3-1.7b
//
// 結果は 1 行 1 件の JSON で標準出力に書きます。-max-abs-diff と -min-top1 で合格条件を指定でき、
// 満たさない場合は終了コード 1 で終わります。
package main

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/cardy0828/pascallm/internal/lfm3"
)

// options はコマンドライン引数です。
type options struct {
	modelDir   string
	pq4Path    string
	reference  string
	maxTokens  int
	maxAbsDiff float64 // 全語彙を保存した位置での logits の最大絶対差の上限（0 なら判定しない）
	minTop1    float64 // 上位 1 位の一致率の下限（0〜1、0 なら判定しない）
}

// parseOptions はコマンドライン引数を読みます。
func parseOptions() (options, error) {
	var opts options
	flag.StringVar(&opts.modelDir, "model-dir", "", "Hugging Face 形式のモデルのディレクトリ")
	flag.StringVar(&opts.pq4Path, "pq4", "", "PQ4 のモデルファイル（.pq4m）")
	flag.StringVar(&opts.reference, "reference", "", "基準データのディレクトリ（reference.json と logits_full.f32）")
	flag.IntVar(&opts.maxTokens, "max-tokens", 0, "処理するトークン数の上限（0 なら全部）")
	flag.Float64Var(&opts.maxAbsDiff, "max-abs-diff", 0, "合格条件: 全語彙の logits の最大絶対差の上限（0 なら判定しない）")
	flag.Float64Var(&opts.minTop1, "min-top1", 0, "合格条件: 上位 1 位の一致率の下限（0〜1、0 なら判定しない）")
	flag.Parse()
	if (opts.modelDir == "") == (opts.pq4Path == "") {
		return options{}, errors.New("specify exactly one of -model-dir and -pq4")
	}
	if opts.reference == "" {
		return options{}, errors.New("-reference is required")
	}
	return opts, nil
}

// reference は reference.json の内容です（使う項目だけ）。
type reference struct {
	TokenIDs      []int             `json:"token_ids"`
	VocabSize     int               `json:"vocab_size"`
	PositionsFull []int             `json:"positions_full"`
	TopK          int               `json:"top_k"`
	TopIDs        [][]int           `json:"top_ids"`
	TopValues     [][]float64       `json:"top_values"`
	Environment   map[string]any    `json:"environment"`
	fullLogits    map[int][]float32 // 位置 → 全語彙の logits
}

// loadReference は基準データを読みます。logits_full.f32 は positions_full の順に vocab 個ずつ並んだ float32 です。
func loadReference(directory string) (*reference, error) {
	raw, err := os.ReadFile(filepath.Join(directory, "reference.json"))
	if err != nil {
		return nil, err
	}
	var ref reference
	if err := json.Unmarshal(raw, &ref); err != nil {
		return nil, fmt.Errorf("reference.json: %w", err)
	}
	logitBytes, err := os.ReadFile(filepath.Join(directory, "logits_full.f32"))
	if err != nil {
		return nil, err
	}
	if len(logitBytes) != 4*ref.VocabSize*len(ref.PositionsFull) {
		return nil, fmt.Errorf("logits_full.f32 has %d bytes, want %d", len(logitBytes), 4*ref.VocabSize*len(ref.PositionsFull))
	}
	ref.fullLogits = make(map[int][]float32, len(ref.PositionsFull))
	for rowIndex, position := range ref.PositionsFull {
		row := make([]float32, ref.VocabSize)
		for index := range row {
			offset := 4 * (rowIndex*ref.VocabSize + index)
			row[index] = math.Float32frombits(binary.LittleEndian.Uint32(logitBytes[offset:]))
		}
		ref.fullLogits[position] = row
	}
	return &ref, nil
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

// topIndices は、logits の大きい順に count 個の語彙番号を返します。例: [0.1, 0.9, 0.5], 2 -> [1, 2]
func topIndices(logits []float32, count int) []int {
	indices := make([]int, len(logits))
	for index := range indices {
		indices[index] = index
	}
	sort.SliceStable(indices, func(a, b int) bool { return logits[indices[a]] > logits[indices[b]] })
	return indices[:count]
}

// overlap は、2 つの候補一覧に共通する要素の数を返します。
func overlap(a, b []int) int {
	seen := make(map[int]bool, len(a))
	for _, value := range a {
		seen[value] = true
	}
	count := 0
	for _, value := range b {
		if seen[value] {
			count++
		}
	}
	return count
}

// logitDifferences は、全語彙の logits の最大絶対差・RMS 差と、KL ダイバージェンス KL(基準 || 実装) を返します。
// KL は softmax を float64 で計算します。0 に近いほど、次トークンの確率分布が一致しています。
func logitDifferences(want, got []float32) (maxAbs, rms, kl float64) {
	var sumSquares float64
	for index := range want {
		difference := math.Abs(float64(want[index]) - float64(got[index]))
		maxAbs = math.Max(maxAbs, difference)
		sumSquares += difference * difference
	}
	rms = math.Sqrt(sumSquares / float64(len(want)))
	wantLog := logSoftmax(want)
	gotLog := logSoftmax(got)
	for index := range wantLog {
		probability := math.Exp(wantLog[index])
		kl += probability * (wantLog[index] - gotLog[index])
	}
	return maxAbs, rms, kl
}

// logSoftmax は log(softmax(values)) を float64 で返します。
func logSoftmax(values []float32) []float64 {
	maximum := math.Inf(-1)
	for _, value := range values {
		maximum = math.Max(maximum, float64(value))
	}
	var sum float64
	for _, value := range values {
		sum += math.Exp(float64(value) - maximum)
	}
	logSum := maximum + math.Log(sum)
	result := make([]float64, len(values))
	for index, value := range values {
		result[index] = float64(value) - logSum
	}
	return result
}

// openModel は、引数に応じて Hugging Face 形式または PQ4 の重みを読み込みます。
// 戻り値の close は、読み込んだファイルを閉じる関数です。
func openModel(opts options) (*lfm3.Model, string, func(), error) {
	if opts.modelDir != "" {
		config, source, err := lfm3.OpenSafetensorsModel(opts.modelDir)
		if err != nil {
			return nil, "", nil, err
		}
		weights, err := lfm3.LoadWeights(config, source)
		if err != nil {
			source.File.Close()
			return nil, "", nil, err
		}
		return lfm3.NewModel(weights), "safetensors-fp32", func() { source.File.Close() }, nil
	}
	config, source, err := lfm3.OpenPQ4Model(opts.pq4Path)
	if err != nil {
		return nil, "", nil, err
	}
	weights, err := lfm3.LoadWeights(config, source)
	if err != nil {
		source.File.Close()
		return nil, "", nil, err
	}
	label := "pq4-" + source.File.Header.Converter["method"]
	return lfm3.NewModel(weights), label, func() { source.File.Close() }, nil
}

// summary は全位置の比較結果の集計です。
type summary struct {
	positions     int
	top1Matches   int
	top5Overlap   int
	topKMaxAbs    float64 // 基準の上位 K 候補の logits についての最大絶対差（全位置）
	fullMaxAbs    float64
	fullMaxKL     float64
	fullMeanKL    float64
	fullPositions int
}

// run は読み込み・全トークンの処理・比較・合否判定を行います。
func run(opts options) error {
	ref, err := loadReference(opts.reference)
	if err != nil {
		return err
	}
	loadStarted := time.Now()
	model, label, closeModel, err := openModel(opts)
	if err != nil {
		return err
	}
	defer closeModel()
	loadSeconds := time.Since(loadStarted).Seconds()

	tokens := ref.TokenIDs
	if opts.maxTokens > 0 && opts.maxTokens < len(tokens) {
		tokens = tokens[:opts.maxTokens]
	}
	isFull := map[int]bool{}
	for _, position := range ref.PositionsFull {
		isFull[position] = true
	}

	var result summary
	forwardStarted := time.Now()
	for position, token := range tokens {
		logits, err := model.Forward(token)
		if err != nil {
			return err
		}
		goTop := topIndices(logits, 5)
		wantTop := ref.TopIDs[position]
		result.positions++
		if goTop[0] == wantTop[0] {
			result.top1Matches++
		}
		result.top5Overlap += overlap(goTop, wantTop[:5])
		for rank, id := range wantTop {
			result.topKMaxAbs = math.Max(result.topKMaxAbs, math.Abs(float64(logits[id])-ref.TopValues[position][rank]))
		}
		if isFull[position] {
			maxAbs, rms, kl := logitDifferences(ref.fullLogits[position], logits)
			result.fullPositions++
			result.fullMaxAbs = math.Max(result.fullMaxAbs, maxAbs)
			result.fullMaxKL = math.Max(result.fullMaxKL, kl)
			result.fullMeanKL += kl
			emit(map[string]any{"kind": "position", "position": position, "max_abs_diff": maxAbs, "rms_diff": rms,
				"kl": kl, "top1_go": goTop[0], "top1_ref": wantTop[0]})
		}
	}
	forwardSeconds := time.Since(forwardStarted).Seconds()
	if result.fullPositions > 0 {
		result.fullMeanKL /= float64(result.fullPositions)
	}

	top1Rate := float64(result.top1Matches) / float64(result.positions)
	ok := true
	var failures []string
	if opts.maxAbsDiff > 0 && result.fullMaxAbs > opts.maxAbsDiff {
		ok = false
		failures = append(failures, fmt.Sprintf("full max abs diff %.4g > %.4g", result.fullMaxAbs, opts.maxAbsDiff))
	}
	if opts.minTop1 > 0 && top1Rate < opts.minTop1 {
		ok = false
		failures = append(failures, fmt.Sprintf("top-1 agreement %.4f < %.4f", top1Rate, opts.minTop1))
	}
	emit(map[string]any{
		"kind":              "summary",
		"weights":           label,
		"positions":         result.positions,
		"top1_agreement":    top1Rate,
		"top5_overlap":      float64(result.top5Overlap) / float64(5*result.positions),
		"topk_max_abs_diff": result.topKMaxAbs,
		"full_positions":    result.fullPositions,
		"full_max_abs_diff": result.fullMaxAbs,
		"full_max_kl":       result.fullMaxKL,
		"full_mean_kl":      result.fullMeanKL,
		"load_seconds":      loadSeconds,
		"seconds_per_token": forwardSeconds / float64(result.positions),
		"ok":                ok,
		"failures":          failures,
	})
	if !ok {
		return fmt.Errorf("comparison failed: %v", failures)
	}
	return nil
}

func main() {
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
