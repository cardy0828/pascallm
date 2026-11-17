// Command lfm3-gpu は、GPU 実装（internal/lfm3gpu）の確認と速度の計測を行います（M2）。
//
// モード:
//   - check: 同じ .pq4m を CPU の参照実装と GPU 実装で 1 トークンずつ処理し、logits の差・上位 1 位の一致・
//     KL(CPU || GPU) を位置ごとに比べます。
//   - bench: BOS から貪欲法で -generate 個のトークンを生成し、1 秒あたりのトークン数を測ります。
//     CUDA Graph を再生する場合と、カーネルを 1 個ずつ起動する場合の両方を測り、生成結果が同じかも確かめます。
//   - candidates: GPU で絞った選び方の候補を、logits 全体から CPU で求めた値と比べます（candidates.go）。
//   - prefill: 入力処理（Prefill）の結果を 1 トークンずつの結果と比べ、入力処理の速さを測ります（prefill.go）。
//   - eval: 評価文の区切りごとに、FP32 の基準との KL・上位 1 位の一致・perplexity の比を測ります（eval.go、M5）。
//   - profile: BOS から -generate 個生成した後の位置で、1 トークンの GPU 時間と、カーネルの種類ごとの負担
//     （その種類だけを抜いた CUDA Graph との時間の差）を測ります。
//
// 実行例:
//
//	go run ./cmd/lfm3-gpu -mode check \
//	    -tokens ./data/kl/tokens-1024.json -positions 64
//	go run ./cmd/lfm3-gpu -mode bench -generate 128
//	go run ./cmd/lfm3-gpu -mode profile -generate 32
//
// 結果は 1 行 1 件の JSON で標準出力に書きます。確認の基準を満たさない場合は終了コード 1 になります。
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
	"os"
	"runtime"
	"time"

	"github.com/cardy0828/pascallm/internal/lfm3"
	"github.com/cardy0828/pascallm/internal/lfm3gpu"
)

// defaultDeviceUUID は空なら PASCA_DEVICE_UUID を使用します。GPUは明示的に選択します。
const defaultDeviceUUID = ""

// defaultModelPath は既定のモデルです。M5 の search + imatrix で変換したもの
// （KL 0.099、mse 版の 0.129 より小さく、生成速度は同じ。開発時の内部測定）。
const defaultModelPath = "./data/lfm3-1.7b-pq4-search-imatrix.pq4m"

// check モードの合格基準です（実装の誤りを見つけるための基準）。
//
// CPU と GPU は同じ重みで同じ手順を計算しますが、合計の順序（差は 1e-7 程度）と KV キャッシュの FP16 化
// （差は 2e-4 程度）で、わずかに値が違います。各層の入口で活性値を int8 に丸めるため、このわずかな差で
// 一部の要素の丸めが 1 段ずれ（1 段は最大値の 1/127）、24 層を通る間に広がります。
// 開発時の内部調査では、q・k・ゲートは 1e-7〜2e-4、attention の出力は
// int8 の 1 段以内で一致し、64 位置で KL(CPU || GPU) の平均は 0.0035、上位 1 位の一致率は 96.9% でした。
// KV キャッシュを FP32 にしても KL は 0.0039 で変わらなかったため、差は int8 化の丸めによるものです。
// 試しに RoPE の回転の向きを逆にした版では、KL の平均が 1.08、上位 1 位の一致率が 46.9% になり、この基準で検出できました。
const (
	minimumTopAgreement = 0.95 // 上位 1 位の一致率の下限
	maximumMeanKL       = 0.01 // KL(CPU || GPU) の平均の上限
)

// options はコマンドライン引数です。
type options struct {
	mode       string
	pq4Path    string
	tokensPath string
	positions  int
	generate   int
	warmup     int
	deviceUUID string
	maxContext int
	baseLogits string // eval で使う基準の logits（float32）
	label      string // eval の結果に付ける名前
}

// parseOptions はコマンドライン引数を読みます。
func parseOptions() (options, error) {
	var opts options
	flag.StringVar(&opts.mode, "mode", "check", "check（CPU との比較）、bench（生成速度）、profile（カーネルごとの時間）、candidates（選び方の候補の確認）、prefill（入力処理の確認と速さ）、eval（基準との品質の比較）")
	flag.StringVar(&opts.pq4Path, "pq4", defaultModelPath, "PQ4 のモデルファイル（.pq4m）")
	flag.StringVar(&opts.tokensPath, "tokens", "", "check・candidates・prefill で使うトークン列の JSON（{\"token_ids\": [...]}）。eval では区切りの JSON（{\"chunks\": ..., \"domains\": ...}）")
	flag.StringVar(&opts.baseLogits, "base-logits", "", "eval で使う基準の logits（区切りごとの後半の位置、float32）")
	flag.StringVar(&opts.label, "label", "", "eval の結果に付ける名前")
	flag.IntVar(&opts.positions, "positions", 64, "check で比べる位置の数（先頭から）")
	flag.IntVar(&opts.generate, "generate", 128, "bench で生成するトークン数")
	flag.IntVar(&opts.warmup, "warmup", 2, "bench で計測前に捨てる生成の回数")
	flag.StringVar(&opts.deviceUUID, "device-uuid", defaultDeviceUUID, "使う GPU の UUID（省略時 PASCA_DEVICE_UUID）")
	flag.IntVar(&opts.maxContext, "max-context", 4096, "KV キャッシュに入る最大の位置数")
	flag.Parse()
	switch {
	case opts.pq4Path == "":
		return options{}, errors.New("-pq4 must not be empty")
	case (opts.mode == "check" || opts.mode == "candidates" || opts.mode == "prefill" || opts.mode == "eval") &&
		opts.tokensPath == "":
		return options{}, errors.New("-tokens is required for -mode check, candidates, prefill and eval")
	case opts.mode == "eval" && opts.baseLogits == "":
		return options{}, errors.New("-base-logits is required for -mode eval")
	case opts.mode != "check" && opts.mode != "bench" && opts.mode != "profile" && opts.mode != "candidates" &&
		opts.mode != "prefill" && opts.mode != "eval":
		return options{}, fmt.Errorf("unknown -mode %q (want check, bench, profile, candidates, prefill or eval)", opts.mode)

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
	return content.TokenIDs, nil
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

// openGPU は、オプションに従って GPU 実装を開きます。
func openGPU(opts options) (*lfm3gpu.Model, error) {
	started := time.Now()
	model, err := lfm3gpu.Open(opts.pq4Path, lfm3gpu.Options{DeviceUUID: opts.deviceUUID, MaxContext: opts.maxContext})
	if err != nil {
		return nil, err
	}
	emit(map[string]any{"kind": "gpu_open", "device_uuid": opts.deviceUUID, "allocated_bytes": model.AllocatedBytes(),
		"max_context": opts.maxContext, "seconds": time.Since(started).Seconds()})
	return model, nil
}

// runCheck は、CPU の参照実装と GPU 実装の logits を先頭 opts.positions 個の位置で比べます。
func runCheck(opts options) error {
	tokens, err := loadTokens(opts.tokensPath)
	if err != nil {
		return err
	}
	positions := min(opts.positions, len(tokens))
	config, source, err := lfm3.OpenPQ4Model(opts.pq4Path)
	if err != nil {
		return err
	}
	defer source.File.Close()
	weights, err := lfm3.LoadWeights(config, source)
	if err != nil {
		return err
	}
	cpu := lfm3.NewModel(weights)
	gpu, err := openGPU(opts)
	if err != nil {
		return err
	}
	defer gpu.Close()

	vocab := config.VocabSize
	cpuLog := make([]float64, vocab)
	gpuLog := make([]float64, vocab)
	var sameTop int
	var klSum, klMax, maxAbsDiff float64
	for position := 0; position < positions; position++ {
		cpuLogits, err := cpu.Forward(tokens[position])
		if err != nil {
			return err
		}
		gpuLogits, err := gpu.Forward(tokens[position])
		if err != nil {
			return err
		}
		var positionMaxDiff float64
		for index := range cpuLogits {
			positionMaxDiff = math.Max(positionMaxDiff, math.Abs(float64(cpuLogits[index]-gpuLogits[index])))
		}
		logSoftmax(cpuLogits, cpuLog)
		logSoftmax(gpuLogits, gpuLog)
		var kl float64
		for index, value := range cpuLog {
			kl += math.Exp(value) * (value - gpuLog[index])
		}
		cpuTop, gpuTop := argmax(cpuLogits), argmax(gpuLogits)
		if cpuTop == gpuTop {
			sameTop++
		}
		klSum += kl
		klMax = math.Max(klMax, kl)
		maxAbsDiff = math.Max(maxAbsDiff, positionMaxDiff)
		emit(map[string]any{"kind": "position", "position": position, "max_abs_diff": positionMaxDiff, "kl": kl,
			"top1_cpu": cpuTop, "top1_gpu": gpuTop})
	}
	topAgreement := float64(sameTop) / float64(positions)
	meanKL := klSum / float64(positions)
	ok := topAgreement >= minimumTopAgreement && meanKL <= maximumMeanKL
	emit(map[string]any{"kind": "summary", "mode": "check", "positions": positions, "top1_agreement": topAgreement,
		"mean_kl": meanKL, "max_kl": klMax, "max_abs_diff": maxAbsDiff,
		"criteria": map[string]float64{"min_top1_agreement": minimumTopAgreement, "max_mean_kl": maximumMeanKL}, "ok": ok})
	if !ok {
		return fmt.Errorf("GPU logits differ from the CPU reference (top1 %.4f, mean KL %.3g)", topAgreement, meanKL)
	}
	return nil
}

// generateGreedy は、BOS から貪欲法で count 個のトークンを生成し、トークン列と経過時間を返します。
// 経過時間には、プロンプト（BOS）の 1 トークンの処理を含みません。
func generateGreedy(model *lfm3gpu.Model, count int) ([]int, time.Duration, error) {
	if err := model.Reset(); err != nil {
		return nil, 0, err
	}
	next, err := model.Feed(model.Config().BOSTokenID)
	if err != nil {
		return nil, 0, err
	}
	generated := []int{next}
	started := time.Now()
	for len(generated) < count {
		if next, err = model.ContinueGreedy(); err != nil {
			return nil, 0, err
		}
		generated = append(generated, next)
	}
	return generated, time.Since(started), nil
}

// equalTokens は 2 つのトークン列が同じかを返します。
func equalTokens(a, b []int) bool {
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

// runBench は、CUDA Graph あり・なしの両方で貪欲生成の速度を測ります。
func runBench(opts options) error {
	model, err := openGPU(opts)
	if err != nil {
		return err
	}
	defer model.Close()
	var reference []int
	for _, useGraph := range []bool{true, false} {
		model.SetUseGraph(useGraph)
		for round := 0; round < opts.warmup; round++ {
			if _, _, err := generateGreedy(model, opts.generate); err != nil {
				return err
			}
		}
		generated, elapsed, err := generateGreedy(model, opts.generate)
		if err != nil {
			return err
		}
		// 最初のトークンは BOS の処理で得たもので、計測時間には含まれない。
		measured := len(generated) - 1
		emit(map[string]any{"kind": "bench", "use_graph": useGraph, "generated": len(generated),
			"tokens_per_second": float64(measured) / elapsed.Seconds(),
			"ms_per_token":      elapsed.Seconds() * 1e3 / float64(measured), "first_tokens": generated[:min(16, len(generated))]})
		if reference == nil {
			reference = generated
		} else if !equalTokens(reference, generated) {
			return errors.New("generated tokens differ between graph replay and direct launches")
		}
	}
	return nil
}

// profileReplays は、profile で 1 つの Graph を続けて再生する回数です（平均を取る）。
const profileReplays = 200

// runProfile は、BOS から opts.generate 個を生成した後の位置から、1 トークンの GPU 時間と、
// カーネルの種類ごとの負担（その種類を抜いた Graph との時間の差）を測り、負担の大きい順に出力します。
func runProfile(opts options) error {
	model, err := openGPU(opts)
	if err != nil {
		return err
	}
	defer model.Close()
	if _, _, err := generateGreedy(model, opts.generate); err != nil {
		return err
	}
	full, shares, err := model.MeasureKernelShares(profileReplays)
	if err != nil {
		return err
	}
	var sum float64
	for _, share := range shares {
		sum += share.Microseconds
		emit(map[string]any{"kind": "kernel", "label": share.Label, "launches_per_token": share.PerToken,
			"us_per_token": share.Microseconds, "us_per_launch": share.Microseconds / float64(share.PerToken),
			"fraction": share.Microseconds / full})
	}
	emit(map[string]any{"kind": "profile_summary", "context_start": opts.generate, "replays": profileReplays,
		"graph_us_per_token": full, "graph_tokens_per_second": 1e6 / full, "sum_of_shares_us": sum})
	return nil
}

func main() {
	// CUDA のコンテキストは OS スレッドごとに決まるため、main のゴルーチンを 1 つのスレッドに固定する。
	runtime.LockOSThread()
	opts, err := parseOptions()
	if err == nil {
		switch opts.mode {
		case "check":
			err = runCheck(opts)
		case "bench":
			err = runBench(opts)
		case "candidates":
			err = runCandidates(opts)
		case "prefill":
			err = runPrefill(opts)
		case "eval":
			err = runEval(opts)
		default:
			err = runProfile(opts)
		}
	}
	if err != nil {
		emit(map[string]any{"kind": "result", "ok": false, "error": err.Error()})
		os.Exit(1)
	}
	emit(map[string]any{"kind": "result", "ok": true})
}
