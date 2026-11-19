// Command gemma4-gpu は、Gemma 4 の GPU 実装（internal/gemma4gpu）の確認と計測を行います（M7）。
//
//	-mode eval     llama.cpp の基準（llama-perplexity --kl-divergence-base のファイル）と次トークンの分布を比べる。
//	               -base-logits に CPU の参照実装が保存した logits（gemma4-ref -save-logits）を渡すと、それとも比べる。
//	               -prefill-tokens K で、各区切りの最初の K 個を入力処理（Prefill）で入れる（入力処理の確認）
//	-mode check    最初の -context 個の位置で、CPU の参照実装（int8 の活性値）と logits を位置ごとに比べる
//	-mode layers   位置 -context − 1 のトークンについて、各層の後の hidden を CPU の参照実装と比べる（誤りのある層を探す）
//	-mode paths    最初の -context 個のトークンについて、浅い層（-layers 層まで）の hidden を、入力処理の経路と
//	               生成の経路で比べる（どちらも GPU。丸めの増幅が小さい浅い層で、経路の違いによる誤りを探す）
//	-mode bench    -context 個のトークンを入れた後、貪欲法で -generate 個を生成する速さを測る
//	-mode prefill  入力処理の速さを測る（-context 個を 1 回で入れる時間）
//	-mode profile  1 トークンと入力処理 1 区切りの、カーネルの種類ごとの GPU 時間を出す
//
// 結果は 1 行 1 件の JSON です。eval は、基準を満たさないと終了コード 1 になります。
//
// 実行例:
//
//	gemma4-gpu -mode eval -kld ./data/m7/gemma12b-c128.kld -chunks 4
//	gemma4-gpu -mode bench -kld ./data/m7/gemma12b-c4096.kld -context 16 -generate 128
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"runtime"
	"sort"
	"time"
	"unsafe"

	"github.com/cardy0828/pascallm/internal/compare"
	"github.com/cardy0828/pascallm/internal/gemma4"
	"github.com/cardy0828/pascallm/internal/gemma4gpu"
	"github.com/cardy0828/pascallm/internal/llamakld"
)

// 既定値と、eval の合格の基準です。
const (
	defaultDeviceUUID = ""
	defaultPQ4        = "./data/gemma4-12b-qat.pq4m"
	// llama.cpp との比較の基準です。どちらも活性値を int8 にするため完全には一致せず、CPU の参照実装（int8）と
	// llama.cpp の差（KL 平均 0.03、上位 1 位一致 95%）と同じ程度になります。
	minimumTop1Agreement = 0.90
	maximumMeanKL        = 0.10
	float32Bytes         = 4
)

// options はコマンドライン引数です。
type options struct {
	mtpIntegrated bool
	compactVRAM   bool
	prefillChunk  int
	kvCache       string
	mtpDraft      int
	mtpAdaptive   bool
	mtpPath       string
	mode          string
	pq4Path       string
	deviceUUID    string
	maxContext    int
	kldPath       string
	chunks        int
	baseLogits    string
	prefillTokens int
	context       int
	generate      int
	repeats       int
	useGraph      bool
	layers        int
}

// parseOptions はコマンドライン引数を読みます。
func parseOptions() (options, error) {
	var opts options
	flag.StringVar(&opts.mode, "mode", "eval", "eval・bench・prefill・profile のどれか")
	flag.StringVar(&opts.pq4Path, "pq4", defaultPQ4, "Gemma 4 の .pq4m")
	flag.StringVar(&opts.deviceUUID, "device-uuid", defaultDeviceUUID, "使う GPU の UUID（省略時 PASCA_DEVICE_UUID）")
	flag.BoolVar(&opts.mtpIntegrated, "mtp-integrated", false, "固定3候補・FP16 KV の貪欲 MTP を1本の CUDA Graph で実行する")
	flag.BoolVar(&opts.compactVRAM, "compact-vram", false, "Gemma4 専用。重みと KV を一括確保し、保存位置の復元幅を入力チャンクに合わせる")
	flag.IntVar(&opts.prefillChunk, "prefill-chunk", 0, "Gemma4 の入力を区切るトークン数（1〜1024、0 は既定）")
	flag.IntVar(&opts.maxContext, "max-context", 8192, "扱える最大の位置数")
	flag.StringVar(&opts.kldPath, "kld", "", "llama-perplexity --kl-divergence-base のファイル（トークン列と基準）")
	flag.IntVar(&opts.chunks, "chunks", 1, "eval で比べる区切りの数")
	flag.StringVar(&opts.baseLogits, "base-logits", "", "eval: CPU の参照実装の logits（区切り 0 の分。省略可）")
	flag.IntVar(&opts.prefillTokens, "prefill-tokens", 0, "eval: 各区切りの最初の何個を Prefill で入れるか")
	flag.IntVar(&opts.context, "context", 16, "bench・prefill・profile: 先に入れるトークン数")
	flag.IntVar(&opts.generate, "generate", 128, "bench: 生成するトークン数")
	flag.IntVar(&opts.repeats, "repeats", 3, "bench・prefill: 繰り返す回数")
	flag.IntVar(&opts.layers, "layers", 8, "paths: 比べる層の数の上限")
	flag.BoolVar(&opts.useGraph, "graph", true, "CUDA Graph を使う（false ならカーネルを 1 個ずつ起動する）")
	flag.StringVar(&opts.kvCache, "kv-cache", "fp16", "Gemma4 KV: fp16、k-fp32、fp32（診断用）")
	flag.StringVar(&opts.mtpPath, "mtp", "", "Gemma4 12B assistant Q4_0 GGUF（貪欲生成の MTP、空なら無効）")
	flag.IntVar(&opts.mtpDraft, "mtp-draft", 3, "MTP 候補数 1..7（自動調整時は上限）")
	flag.BoolVar(&opts.mtpAdaptive, "mtp-auto", false, "MTP の候補数を実測時間と受理数から調整")
	flag.Parse()
	if opts.kldPath == "" {
		return opts, fmt.Errorf("-kld は必須です（トークン列を読むため）")
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

// chunkInput は、基準ファイルの区切り chunk のトークン列を、llama-perplexity と同じく先頭を BOS にして返します。
func chunkInput(base *llamakld.File, chunk, bos int) []int {
	tokens := base.ChunkTokens(chunk)
	tokens[0] = bos
	return tokens
}

// loadSavedLogits は、gemma4-ref -save-logits のファイルを読みます（[位置][語彙] の float32）。
func loadSavedLogits(path string, vocab int) ([][]float32, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	rowBytes := float32Bytes * vocab
	if len(raw) == 0 || len(raw)%rowBytes != 0 {
		return nil, fmt.Errorf("%s has %d bytes, which is not a multiple of %d", path, len(raw), rowBytes)
	}
	values := unsafe.Slice((*float32)(unsafe.Pointer(&raw[0])), len(raw)/float32Bytes)
	rows := make([][]float32, len(raw)/rowBytes)
	for index := range rows {
		rows[index] = values[index*vocab : (index+1)*vocab]
	}
	return rows, nil
}

// runEval は、基準ファイルの各区切りを GPU で処理し、llama.cpp（と、あれば CPU の参照実装）と比べます。
func runEval(model *gemma4gpu.Model, base *llamakld.File, opts options) error {
	config := model.Config()
	if base.VocabSize != config.VocabSize {
		return fmt.Errorf("the base file has %d tokens in the vocabulary, the model has %d", base.VocabSize, config.VocabSize)
	}
	if opts.prefillTokens < 0 || opts.prefillTokens > base.FirstScored() {
		return fmt.Errorf("-prefill-tokens must be between 0 and %d", base.FirstScored())
	}
	var saved [][]float32
	if opts.baseLogits != "" {
		var err error
		if saved, err = loadSavedLogits(opts.baseLogits, config.VocabSize); err != nil {
			return err
		}
	}
	chunks := min(opts.chunks, base.Chunks)
	baseLogProbs := make([]float32, base.VocabSize)
	savedLogProbs := make([]float32, base.VocabSize)
	savedLogProbs64 := make([]float64, base.VocabSize)
	var againstLlama, againstReference compare.Accumulator
	started := time.Now()
	for chunk := 0; chunk < chunks; chunk++ {
		tokens := chunkInput(base, chunk, config.BOSTokenID)
		if err := model.Reset(); err != nil {
			return err
		}
		if err := model.Prefill(tokens[:opts.prefillTokens]); err != nil {
			return err
		}
		for position := opts.prefillTokens; position < base.ContextSize-1; position++ {
			logits, err := model.Forward(tokens[position])
			if err != nil {
				return err
			}
			if position < base.FirstScored() {
				continue
			}
			floor, err := base.LogProbs(chunk, position, baseLogProbs)
			if err != nil {
				return err
			}
			againstLlama.Add(baseLogProbs, floor, logits, tokens[position+1])
			savedIndex := chunk*base.ScoredPerChunk() + position - base.FirstScored()
			if savedIndex < len(saved) {
				compare.LogSoftmax(saved[savedIndex], savedLogProbs64)
				for token, value := range savedLogProbs64 {
					savedLogProbs[token] = float32(value)
				}
				// CPU の参照実装の logits には切り上げがないので、下限は負の無限大にします。
				againstReference.Add(savedLogProbs, float32(math.Inf(-1)), logits, tokens[position+1])
			}
		}
	}
	llama := againstLlama.Summary()
	record := map[string]any{"event": "eval", "chunks": chunks, "context": base.ContextSize, "prefill_tokens": opts.prefillTokens,
		"against_llama": llama, "seconds": time.Since(started).Seconds()}
	if len(saved) > 0 {
		record["against_reference"] = againstReference.Summary()
	}
	passed := llama.Top1Agreement >= minimumTop1Agreement && llama.KLMean <= maximumMeanKL
	record["passed"] = passed
	emit(record)
	if !passed {
		return fmt.Errorf("eval failed: top-1 agreement %.3f (need %.2f), mean KL %.4f (limit %.2f)",
			llama.Top1Agreement, minimumTop1Agreement, llama.KLMean, maximumMeanKL)
	}
	return nil
}

// runCheck は、最初の -context 個の位置で、GPU の logits を CPU の参照実装（int8 の活性値）と位置ごとに比べます。
// 位置ごとに、logits の差の絶対値の最大、上位 1 位が同じか、KL(CPU || GPU) を出します（誤りの切り分け用）。
// CPU の参照実装は 1 トークン 1 秒以上かかるため、-context は小さくしてください。
func runCheck(model *gemma4gpu.Model, base *llamakld.File, opts options) error {
	tokens, err := contextTokens(base, model.Config().BOSTokenID, opts.context)
	if err != nil {
		return err
	}
	weights, err := gemma4.OpenWeights(opts.pq4Path)
	if err != nil {
		return err
	}
	defer weights.Close()
	reference := gemma4.NewModel(weights, gemma4.ActivationsInt8)
	if err := model.Reset(); err != nil {
		return err
	}
	vocab := model.Config().VocabSize
	referenceLogProbs := make([]float64, vocab)
	referenceLogProbs32 := make([]float32, vocab)
	for position, token := range tokens {
		want, err := reference.Forward(token)
		if err != nil {
			return err
		}
		got, err := model.Forward(token)
		if err != nil {
			return err
		}
		var maxDiff float64
		wantTop, gotTop := 0, 0
		for index := range want {
			maxDiff = math.Max(maxDiff, math.Abs(float64(want[index])-float64(got[index])))
			if want[index] > want[wantTop] {
				wantTop = index
			}
			if got[index] > got[gotTop] {
				gotTop = index
			}
		}
		compare.LogSoftmax(want, referenceLogProbs)
		for index, value := range referenceLogProbs {
			referenceLogProbs32[index] = float32(value)
		}
		var single compare.Accumulator
		single.Add(referenceLogProbs32, float32(math.Inf(-1)), got, wantTop)
		emit(map[string]any{"event": "check", "position": position, "max_logit_diff": maxDiff, "same_top": wantTop == gotTop,
			"reference_top_logit": want[wantTop], "kl": single.Summary().KLMean})
	}
	return nil
}

// relativeDifference は、2 つのベクトルの差の RMS を、want の RMS で割った値を返します。
// 例: want = [1, 1]、got = [1.1, 0.9] -> 0.1
func relativeDifference(want, got []float32) float64 {
	var differenceSquares, wantSquares float64
	for index := range want {
		difference := float64(want[index]) - float64(got[index])
		differenceSquares += difference * difference
		wantSquares += float64(want[index]) * float64(want[index])
	}
	if wantSquares == 0 {
		return math.Sqrt(differenceSquares)
	}
	return math.Sqrt(differenceSquares / wantSquares)
}

// runLayers は、-context 個のトークンのうち最後の 1 個について、埋め込みの後と各層の後の hidden を、
// CPU の参照実装（int8 の活性値）と GPU で比べます。それより前のトークンは、両方に普通に入れます。
// 差が急に大きくなる層があれば、その層の計算に違いがあります。
func runLayers(model *gemma4gpu.Model, base *llamakld.File, opts options) error {
	tokens, err := contextTokens(base, model.Config().BOSTokenID, opts.context)
	if err != nil {
		return err
	}
	weights, err := gemma4.OpenWeights(opts.pq4Path)
	if err != nil {
		return err
	}
	defer weights.Close()
	reference := gemma4.NewModel(weights, gemma4.ActivationsInt8)
	if err := model.Reset(); err != nil {
		return err
	}
	last := len(tokens) - 1
	for _, token := range tokens[:last] {
		if _, err := reference.Forward(token); err != nil {
			return err
		}
		if _, err := model.Forward(token); err != nil {
			return err
		}
	}
	var wanted [][]float32
	reference.Trace = func(_ int, hidden []float32) { wanted = append(wanted, append([]float32(nil), hidden...)) }
	if _, err := reference.Forward(tokens[last]); err != nil {
		return err
	}
	for layers, want := range wanted {
		got, err := model.DebugHidden(tokens[last], layers)
		if err != nil {
			return err
		}
		emit(map[string]any{"event": "layer", "position": last, "layers": layers, "relative_difference": relativeDifference(want, got)})
	}
	return nil
}

// runPaths は、-context 個のトークンについて、最初の 0〜-layers 層の後の hidden を、入力処理（1 区切りでまとめて）と
// 生成（1 個ずつ）の 2 つの経路で求めて比べます。層の数ごとに、差（生成の経路の RMS に対する比）の最大と、
// それが出たトークンの位置、差の中央値を出します。2 つの経路は同じ int8 化と FP16 を使うため、浅い層ではほぼ一致するはずです。
func runPaths(model *gemma4gpu.Model, base *llamakld.File, opts options) error {
	tokens, err := contextTokens(base, model.Config().BOSTokenID, opts.context)
	if err != nil {
		return err
	}
	hiddenSize := model.Config().HiddenSize
	for layers := 0; layers <= opts.layers; layers++ {
		if err := model.Reset(); err != nil {
			return err
		}
		batched, err := model.DebugPrefillHidden(tokens, layers)
		if err != nil {
			return err
		}
		batched = append([]float32(nil), batched...)
		if err := model.Reset(); err != nil {
			return err
		}
		var worst float64
		worstPosition := 0
		differences := make([]float64, 0, len(tokens))
		for position, token := range tokens {
			single, err := model.DebugHidden(token, layers)
			if err != nil {
				return err
			}
			difference := relativeDifference(single, batched[position*hiddenSize:(position+1)*hiddenSize])
			differences = append(differences, difference)
			if difference > worst {
				worst, worstPosition = difference, position
			}
			// 全層を処理して位置を進める（次の位置が見る KV キャッシュを、全層について書く）。
			if _, err := model.Forward(token); err != nil {
				return err
			}
		}
		emit(map[string]any{"event": "paths", "layers": layers, "tokens": len(tokens), "max_relative_difference": worst,
			"worst_position": worstPosition, "median_relative_difference": median(differences)})
	}
	return nil
}

// contextTokens は、基準ファイルの区切り 0 から、先頭を BOS にした count 個のトークンを返します。
func contextTokens(base *llamakld.File, bos, count int) ([]int, error) {
	if count < 1 || count > base.ContextSize {
		return nil, fmt.Errorf("-context must be between 1 and %d", base.ContextSize)
	}
	return chunkInput(base, 0, bos)[:count], nil
}

// median は中央値を返します。例: median([3, 1, 2]) = 2
func median(values []float64) float64 {
	sorted := append([]float64(nil), values...)
	sort.Float64s(sorted)
	return sorted[len(sorted)/2]
}

// runBench は、-context 個のトークンを入れた後、貪欲法で -generate 個を生成する速さを測ります。
// 最初のトークンは Feed（ホストから入力）、以後は ContinueGreedy（GPU 側で argmax を次の入力にする）です。
func runBench(model *gemma4gpu.Model, base *llamakld.File, opts options) error {
	tokens, err := contextTokens(base, model.Config().BOSTokenID, opts.context)
	if err != nil {
		return err
	}
	var speeds []float64
	for repeat := 0; repeat < opts.repeats; repeat++ {
		if err := model.Reset(); err != nil {
			return err
		}
		if err := model.Prefill(tokens[:len(tokens)-1]); err != nil {
			return err
		}
		if _, err := model.Feed(tokens[len(tokens)-1]); err != nil {
			return err
		}
		started := time.Now()
		for step := 0; step < opts.generate; step++ {
			if _, err := model.ContinueGreedy(); err != nil {
				return err
			}
		}
		speeds = append(speeds, float64(opts.generate)/time.Since(started).Seconds())
	}
	emit(map[string]any{"event": "bench", "context": opts.context, "generate": opts.generate, "graph": opts.useGraph,
		"tokens_per_second": speeds, "median_tokens_per_second": median(speeds),
		"gpu_bytes": model.AllocatedBytes()})
	return nil
}

// runPrefill は、-context 個のトークンを Prefill で入れる速さを測ります（最初の 1 回は準備として捨てる）。
func runPrefill(model *gemma4gpu.Model, base *llamakld.File, opts options) error {
	tokens, err := contextTokens(base, model.Config().BOSTokenID, opts.context)
	if err != nil {
		return err
	}
	var speeds []float64
	for repeat := 0; repeat <= opts.repeats; repeat++ {
		if err := model.Reset(); err != nil {
			return err
		}
		started := time.Now()
		if err := model.Prefill(tokens); err != nil {
			return err
		}
		if repeat > 0 {
			speeds = append(speeds, float64(len(tokens))/time.Since(started).Seconds())
		}
	}
	emit(map[string]any{"event": "prefill", "tokens": len(tokens), "tokens_per_second": speeds,
		"median_tokens_per_second": median(speeds)})
	return nil
}

// runProfile は、カーネルの種類ごとの GPU 時間を出します。-context 個を入れた後の 1 トークンと、
// 入力処理 1 区切り（-prefill-chunk の行数まで）の 2 つです。
func runProfile(model *gemma4gpu.Model, base *llamakld.File, opts options) error {
	tokens, err := contextTokens(base, model.Config().BOSTokenID, opts.context)
	if err != nil {
		return err
	}
	if err := model.Reset(); err != nil {
		return err
	}
	if err := model.Prefill(tokens[:len(tokens)-1]); err != nil {
		return err
	}
	// 1 回目はクロックが上がる前のことがあるため、同じ位置で 2 回測り、2 回目を出します。
	var microseconds map[string]float64
	var launches map[string]int
	for attempt := 0; attempt < 2; attempt++ {
		if attempt > 0 {
			if err := model.Reset(); err != nil {
				return err
			}
			if err := model.Prefill(tokens[:len(tokens)-1]); err != nil {
				return err
			}
		}
		if microseconds, launches, err = model.MeasureToken(tokens[len(tokens)-1]); err != nil {
			return err
		}
	}
	emit(map[string]any{"event": "profile_token", "context": opts.context, "microseconds": microseconds, "launches": launches})

	chunk := tokens[:min(len(tokens), model.PrefillChunkSize())]
	for attempt := 0; attempt < 2; attempt++ {
		if err := model.Reset(); err != nil {
			return err
		}
		if microseconds, launches, err = model.MeasurePrefill(chunk); err != nil {
			return err
		}
	}
	emit(map[string]any{"event": "profile_prefill", "tokens": len(chunk), "microseconds": microseconds, "launches": launches})
	return nil
}

// run は、モデルを開いて指定の処理を行います。
func run(opts options) error {
	runtime.LockOSThread() // CUDA のコンテキストは OS スレッドごとに決まる
	defer runtime.UnlockOSThread()
	base, err := llamakld.Open(opts.kldPath)
	if err != nil {
		return err
	}
	defer base.Close()
	started := time.Now()
	model, err := gemma4gpu.Open(opts.pq4Path, gemma4gpu.Options{
		MTPIntegrated:  opts.mtpIntegrated,
		CompactVRAM:    opts.compactVRAM,
		PrefillChunk:   opts.prefillChunk,
		KVCache:        opts.kvCache,
		DeviceUUID:     opts.deviceUUID,
		MaxContext:     opts.maxContext,
		MTPPath:        opts.mtpPath,
		MTPDraftTokens: opts.mtpDraft,
		MTPAdaptive:    opts.mtpAdaptive,
	})
	if err != nil {
		return err
	}
	defer model.Close()
	if opts.mtpPath != "" {
		defer func() { emit(map[string]any{"event": "mtp_stats", "stats": model.MTPStats()}) }()
	}
	model.SetUseGraph(opts.useGraph)
	emit(map[string]any{"event": "open", "seconds": time.Since(started).Seconds(), "gpu_bytes": model.AllocatedBytes(),
		"max_context": opts.maxContext, "kv_cache": model.KVCacheFormat()})
	switch opts.mode {
	case "mtp-check":
		return runMTPCheck(model, base, opts)
	case "eval":
		return runEval(model, base, opts)
	case "check":
		return runCheck(model, base, opts)
	case "layers":
		return runLayers(model, base, opts)
	case "paths":
		return runPaths(model, base, opts)
	case "bench":
		return runBench(model, base, opts)
	case "prefill":
		return runPrefill(model, base, opts)
	case "profile":
		return runProfile(model, base, opts)
	default:
		return fmt.Errorf("unknown -mode %q", opts.mode)
	}
}

func main() {
	opts, err := parseOptions()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	if err := run(opts); err != nil {
		fmt.Fprintln(os.Stderr, "gemma4-gpu:", err)
		os.Exit(1)
	}
}

// runMTPCheck は一括検証の全 logits、貪欲出力列、位置、入力分岐、復元を通常生成と比較します。
// 例: -mode mtp-check -context 1024 -generate 128 -mtp assistant.gguf。
func runMTPCheck(m *gemma4gpu.Model, base *llamakld.File, opts options) error {
	if opts.mtpPath == "" {
		return fmt.Errorf("-mtp is required")
	}
	tokens, err := contextTokens(base, m.Config().BOSTokenID, opts.context)
	if err != nil {
		return err
	}
	prepare := func(enabled bool) error {
		if err := m.Reset(); err != nil {
			return err
		}
		if err := m.SetMTPEnabled(enabled); err != nil {
			return err
		}
		return m.Prefill(tokens[:len(tokens)-1])
	}
	if err = prepare(false); err != nil {
		return err
	}
	inputs := []int{tokens[len(tokens)-1], 2, 100, 200, 300, 400, 500, 600}
	inputs = inputs[:min(opts.mtpDraft+1, m.MaxContext()-m.Position())]
	batched, err := m.DebugMTPVerify(inputs)
	if err != nil {
		return err
	}
	if err = prepare(false); err != nil {
		return err
	}
	maxDiff := float64(0)
	different := 0
	for row, token := range inputs {
		logits, err := m.Forward(token)
		if err != nil {
			return err
		}
		for id, value := range logits {
			diff := math.Abs(float64(value - batched[row][id]))
			maxDiff = math.Max(maxDiff, diff)
			if diff != 0 {
				different++
			}
		}
	}
	emit(map[string]any{"event": "mtp_verify_equivalence", "context": opts.context, "max_absolute_difference": maxDiff, "different_logits": different})
	if maxDiff > 0.00001 {
		return fmt.Errorf("MTP verification differs from single-token target: %g", maxDiff)
	}
	var want []int
	for _, enabled := range []bool{false, true} {
		if err = prepare(enabled); err != nil {
			return err
		}
		current := tokens[len(tokens)-1]
		for i := 0; i < opts.generate; i++ {
			next, err := m.Feed(current)
			if err != nil {
				return err
			}
			if m.Position() != len(tokens)+i {
				return fmt.Errorf("MTP logical position mismatch")
			}
			if !enabled {
				want = append(want, next)
			} else if next != want[i] {
				return fmt.Errorf("MTP token %d got %d want %d", i, next, want[i])
			}
			current = next
		}
	}
	// 意図的に異なる入力へ分岐し、先読みを破棄した後も正しい出力を返すか確認します。
	var branchWant []int
	for _, enabled := range []bool{false, true} {
		if err = prepare(enabled); err != nil {
			return err
		}
		current := tokens[len(tokens)-1]
		for i := 0; i < min(16, m.MaxContext()-len(tokens)+1); i++ {
			next, err := m.Feed(current)
			if err != nil {
				return err
			}
			if !enabled {
				branchWant = append(branchWant, next)
			} else if next != branchWant[i] {
				return fmt.Errorf("MTP branch token %d differs", i)
			}
			current = next
			if i%3 == 1 {
				current = inputs[(i/3)%len(inputs)]
			}
		}
	}
	if err = prepare(true); err != nil {
		return err
	}
	if err = m.SaveState(); err != nil {
		return err
	}
	first, err := m.Feed(tokens[len(tokens)-1])
	if err != nil {
		return err
	}
	if _, err = m.Feed(first); err != nil {
		return err
	}
	if err = m.RestoreState(); err != nil {
		return err
	}
	restored, err := m.Feed(tokens[len(tokens)-1])
	if err != nil {
		return err
	}
	if restored != first {
		return fmt.Errorf("MTP restore result differs")
	}
	// 先読みが残る途中で MTP を切り替えても、通常 ContinueGreedy の入力を失わない。
	if err = prepare(true); err != nil {
		return err
	}
	current, err := m.Feed(tokens[len(tokens)-1])
	if err != nil {
		return err
	}
	current, err = m.Feed(current)
	if err != nil {
		return err
	}
	if err = m.SetMTPEnabled(false); err != nil {
		return err
	}
	switched, err := m.ContinueGreedy()
	if err != nil {
		return err
	}
	if err = prepare(false); err != nil {
		return err
	}
	current, err = m.Feed(tokens[len(tokens)-1])
	if err != nil {
		return err
	}
	current, err = m.Feed(current)
	if err != nil {
		return err
	}
	expected, err := m.ContinueGreedy()
	if err != nil {
		return err
	}
	if switched != expected {
		return fmt.Errorf("MTP toggle result differs")
	}
	emit(map[string]any{"event": "mtp_check", "context": opts.context, "tokens": opts.generate, "passed": true, "stats": m.MTPStats()})
	return nil
}
