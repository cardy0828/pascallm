package main

import (
	"fmt"
	"math"
	"slices"
	"sort"
	"time"

	"github.com/cardy0828/pascallm/internal/lfm3gpu"
)

// この file は -mode prefill です。入力処理（lfm3gpu.Prefill）の結果を 1 トークンずつ Forward した結果と比べ、
// 入力処理の速さとカーネルの種類ごとの時間を測ります。
//
// 比べ方: tokens の先頭 n 個を Prefill で入れてから tokens[n] を Forward した logits と、先頭 n+1 個を 1 個ずつ
// Forward したときの最後の logits を比べます（KL・上位 1 位・最大差）。区切り（512 トークン）の境目の前後と、
// Prefill を 2 回に分けた場合を含めます。

// prefillCheckLengths は、比べるプロンプトの長さ n です（区切りの境目 512 の前後を含む）。
var prefillCheckLengths = []int{1, 2, 7, 31, 32, 63, 64, 65, 100, 255, 256, 300, 511, 512, 513, 700, 1000, 1023}

// prefillSplitCases は、Prefill を 2 回に分ける場合です（[分ける位置, n]）。2 回目の位置の続きと履歴を確かめます。
var prefillSplitCases = [][2]int{{5, 64}, {250, 700}, {512, 1023}}

// prefillRestoreCases は、分ける位置で SaveState し、別のトークン列（tokens の後ろの部分）を入れて KV キャッシュと
// 履歴を書き換えてから RestoreState し、続きを入れる場合です（[分ける位置, n]）。保存と復元を確かめます。
var prefillRestoreCases = [][2]int{{5, 64}, {300, 700}}

// prefillRestoreDirtyTokens は、復元の確認で、保存の後に入れる別のトークン列の長さです。
const prefillRestoreDirtyTokens = 200

// prefillSpeedLengths は、速さを測るトークン数です（評価文のトークン列を繰り返して作る）。
var prefillSpeedLengths = []int{128, 512, 1024, 2048, 4096}

// 速さの測定の回数です。
const (
	prefillSpeedWarmups = 3 // 測る前に捨てる回数（クロックを上げるため。512 トークンを処理する）
	prefillSpeedRepeats = 3 // 長さごとに測る回数（最も速い値と中央値を出す）
)

// prefillCase は、比べる 1 件（分ける位置 split（0 なら分けない）と長さ length）です。
// restore が true なら、split で状態を保存し、別のトークン列を入れてから保存した状態に戻します。
type prefillCase struct {
	split   int
	length  int
	restore bool
}

// compareLogits は、want と got の KL(want || got)・上位 1 位が同じか・最大の差を返します。
// wantLog・gotLog は作業領域（長さ vocab）です。
func compareLogits(want, got []float32, wantLog, gotLog []float64) (float64, bool, float64) {
	var maxDiff float64
	for index := range want {
		maxDiff = math.Max(maxDiff, math.Abs(float64(want[index]-got[index])))
	}
	logSoftmax(want, wantLog)
	logSoftmax(got, gotLog)
	var kl float64
	for index, value := range wantLog {
		kl += math.Exp(value) * (value - gotLog[index])
	}
	return kl, argmax(want) == argmax(got), maxDiff
}

// runPrefill は、入力処理の正しさを確かめ、速さとカーネルごとの時間を測ります。
func runPrefill(opts options) error {
	tokens, err := loadTokens(opts.tokensPath)
	if err != nil {
		return err
	}
	model, err := openGPU(opts)
	if err != nil {
		return err
	}
	defer model.Close()

	ok, err := checkPrefill(model, tokens)
	if err != nil {
		return err
	}
	if err := measurePrefillSpeed(model, tokens, opts.maxContext); err != nil {
		return err
	}
	if err := measurePrefillKernels(model, tokens); err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("prefill logits differ from the token-by-token results")
	}
	return nil
}

// checkPrefill は、prefillCheckLengths と prefillSplitCases の全件で、Prefill の後の logits を 1 個ずつの結果と比べます。
func checkPrefill(model *lfm3gpu.Model, tokens []int) (bool, error) {
	var cases []prefillCase
	for _, length := range prefillCheckLengths {
		cases = append(cases, prefillCase{length: length})
	}
	for _, split := range prefillSplitCases {
		cases = append(cases, prefillCase{split: split[0], length: split[1]})
	}
	for _, split := range prefillRestoreCases {
		cases = append(cases, prefillCase{split: split[0], length: split[1], restore: true})
	}
	longest := 0
	for _, c := range cases {
		longest = max(longest, c.length)
	}
	if longest >= len(tokens) {
		return false, fmt.Errorf("the token file has %d tokens; prefill check needs %d", len(tokens), longest+1)
	}

	// 1 個ずつ Forward し、必要な長さの logits を残す（位置 n の logits は、先頭 n+1 個を入れた後のもの）。
	wanted := make(map[int][]float32)
	if err := model.Reset(); err != nil {
		return false, err
	}
	for position := 0; position <= longest; position++ {
		logits, err := model.Forward(tokens[position])
		if err != nil {
			return false, err
		}
		for _, c := range cases {
			if c.length == position {
				wanted[position] = slices.Clone(logits)
			}
		}
	}

	vocab := model.Config().VocabSize
	wantLog := make([]float64, vocab)
	gotLog := make([]float64, vocab)
	var klSum, klMax, maxDiff float64
	sameTop := 0
	for _, c := range cases {
		if err := model.Reset(); err != nil {
			return false, err
		}
		if err := model.Prefill(tokens[:c.split]); err != nil {
			return false, err
		}
		if c.restore {
			if err := saveDirtyRestore(model, tokens); err != nil {
				return false, err
			}
		}
		if err := model.Prefill(tokens[c.split:c.length]); err != nil {
			return false, err
		}
		logits, err := model.Forward(tokens[c.length])
		if err != nil {
			return false, err
		}
		kl, top, diff := compareLogits(wanted[c.length], logits, wantLog, gotLog)
		if top {
			sameTop++
		}
		klSum += kl
		klMax = math.Max(klMax, kl)
		maxDiff = math.Max(maxDiff, diff)
		emit(map[string]any{"kind": "prefill_case", "split": c.split, "length": c.length, "restore": c.restore, "kl": kl,
			"same_top1": top, "max_abs_diff": diff})
	}
	topAgreement := float64(sameTop) / float64(len(cases))
	meanKL := klSum / float64(len(cases))
	ok := topAgreement >= minimumTopAgreement && meanKL <= maximumMeanKL
	emit(map[string]any{"kind": "prefill_summary", "cases": len(cases), "top1_agreement": topAgreement, "mean_kl": meanKL,
		"max_kl": klMax, "max_abs_diff": maxDiff,
		"criteria": map[string]float64{"min_top1_agreement": minimumTopAgreement, "max_mean_kl": maximumMeanKL}, "ok": ok})
	return ok, nil
}

// saveDirtyRestore は、現在の位置で SaveState し、tokens の末尾の prefillRestoreDirtyTokens 個を Prefill して
// 1 個 Forward（KV キャッシュと shortconv の履歴を書き換える）してから、RestoreState で保存した位置に戻します。
func saveDirtyRestore(model *lfm3gpu.Model, tokens []int) error {
	saved := model.Position()
	if err := model.SaveState(); err != nil {
		return err
	}
	dirty := tokens[len(tokens)-prefillRestoreDirtyTokens:]
	if err := model.Prefill(dirty[:len(dirty)-1]); err != nil {
		return err
	}
	if _, err := model.Forward(dirty[len(dirty)-1]); err != nil {
		return err
	}
	if err := model.RestoreState(); err != nil {
		return err
	}
	if model.Position() != saved {
		return fmt.Errorf("RestoreState moved to position %d; want %d", model.Position(), saved)
	}
	return nil
}

// repeatTokens は、tokens を繰り返して長さ length の列を作ります。例: repeatTokens([1 2 3], 5) → [1 2 3 1 2]
func repeatTokens(tokens []int, length int) []int {
	out := make([]int, length)
	for index := range out {
		out[index] = tokens[index%len(tokens)]
	}
	return out
}

// measurePrefillSpeed は、prefillSpeedLengths の各長さで、位置 0 から Prefill する時間を測ります。
func measurePrefillSpeed(model *lfm3gpu.Model, tokens []int, maxContext int) error {
	warmup := repeatTokens(tokens, lfm3gpu.PrefillChunkTokens)
	for range prefillSpeedWarmups {
		if err := model.Reset(); err != nil {
			return err
		}
		if err := model.Prefill(warmup); err != nil {
			return err
		}
	}
	for _, length := range prefillSpeedLengths {
		if length > maxContext {
			continue
		}
		sequence := repeatTokens(tokens, length)
		var rates []float64
		for range prefillSpeedRepeats {
			if err := model.Reset(); err != nil {
				return err
			}
			started := time.Now()
			if err := model.Prefill(sequence); err != nil {
				return err
			}
			rates = append(rates, float64(length)/time.Since(started).Seconds())
		}
		sort.Float64s(rates)
		emit(map[string]any{"kind": "prefill_speed", "tokens": length, "tokens_per_second_best": rates[len(rates)-1],
			"tokens_per_second_median": rates[len(rates)/2], "repeats": len(rates)})
	}
	return nil
}

// measurePrefillKernels は、位置 0 からの 512 トークンの 1 区切りで、カーネルの種類ごとの GPU 時間を測ります。
func measurePrefillKernels(model *lfm3gpu.Model, tokens []int) error {
	if err := model.Reset(); err != nil {
		return err
	}
	times, err := model.MeasurePrefill(repeatTokens(tokens, lfm3gpu.PrefillChunkTokens))
	if err != nil {
		return err
	}
	labels := make([]string, 0, len(times))
	total := 0.0
	for label, microseconds := range times {
		labels = append(labels, label)
		total += microseconds
	}
	sort.Slice(labels, func(a, b int) bool { return times[labels[a]] > times[labels[b]] })
	for _, label := range labels {
		emit(map[string]any{"kind": "prefill_kernel", "label": label, "us": times[label], "fraction": times[label] / total})
	}
	const microsecondsPerSecond = 1e6
	emit(map[string]any{"kind": "prefill_kernel_summary", "tokens": lfm3gpu.PrefillChunkTokens, "sum_us": total,
		"tokens_per_second_from_sum": float64(lfm3gpu.PrefillChunkTokens) / (total / microsecondsPerSecond)})
	return model.Reset()
}
