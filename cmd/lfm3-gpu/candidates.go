package main

import (
	"fmt"
	"math"
	"slices"
	"time"

	"github.com/cardy0828/pascallm/internal/lfm3gpu"
)

// この file は -mode candidates です。GPU で絞った選び方の候補（lfm3gpu.FeedCandidates）が、logits 全体から CPU で
// 求めた値と一致するかを確かめ、1 トークンの時間を Feed（argmax だけ）・FeedCandidates・Forward（logits 全体を移す）で比べます。

// 候補の比較の許容差です。ペナルティの計算は GPU と CPU で丸めが少し違う（GPU は積和を 1 命令で行う）ため、
// 値は完全には一致しません。
const (
	candidateValueTolerance  = 1e-4 // 同じ順位の値の差の上限
	candidateTieTolerance    = 1e-5 // 番号が違ってもよい「同じ値」とみなす差
	candidateExpSumTolerance = 1e-4 // exp の合計の相対差の上限
	candidateSeenPrefix      = 8    // ペナルティの対象として最初に送るトークン数（プロンプトの代わり）
	candidateTimingSteps     = 300  // 時間を測るトークン数
	candidateTimingWarmup    = 100  // 時間を測る前に捨てるトークン数（クロックを上げるため）
)

// candidateSettings は、位置ごとに順に使う設定です（温度・ペナルティ・選んでよい番号の上限を変える）。
var candidateSettings = []lfm3gpu.CandidateParams{
	{ValidTokens: 125033, InverseTemperature: 10, RepetitionPenalty: 1.1},
	{ValidTokens: 125033, InverseTemperature: 1 / 0.7, RepetitionPenalty: 1.3, PresencePenalty: 0.5, FrequencyPenalty: 0.3},
	{ValidTokens: 128000, InverseTemperature: 1, RepetitionPenalty: 1},
	{ValidTokens: 4096, InverseTemperature: 2, RepetitionPenalty: 0.9, FrequencyPenalty: -0.5},
}

// penaltyMirror は、GPU に送ったペナルティの対象と生成した回数の、CPU 側の写しです。
type penaltyMirror struct {
	seen   map[int]bool
	counts map[int]int
}

// penalize は、CPU で 1 トークンの値にペナルティを適用します（kernels/lfm3.cu の penalized_logit と同じ手順）。
// 例: logit 4、seen、repetition 2 → 2
func (p penaltyMirror) penalize(id int, logit float32, params lfm3gpu.CandidateParams) float32 {
	if p.seen[id] {
		if logit > 0 {
			logit /= params.RepetitionPenalty
		} else {
			logit *= params.RepetitionPenalty
		}
	}
	if count := p.counts[id]; count > 0 {
		logit -= params.PresencePenalty + params.FrequencyPenalty*float32(count)
	}
	return logit
}

// expectedCandidates は、logits 全体から CPU で上位 CandidateCount 個・最大値・exp の合計を求めます。
func expectedCandidates(logits []float32, mirror penaltyMirror, params lfm3gpu.CandidateParams) ([]int, []float32, float32, float64) {
	values := make([]float32, params.ValidTokens)
	order := make([]int, params.ValidTokens)
	for id := range values {
		values[id] = mirror.penalize(id, logits[id], params)
		order[id] = id
	}
	slices.SortFunc(order, func(a, b int) int {
		switch {
		case values[a] > values[b]:
			return -1
		case values[a] < values[b]:
			return 1
		default:
			return a - b
		}
	})
	maxValue := values[order[0]]
	expSum := 0.0
	for _, value := range values {
		expSum += math.Exp(float64(value-maxValue) * float64(params.InverseTemperature))
	}
	top := order[:lfm3gpu.CandidateCount]
	topValues := make([]float32, len(top))
	for rank, id := range top {
		topValues[rank] = values[id]
	}
	return top, topValues, maxValue, expSum
}

// runCandidates は、GPU の候補を位置ごとに CPU の値と比べ、続けて 1 トークンの時間を測ります。
func runCandidates(opts options) error {
	tokens, err := loadTokens(opts.tokensPath)
	if err != nil {
		return err
	}
	positions := min(opts.positions, len(tokens))
	model, err := openGPU(opts)
	if err != nil {
		return err
	}
	defer model.Close()

	if err := model.ResetPenalties(tokens[:candidateSeenPrefix]); err != nil {
		return err
	}
	mirror := penaltyMirror{seen: map[int]bool{}, counts: map[int]int{}}
	for _, id := range tokens[:candidateSeenPrefix] {
		mirror.seen[id] = true
	}
	var maxValueDiff, maxExpSumRelative float64
	idMismatches := 0
	for position := range positions {
		params := candidateSettings[position%len(candidateSettings)]
		accept := position >= candidateSeenPrefix
		candidates, err := model.FeedCandidates(tokens[position], accept, params)
		if err != nil {
			return err
		}
		if accept {
			mirror.seen[tokens[position]] = true
			mirror.counts[tokens[position]]++
		}
		gotIDs := slices.Clone(candidates.IDs)
		gotValues := slices.Clone(candidates.Logits)
		gotMax, gotExpSum := candidates.MaxLogit, candidates.ExpSum
		logits, err := model.CopyLogits()
		if err != nil {
			return err
		}
		wantIDs, wantValues, wantMax, wantExpSum := expectedCandidates(logits, mirror, params)

		positionMismatches := 0
		for rank := range wantIDs {
			maxValueDiff = math.Max(maxValueDiff, math.Abs(float64(gotValues[rank]-wantValues[rank])))
			if int(gotIDs[rank]) != wantIDs[rank] {
				gotValue := mirror.penalize(int(gotIDs[rank]), logits[gotIDs[rank]], params)
				if math.Abs(float64(gotValue-wantValues[rank])) > candidateTieTolerance {
					positionMismatches++
				}
			}
		}
		idMismatches += positionMismatches
		maxValueDiff = math.Max(maxValueDiff, math.Abs(float64(gotMax-wantMax)))
		relative := math.Abs(float64(gotExpSum)-wantExpSum) / wantExpSum
		maxExpSumRelative = math.Max(maxExpSumRelative, relative)
		if positionMismatches > 0 || relative > candidateExpSumTolerance {
			emit(map[string]any{"kind": "candidate_mismatch", "position": position, "id_mismatches": positionMismatches,
				"exp_sum_relative": relative, "got_first": gotIDs[:4], "want_first": wantIDs[:4]})
		}
	}
	ok := idMismatches == 0 && maxValueDiff <= candidateValueTolerance && maxExpSumRelative <= candidateExpSumTolerance
	emit(map[string]any{"kind": "candidate_summary", "positions": positions, "id_mismatches": idMismatches,
		"max_value_diff": maxValueDiff, "max_exp_sum_relative": maxExpSumRelative, "ok": ok})

	if err := timeChoosers(model, tokens); err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("GPU candidates differ from the CPU values")
	}
	return nil
}

// timeChoosers は、1 トークンの時間を Feed・FeedCandidates・Forward で測ります（入力は tokens を順に使う）。
func timeChoosers(model *lfm3gpu.Model, tokens []int) error {
	params := candidateSettings[0]
	methods := []struct {
		name string
		step func(token int) error
	}{
		{"feed_argmax", func(token int) error { _, err := model.Feed(token); return err }},
		{"feed_candidates", func(token int) error { _, err := model.FeedCandidates(token, true, params); return err }},
		{"forward_full_logits", func(token int) error { _, err := model.Forward(token); return err }},
	}
	for _, method := range methods {
		if err := model.Reset(); err != nil {
			return err
		}
		if err := model.ResetPenalties(nil); err != nil {
			return err
		}
		var started time.Time
		for step := range candidateTimingWarmup + candidateTimingSteps {
			if step == candidateTimingWarmup {
				started = time.Now()
			}
			if err := method.step(tokens[step%len(tokens)]); err != nil {
				return err
			}
		}
		perToken := time.Since(started).Seconds() / candidateTimingSteps
		emit(map[string]any{"kind": "candidate_timing", "method": method.name, "ms_per_token": perToken * 1000,
			"tokens_per_second": 1 / perToken, "context_end": candidateTimingWarmup + candidateTimingSteps})
	}
	times, err := model.MeasureCandidateKernels(params, candidateTimingSteps)
	if err != nil {
		return err
	}
	emit(map[string]any{"kind": "candidate_kernels", "partial_us": times.PartialMicroseconds, "final_us": times.FinalMicroseconds})
	return nil
}
