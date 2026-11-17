package lfm3gpu

import (
	"fmt"

	"github.com/cardy0828/pascallm/internal/cuda"
	"github.com/cardy0828/pascallm/internal/gpucore"
)

// 選び方の候補を GPU で絞る処理の寸法です（kernels/lfm3.cu の kCandidate* と合わせる）。
const (
	// CandidateCount は、FeedCandidates が返す候補の数です。
	CandidateCount        = 128
	candidatePartThreads  = 512  // candidates_partial の 1 スレッドブロックのスレッド数
	candidateFinalThreads = 1024 // candidates_final のスレッド数
	candidateParts        = 64   // candidates_partial のスレッドブロック数（語彙を分ける区間の数）
	candidatePartSpan     = 2048 // 1 区間の長さの上限（1 ブロックのワープ数 16 × CandidateCount）
	candidateStatCount    = 2    // candidates_final が書く統計の数（全体の最大値、exp の合計）
	// candidateResultFloats は、2 段目の結果 [値 × 128][番号 × 128][統計 × 2] の 4 バイト単位の数です。
	candidateResultFloats = 2*CandidateCount + candidateStatCount
)

// CandidateParams は、候補を絞るときに GPU で適用する設定です。
type CandidateParams = gpucore.CandidateParams

// Candidates は GPU で絞った候補です。LFM3 と Gemma 4 で同じ形式を使います。
type Candidates = gpucore.Candidates

// ResetPenalties は、ペナルティの対象の印（seen）を seen のトークンだけにし、生成した回数を 0 にします。
// 生成の始めに、プロンプトのトークン列を渡して呼びます。例: ResetPenalties(promptIDs)
func (m *Model) ResetPenalties(seen []int) error {
	flags := m.hostSeen.Bytes()
	clear(flags)
	for _, id := range seen {
		if id < 0 || id >= len(flags) {
			return fmt.Errorf("lfm3gpu: token %d is outside the vocabulary (size %d)", id, len(flags))
		}
		flags[id] = 1
	}
	if err := m.hostSeen.CopyToDeviceAsync(m.stream, m.buffers.seenFlags); err != nil {
		return err
	}
	return cuda.MemsetBytes(m.stream, m.buffers.tokenCounts, 0, 4*m.config.VocabSize)
}

// FeedCandidates は token を 1 個処理して位置を 1 進め、次のトークンの候補（上位 CandidateCount 個）を GPU で求めて返します。
// acceptToken が true なら、token を生成済みとして記録してからペナルティを適用します（生成した回数を 1 増やし、
// seen の印を付ける）。プロンプトの最後のトークンを入れるときは false、生成したトークンを入れるときは true です。
// 例: FeedCandidates(lastPromptToken, false, params)、続いて FeedCandidates(chosen, true, params)
func (m *Model) FeedCandidates(token int, acceptToken bool, params CandidateParams) (Candidates, error) {
	if err := m.checkInput(token); err != nil {
		return Candidates{}, err
	}
	if err := m.checkCandidateParams(params); err != nil {
		return Candidates{}, err
	}
	if err := m.writeInput(token); err != nil {
		return Candidates{}, err
	}
	if err := m.runToken(); err != nil {
		return Candidates{}, err
	}
	accepted := int32(-1)
	if acceptToken {
		accepted = int32(token)
	}
	if err := m.launchCandidates(params, accepted); err != nil {
		return Candidates{}, err
	}
	if err := m.hostCandidates.CopyFromDeviceAsync(m.stream, m.buffers.candidateResult); err != nil {
		return Candidates{}, err
	}
	if err := m.stream.Synchronize(); err != nil {
		return Candidates{}, err
	}
	m.position++
	values := m.hostCandidates.Float32s()
	return Candidates{
		Logits:   values[:CandidateCount],
		IDs:      m.hostCandidates.Int32s()[CandidateCount : 2*CandidateCount],
		MaxLogit: values[2*CandidateCount],
		ExpSum:   values[2*CandidateCount+1],
	}, nil
}

// checkCandidateParams は、ValidTokens が候補の処理で扱える範囲かを確かめます。
// 例: LFM3 では 125033（語彙 128000 以下、64 区間 × 2048 以下、候補の数以上）
func (m *Model) checkCandidateParams(params CandidateParams) error {
	limit := min(m.config.VocabSize, candidateParts*candidatePartSpan)
	if params.ValidTokens < CandidateCount || params.ValidTokens > limit {
		return fmt.Errorf("lfm3gpu: ValidTokens %d must be between %d and %d", params.ValidTokens, CandidateCount, limit)
	}
	if params.InverseTemperature <= 0 || params.RepetitionPenalty <= 0 {
		return fmt.Errorf("lfm3gpu: InverseTemperature %g and RepetitionPenalty %g must be positive",
			params.InverseTemperature, params.RepetitionPenalty)
	}
	return nil
}

// launchCandidates は、候補を絞る 2 個のカーネル（candidates_partial → candidates_final）を積みます。
// 1 トークンの Graph の後に、通常の起動で積みます（設定が要求ごとに変わるため）。
func (m *Model) launchCandidates(params CandidateParams, accepted int32) error {
	if err := m.launchCandidatePartial(params, accepted); err != nil {
		return err
	}
	return m.launchCandidateFinal(params)
}

// launchCandidatePartial は 1 段目（candidates_partial）を積みます。accepted は記録するトークン（なければ -1）です。
func (m *Model) launchCandidatePartial(params CandidateParams, accepted int32) error {
	b := &m.buffers
	partial := new(cuda.KernelArgs).
		Pointer(b.logits).Int32(int32(params.ValidTokens)).
		Pointer(b.seenFlags).Pointer(b.tokenCounts).Int32(accepted).
		Float32(params.InverseTemperature).Float32(params.RepetitionPenalty).
		Float32(params.PresencePenalty).Float32(params.FrequencyPenalty).
		Pointer(b.candidatePartValues).Pointer(b.candidatePartIDs).Pointer(b.candidatePartMax).Pointer(b.candidatePartSum)
	return m.kernels.candidatesPartial.Launch(dim1(candidateParts), dim1(candidatePartThreads), 0, m.stream, partial)
}

// launchCandidateFinal は 2 段目（candidates_final）を積み、結果を candidateResult に書きます。
func (m *Model) launchCandidateFinal(params CandidateParams) error {
	b := &m.buffers
	const bytesPerValue = 4
	final := new(cuda.KernelArgs).
		Pointer(b.candidatePartValues).Pointer(b.candidatePartIDs).Pointer(b.candidatePartMax).Pointer(b.candidatePartSum).
		Int32(candidateParts).Float32(params.InverseTemperature).
		Pointer(b.candidateResult).
		Pointer(b.candidateResult + cuda.DevicePtr(bytesPerValue*CandidateCount)).
		Pointer(b.candidateResult + cuda.DevicePtr(bytesPerValue*2*CandidateCount))
	return m.kernels.candidatesFinal.Launch(dim1(1), dim1(candidateFinalThreads), 0, m.stream, final)
}

// CandidateKernelTimes は、候補を絞る 2 個のカーネルの GPU 上の時間（1 回あたり、マイクロ秒）です。
type CandidateKernelTimes struct {
	PartialMicroseconds float64 // candidates_partial
	FinalMicroseconds   float64 // candidates_final
}

// MeasureCandidateKernels は、直前に処理した位置の logits に対して候補のカーネルを replays 回ずつ動かし、
// 1 回あたりの GPU 上の時間を返します（確認用。ペナルティの記録はしない）。例: MeasureCandidateKernels(params, 200)
func (m *Model) MeasureCandidateKernels(params CandidateParams, replays int) (CandidateKernelTimes, error) {
	if err := m.checkCandidateParams(params); err != nil {
		return CandidateKernelTimes{}, err
	}
	events := make([]*cuda.Event, 3)
	for index := range events {
		event, err := cuda.NewEvent()
		if err != nil {
			return CandidateKernelTimes{}, err
		}
		defer event.Close()
		events[index] = event
	}
	const noAcceptedToken = -1
	if err := events[0].Record(m.stream); err != nil {
		return CandidateKernelTimes{}, err
	}
	for range replays {
		if err := m.launchCandidatePartial(params, noAcceptedToken); err != nil {
			return CandidateKernelTimes{}, err
		}
	}
	if err := events[1].Record(m.stream); err != nil {
		return CandidateKernelTimes{}, err
	}
	for range replays {
		if err := m.launchCandidateFinal(params); err != nil {
			return CandidateKernelTimes{}, err
		}
	}
	if err := events[2].Record(m.stream); err != nil {
		return CandidateKernelTimes{}, err
	}
	if err := events[2].Synchronize(); err != nil {
		return CandidateKernelTimes{}, err
	}
	partial, err := cuda.ElapsedMilliseconds(events[0], events[1])
	if err != nil {
		return CandidateKernelTimes{}, err
	}
	final, err := cuda.ElapsedMilliseconds(events[1], events[2])
	if err != nil {
		return CandidateKernelTimes{}, err
	}
	const microsecondsPerMillisecond = 1000
	return CandidateKernelTimes{
		PartialMicroseconds: partial * microsecondsPerMillisecond / float64(replays),
		FinalMicroseconds:   final * microsecondsPerMillisecond / float64(replays),
	}, nil
}
