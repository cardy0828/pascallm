package gemma4gpu

import (
	"errors"
	"fmt"

	"github.com/cardy0828/pascallm/internal/cuda"
	"github.com/cardy0828/pascallm/internal/gpucore"
)

// この file は、選び方の候補を GPU で絞る処理と、位置の保存・復元です。

// 候補を絞る 1 段目（kernels/gemma4.cu の g4_candidates_partial）の寸法です。
const (
	candidatePartThreads = 1024 // 1 スレッドブロックのスレッド数（32 ワープ）
	// candidatePartSpan は、1 区間の長さの上限（1 ブロックのワープ数 32 × CandidateCount）です。
	candidatePartSpan = (candidatePartThreads / gpucore.WarpSize) * gpucore.CandidateCount
)

// noSavedState は、保存した状態がないことを表す SavedPosition の値です。
const noSavedState = -1

// ResetPenalties は、ペナルティの対象の印（seen）を seen のトークンだけにし、生成した回数を 0 にします。
// 生成の始めに、プロンプトのトークン列を渡して呼びます。例: ResetPenalties(promptIDs)
func (m *Model) ResetPenalties(seen []int) error {
	flags := m.hostSeen.Bytes()
	clear(flags)
	for _, id := range seen {
		if id < 0 || id >= len(flags) {
			return fmt.Errorf("gemma4gpu: token %d is outside the vocabulary (size %d)", id, len(flags))
		}
		flags[id] = 1
	}
	if err := m.hostSeen.CopyToDeviceAsync(m.stream, m.buffers.seenFlags); err != nil {
		return err
	}
	return cuda.MemsetBytes(m.stream, m.buffers.tokenCounts, 0, int32Bytes*m.config.VocabSize)
}

// FeedCandidates は token を 1 個処理して位置を 1 進め、次のトークンの候補（上位 CandidateCount 個）を GPU で求めて返します。
// acceptToken が true なら、token を生成済みとして記録してからペナルティを適用します。
// プロンプトの最後のトークンを入れるときは false、生成したトークンを入れるときは true です。
// 例: FeedCandidates(lastPromptToken, false, params)、続いて FeedCandidates(chosen, true, params)
func (m *Model) FeedCandidates(token int, acceptToken bool, params gpucore.CandidateParams) (gpucore.Candidates, error) {
	if err := m.checkInput(token); err != nil {
		return gpucore.Candidates{}, err
	}
	if err := m.checkCandidateParams(params); err != nil {
		return gpucore.Candidates{}, err
	}
	speculative := m.selectionMTPEnabled()
	if speculative {
		if _, err := m.Feed(token); err != nil {
			return gpucore.Candidates{}, err
		}
	} else {
		m.invalidateMTP()
		if err := m.writeInput(token); err != nil {
			return gpucore.Candidates{}, err
		}
		if err := m.runToken(); err != nil {
			return gpucore.Candidates{}, err
		}
	}
	accepted := int32(-1)
	if acceptToken {
		accepted = int32(token)
	}
	if err := m.launchCandidates(params, accepted); err != nil {
		return gpucore.Candidates{}, err
	}
	if err := m.hostCandidates.CopyFromDeviceAsync(m.stream, m.buffers.candidateResult); err != nil {
		return gpucore.Candidates{}, err
	}
	if err := m.stream.Synchronize(); err != nil {
		return gpucore.Candidates{}, err
	}
	if !speculative {
		m.position++
	}
	return gpucore.CandidatesFromResult(m.hostCandidates), nil
}

// checkCandidateParams は、ValidTokens が候補の処理で扱える範囲かを確かめます。
// 例: Gemma 4 では 262144（語彙の大きさ以下、64 区間 × 4096 以下、候補の数以上）
func (m *Model) checkCandidateParams(params gpucore.CandidateParams) error {
	limit := min(m.config.VocabSize, gpucore.CandidateParts*candidatePartSpan)
	if params.ValidTokens < gpucore.CandidateCount || params.ValidTokens > limit {
		return fmt.Errorf("gemma4gpu: ValidTokens %d must be between %d and %d", params.ValidTokens, gpucore.CandidateCount, limit)
	}
	if params.InverseTemperature <= 0 || params.RepetitionPenalty <= 0 {
		return fmt.Errorf("gemma4gpu: InverseTemperature %g and RepetitionPenalty %g must be positive",
			params.InverseTemperature, params.RepetitionPenalty)
	}
	return nil
}

// launchCandidates は、候補を絞る 2 個のカーネル（g4_candidates_partial → candidates_final）を積みます。
// 1 トークンの Graph の後に、通常の起動で積みます（設定が要求ごとに変わるため）。
// accepted は記録するトークン（なければ -1）です。
func (m *Model) launchCandidates(params gpucore.CandidateParams, accepted int32) error {
	b := &m.buffers
	logits := b.logits
	if m.mtp != nil && m.mtp.logitsRow >= 0 {
		logits = m.mtp.logits + cuda.DevicePtr(m.mtp.logitsRow*float32Bytes*m.config.VocabSize)
	}
	partial := new(cuda.KernelArgs).
		Pointer(logits).Int32(int32(params.ValidTokens)).
		Pointer(b.seenFlags).Pointer(b.tokenCounts).Int32(accepted).
		Float32(params.InverseTemperature).Float32(params.RepetitionPenalty).
		Float32(params.PresencePenalty).Float32(params.FrequencyPenalty).
		Pointer(b.candidatePartValues).Pointer(b.candidatePartIDs).Pointer(b.candidatePartMax).Pointer(b.candidatePartSum)
	if err := m.kernels.candidatesPartial.Launch(gpucore.Dim1(gpucore.CandidateParts), gpucore.Dim1(candidatePartThreads), 0,
		m.stream, partial); err != nil {
		return err
	}
	final := new(cuda.KernelArgs).
		Pointer(b.candidatePartValues).Pointer(b.candidatePartIDs).Pointer(b.candidatePartMax).Pointer(b.candidatePartSum).
		Int32(gpucore.CandidateParts).Float32(params.InverseTemperature).
		Pointer(b.candidateResult).
		Pointer(b.candidateResult + cuda.DevicePtr(float32Bytes*gpucore.CandidateCount)).
		Pointer(b.candidateResult + cuda.DevicePtr(float32Bytes*2*gpucore.CandidateCount))
	return m.kernels.candidatesFinal.Launch(gpucore.Dim1(1), gpucore.Dim1(gpucore.CandidateFinalThreads), 0, m.stream, final)
}

// SaveState は、現在の位置を保存します（前に保存した位置は捨てる）。
//
// KV キャッシュは位置ごとに書くため、位置を戻すだけで過去に戻せます。ただし局所の層は輪状バッファなので、
// 保存してから localRingSlack 個より多く進むと、保存した位置から見える k・v が上書きされ、戻せなくなります
// （CanRestore が false になる）。
// 例: Prefill(prompt[:n]) の後に SaveState() → SavedPosition() は n
func (m *Model) SaveState() error {
	m.savedPosition = m.position
	return nil
}

// SavedPosition は、保存した位置を返します。保存した状態がなければ -1 です。
func (m *Model) SavedPosition() int { return m.savedPosition }

// CanRestore は、RestoreState で保存した位置に戻せるかを返します。
// 例: 保存した位置 1000、現在の位置 1800 -> true（進んだのは 800 個で、余裕 1024 以下）
func (m *Model) CanRestore() bool {
	highWater := m.position
	if m.mtp != nil {
		highWater = max(highWater, m.mtp.highWater)
	}
	return m.savedPosition != noSavedState && highWater-m.savedPosition <= localRingSlack
}

// RestoreState は、SaveState で保存した位置に戻します。保存した位置はそのまま残ります。
// 保存した状態がないか、進みすぎて戻せない場合はエラーです（呼び出し側は Reset して最初から入れ直す）。
func (m *Model) RestoreState() error {
	if m.savedPosition == noSavedState {
		return errors.New("gemma4gpu: no saved state to restore")
	}
	if !m.CanRestore() {
		return fmt.Errorf("gemma4gpu: cannot restore position %d from position %d (more than %d tokens ahead)",
			m.savedPosition, m.position, localRingSlack)
	}
	m.invalidateMTP()
	m.position = m.savedPosition
	return nil
}
