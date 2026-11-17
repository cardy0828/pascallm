package generate

import "github.com/cardy0828/pascallm/internal/gpucore"

// prefillChunkTokens は両モデルの入力処理が一度に受け取れる上限です。
const prefillChunkTokens = 512

// Model は生成に必要な GPU 操作です。LFM3 と Gemma 4 の両方が満たします。
// 操作は CUDA コンテキストを持つ同一 OS スレッドから呼びます。
type Model interface {
	VocabSize() int
	MaxContext() int
	AllocatedBytes() int64
	Close() error
	Reset() error
	Prefill([]int) error
	Feed(int) (int, error)
	Forward(int) ([]float32, error)
	CopyLogits() ([]float32, error)
	SaveState() error
	CanRestore() bool
	RestoreState() error
	ResetPenalties([]int) error
	FeedCandidates(int, bool, gpucore.CandidateParams) (gpucore.Candidates, error)
}
