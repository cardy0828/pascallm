package lfm3gpu

import (
	"errors"

	"github.com/cardy0828/pascallm/internal/cuda"
	"github.com/cardy0828/pascallm/internal/lfm3"
)

// この file は、ある位置までの状態の保存と復元です。
//
// KV キャッシュは位置ごとに書くため、位置より先の内容は読まれず、位置を戻すだけで過去に戻せます。
// shortconv の履歴（直近 kernel_size-1 個の入力）は上書きされるため、保存しておかないと戻せません。
// そこで、ある位置での全 shortconv 層の履歴を GPU 上の別の領域にコピーしておき、後でその位置に戻せるようにします。
// 生成（internal/generate）は、プロンプトを入れ終えた位置で保存し、次の要求のプロンプトが前回のプロンプトで
// 始まるなら、生成した部分を捨ててその位置から続けます。
//
// 保存できるのは 1 か所だけです。Reset で無効になります。
// 保存した位置より前の KV キャッシュを書き換える操作は Reset の後だけなので、保存した内容は Reset まで正しいままです。

// noSavedState は、保存した状態がないことを表す SavedPosition の値です。
const noSavedState = -1

// convHistoryBytes は、shortconv 層 1 個の履歴のバイト数です（float32 × hidden × (kernel_size-1)）。
// 例: hidden 2048、kernel_size 7 → 4 × 2048 × 6 = 49,152
func (m *Model) convHistoryBytes() int {
	const float32Bytes = 4
	return float32Bytes * m.config.HiddenSize * (m.config.ConvKernelSize - 1)
}

// prepareSnapshot は、全 shortconv 層の履歴を保存する GPU の領域を確保します（LFM3 1.7B で 16 層 × 48 KB = 768 KB）。
func (m *Model) prepareSnapshot() error {
	convLayers := 0
	for _, layer := range m.layers {
		if layer.kind == lfm3.LayerConv {
			convLayers++
		}
	}
	m.savedPosition = noSavedState
	if convLayers == 0 {
		return nil
	}
	pointer, err := m.memory.alloc(convLayers * m.convHistoryBytes())
	if err != nil {
		return err
	}
	m.savedHistory = pointer
	return nil
}

// copyConvHistories は、各 shortconv 層の履歴と保存領域の間でコピーします。
// save が true なら履歴 → 保存領域、false なら保存領域 → 履歴です。完了まで待ちます。
func (m *Model) copyConvHistories(save bool) error {
	size := m.convHistoryBytes()
	offset := 0
	for _, layer := range m.layers {
		if layer.kind != lfm3.LayerConv {
			continue
		}
		saved := m.savedHistory + cuda.DevicePtr(offset)
		source, destination := saved, layer.convHistory
		if save {
			source, destination = layer.convHistory, saved
		}
		if err := cuda.CopyDeviceAsync(m.stream, destination, source, size); err != nil {
			return err
		}
		offset += size
	}
	return m.stream.Synchronize()
}

// SaveState は、現在の位置と shortconv の履歴を保存します（前に保存した内容は捨てる）。
// 例: Prefill(prompt[:n]) の後に SaveState() → SavedPosition() は n
func (m *Model) SaveState() error {
	if err := m.copyConvHistories(true); err != nil {
		m.savedPosition = noSavedState
		return err
	}
	m.savedPosition = m.position
	return nil
}

// SavedPosition は、保存した位置を返します。保存した状態がなければ -1 です。
func (m *Model) SavedPosition() int { return m.savedPosition }

// RestoreState は、SaveState で保存した位置と shortconv の履歴に戻します。保存した内容はそのまま残ります。
// attention の終了数も Reset と同じ理由で 0 にします。保存した状態がなければエラーです。
func (m *Model) RestoreState() error {
	if m.savedPosition == noSavedState {
		return errors.New("lfm3gpu: no saved state to restore")
	}
	if err := cuda.MemsetBytes(m.stream, m.buffers.attentionDone, 0, m.attentionDoneBytes()); err != nil {
		return err
	}
	if err := m.copyConvHistories(false); err != nil {
		return err
	}
	m.position = m.savedPosition
	return nil
}
