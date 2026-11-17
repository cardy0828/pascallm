package gemma4gpu

import (
	"fmt"
	"github.com/cardy0828/pascallm/internal/cuda"
)

// kvSnapshot は同じモデル専用の不変なホスト側コピーです。外部ファイルは受け入れません。
type kvSnapshot struct {
	owner    *Model
	position int
	layers   [][]byte
}

// SnapshotBytes は現在位置を復元するためのKV payload量です。例: 256位置の全層K/V。
func (m *Model) SnapshotBytes() int64 {
	var size int64
	for _, layer := range m.layers {
		size += int64(min(m.position, layer.ringSize) * layer.kvWidth() * (m.keyBytes + m.valueBytes))
	}
	return size
}

// SnapshotState は prompt 処理直後のKVをホストに複製します。先読み中の状態は受け入れません。
func (m *Model) SnapshotState() (any, error) {
	if m.mtp != nil && len(m.mtp.pendingInputs) > 0 {
		return nil, fmt.Errorf("gemma4gpu: snapshot requires a settled prompt boundary")
	}
	snapshot := &kvSnapshot{owner: m, position: m.position, layers: make([][]byte, len(m.layers))}
	for index, layer := range m.layers {
		count := min(m.position, layer.ringSize) * layer.kvWidth()
		data := make([]byte, count*(m.keyBytes+m.valueBytes))
		keySize := count * m.keyBytes
		if err := cuda.CopyFromDevice(m.stream, data[:keySize], layer.keyCache); err != nil {
			return nil, err
		}
		if err := cuda.CopyFromDevice(m.stream, data[keySize:], layer.valueCache); err != nil {
			return nil, err
		}
		snapshot.layers[index] = data
	}
	return snapshot, nil
}

// RestoreSnapshot は同じモデルの状態だけを復元します。logits/候補/保存位置は次の入力で再構築します。
func (m *Model) RestoreSnapshot(value any) error {
	snapshot, ok := value.(*kvSnapshot)
	if !ok || snapshot == nil || snapshot.owner != m || snapshot.position < 0 || snapshot.position > m.maxContext {
		return fmt.Errorf("gemma4gpu: snapshot belongs to a different model or is invalid")
	}
	if err := m.Reset(); err != nil {
		return err
	}
	for index, layer := range m.layers {
		data := snapshot.layers[index]
		keySize := min(snapshot.position, layer.ringSize) * layer.kvWidth() * m.keyBytes
		if err := cuda.CopyToDevice(m.stream, layer.keyCache, data[:keySize]); err != nil {
			return err
		}
		if err := cuda.CopyToDevice(m.stream, layer.valueCache, data[keySize:]); err != nil {
			return err
		}
	}
	m.position = snapshot.position
	return nil
}
