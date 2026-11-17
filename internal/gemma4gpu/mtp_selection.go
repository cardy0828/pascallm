package gemma4gpu

// mtpSelectionBatch は実際の入力で再利用された候補数を追います。未消費の末尾を受理扱いしません。
type mtpSelectionBatch struct {
	bucket, count, accepted   int
	draftMS, verifyMS, wallMS float64
}

// SetMTPSelectionDriven はサンプラー/文法による実際の選択を受理判定に使います。
// 例: true なら温度付き選択、false なら従来の argmax 検証。assistant がなければ何もしません。
func (m *Model) SetMTPSelectionDriven(enabled bool) error {
	if m.mtp == nil {
		return nil
	}
	enabled = enabled && !m.mtp.greedyOnly
	if enabled == m.mtp.selectionDriven {
		return nil
	}
	if err := m.SetMTPEnabled(m.mtp.enabled); err != nil {
		return err
	}
	m.mtp.selectionDriven = enabled
	return nil
}

// selectionMTPEnabled は通常のサンプラー経路で先読みを利用できるかを返します。
func (m *Model) selectionMTPEnabled() bool {
	return m.mtp != nil && m.mtp.enabled && m.mtp.selectionDriven
}

// finishSelectionBatch は完了・拒否・途中終了した検証バッチを1回だけ集計します。
// truncated は未消費候補があるまま stop/要求終了等で打ち切った場合です。
func (mt *mtpState) finishSelectionBatch(truncated bool) {
	b := mt.selectionBatch
	if b == nil {
		return
	}
	mt.observeBatch(b.bucket, b.count, b.accepted, b.draftMS, b.verifyMS, b.wallMS)
	if truncated {
		mt.stats.SelectionTruncated++
	}
	mt.selectionBatch = nil
}
