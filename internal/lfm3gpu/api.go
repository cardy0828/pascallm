package lfm3gpu

// VocabSize はモデルの語彙数を返します。例: Gemma 4 では 262144。
func (m *Model) VocabSize() int { return m.config.VocabSize }

// CanRestore は状態保存済みかを返します。LFM3 の保存状態は生成が進んでも復元できます。
func (m *Model) CanRestore() bool { return m.savedPosition >= 0 }
