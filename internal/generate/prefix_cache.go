package generate

import (
	"fmt"
	"slices"
	"time"
)

// sharedPrefixTokens は複数会話で共有する最初の保存境界です。
const sharedPrefixTokens = 256

// tokenStorageBytes は対象の64-bit Goで保持するintトークン1個の大きさです。
const tokenStorageBytes int64 = 8

// snapshotModel は型変換しないホストKVコピーを提供するモデルです。
type snapshotModel interface {
	SnapshotBytes() int64
	SnapshotState() (any, error)
	RestoreSnapshot(any) error
}

type prefixEntry struct {
	tokens []int
	state  any
	bytes  int64
}

// PrefixMetrics は要求単位の転送時間と累積LRU状態です。
type PrefixMetrics struct {
	Hits      int     `json:"hits"`
	Entries   int     `json:"entries"`
	Bytes     int64   `json:"bytes"`
	Evictions int     `json:"evictions"`
	Skipped   int     `json:"skipped"`
	RestoreMS float64 `json:"restore_ms"`
	SaveMS    float64 `json:"save_ms"`
}

type prefixCache struct {
	model              snapshotModel
	maxBytes           int64
	maxEntries         int
	entries            []prefixEntry // 古いものから新しいものへ
	bytes              int64
	evictions, skipped int
	metrics            PrefixMetrics
}

// SetPrefixCache はホストKV保持を設定します。例: 256 MiB、8件。0 byteなら無効です。
func (e *Engine) SetPrefixCache(maxBytes int64, maxEntries int) error {
	if maxBytes < 0 || maxEntries < 1 {
		return fmt.Errorf("generate: invalid prefix cache limits")
	}
	if maxBytes == 0 {
		e.prefix = nil
		return nil
	}
	model, ok := e.model.(snapshotModel)
	if !ok {
		return fmt.Errorf("generate: model does not support prefix snapshots")
	}
	e.prefix = &prefixCache{model: model, maxBytes: maxBytes, maxEntries: maxEntries}
	return nil
}

// begin は要求単位の時間を初期化します。
func (p *prefixCache) begin() { p.metrics = PrefixMetrics{} }

// report は保持payload/token列の量と累積退避数を返します。
func (p *prefixCache) report() PrefixMetrics {
	out := p.metrics
	out.Entries = len(p.entries)
	out.Bytes = p.bytes
	out.Evictions = p.evictions
	out.Skipped = p.skipped
	return out
}

// best は prompt より短い完全一致prefixのうち最長を探します。
func (p *prefixCache) best(prompt []int) int {
	best := -1
	for i, entry := range p.entries {
		if isUsablePrefix(entry.tokens, prompt) && (best < 0 || len(entry.tokens) > len(p.entries[best].tokens)) {
			best = i
		}
	}
	return best
}

// restore は選んだ項目を復元し、LRUの末尾へ移します。
func (p *prefixCache) restore(index int) ([]int, error) {
	entry := p.entries[index]
	start := time.Now()
	err := p.model.RestoreSnapshot(entry.state)
	p.metrics.RestoreMS += float64(time.Since(start)) / float64(time.Millisecond)
	if err != nil {
		return nil, err
	}
	p.entries = append(slices.Delete(p.entries, index, index+1), entry)
	p.metrics.Hits++
	return slices.Clone(entry.tokens), nil
}

// save は容量を確保してから現在境界を保存します。同じtoken列の重複保存はしません。
func (p *prefixCache) save(tokens []int) error {
	if len(tokens) == 0 {
		return nil
	}
	for i, entry := range p.entries {
		if slices.Equal(entry.tokens, tokens) {
			p.entries = append(slices.Delete(p.entries, i, i+1), entry)
			return nil
		}
	}
	size := p.model.SnapshotBytes() + int64(len(tokens))*tokenStorageBytes
	if size > p.maxBytes {
		p.skipped++
		return nil
	}
	for len(p.entries) > 0 && (len(p.entries) >= p.maxEntries || p.bytes+size > p.maxBytes) {
		p.bytes -= p.entries[0].bytes
		p.entries = slices.Delete(p.entries, 0, 1)
		p.evictions++
	}
	start := time.Now()
	state, err := p.model.SnapshotState()
	p.metrics.SaveMS += float64(time.Since(start)) / float64(time.Millisecond)
	if err != nil {
		return err
	}
	p.entries = append(p.entries, prefixEntry{tokens: slices.Clone(tokens), state: state, bytes: size})
	p.bytes += size
	return nil
}
