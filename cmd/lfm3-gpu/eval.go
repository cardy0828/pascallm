package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"time"
	"unsafe"
)

// この file は -mode eval です（M5 の品質の比較）。評価文の区切りごとに、GPU 実装の次トークン分布を、
// FP32 の基準（tools/reference/m5_reference.py base の出力）と比べます。
//
// 集計は llama-perplexity --kl-divergence と同じく、各区切り（1024 トークン、先頭は BOS）の後半の位置
// （512 〜 1022）で、次のトークンを予測する分布を比べます。区切りの前半は Prefill で入れ、後半は生成と同じ経路
// （Forward で 1 個ずつ）で入れます。
//   - KL(基準 || GPU) の平均・中央値・99 パーセンタイル・最大
//   - 上位 1 位の一致率
//   - 両者の perplexity と、その比
//
// 分野ごと（区切りの JSON の domains）と全体の結果を出します。

// evalChunks は評価の区切りの JSON（tools/reference/m5_prepare_text.py の出力）です。
type evalChunks struct {
	Chunks  [][]int  `json:"chunks"`
	Domains []string `json:"domains"`
}

// loadEvalChunks は評価の区切りを読み、区切りの長さがそろっていることを確かめます。
func loadEvalChunks(path string) (evalChunks, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return evalChunks{}, err
	}
	var content evalChunks
	if err := json.Unmarshal(raw, &content); err != nil {
		return evalChunks{}, fmt.Errorf("%s: %w", path, err)
	}
	if len(content.Chunks) == 0 || len(content.Chunks) != len(content.Domains) {
		return evalChunks{}, fmt.Errorf("%s: %d chunks and %d domains", path, len(content.Chunks), len(content.Domains))
	}
	for index, chunk := range content.Chunks {
		if len(chunk) != len(content.Chunks[0]) {
			return evalChunks{}, fmt.Errorf("%s: chunk %d has %d tokens, want %d", path, index, len(chunk), len(content.Chunks[0]))
		}
	}
	return content, nil
}

// klStats は、ある範囲（分野または全体）の位置ごとの結果を集めます。
type klStats struct {
	values  []float64
	sameTop int
	baseNLL float64
	testNLL float64
}

// add は 1 位置の結果を加えます。
func (s *klStats) add(kl float64, sameTop bool, baseNLL, testNLL float64) {
	s.values = append(s.values, kl)
	if sameTop {
		s.sameTop++
	}
	s.baseNLL += baseNLL
	s.testNLL += testNLL
}

// percentileOf は、昇順に並べた values の p（0〜1）分位点を返します（最も近い順位の値）。
// 例: percentileOf([1, 2, 3, 4], 0.5) = 2
func percentileOf(sorted []float64, p float64) float64 {
	rank := int(math.Ceil(p*float64(len(sorted)))) - 1
	return sorted[min(max(rank, 0), len(sorted)-1)]
}

// summary は集計結果を JSON の 1 件にします。
func (s *klStats) summary(kind, domain, label string) map[string]any {
	sorted := append([]float64(nil), s.values...)
	sort.Float64s(sorted)
	count := float64(len(sorted))
	var sum float64
	for _, value := range sorted {
		sum += value
	}
	return map[string]any{
		"kind": kind, "domain": domain, "label": label, "scored": len(sorted),
		"mean_kl": sum / count, "median_kl": percentileOf(sorted, 0.5), "p99_kl": percentileOf(sorted, 0.99),
		"max_kl": sorted[len(sorted)-1], "same_top_p": float64(s.sameTop) / count,
		"ppl_base": math.Exp(s.baseNLL / count), "ppl_test": math.Exp(s.testNLL / count),
		"ppl_ratio": math.Exp((s.testNLL - s.baseNLL) / count),
	}
}

// float32Bytes は、float32 の配列をそのままのメモリ表現のバイト列として見せます（コピーしない）。
func float32Bytes(values []float32) []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(&values[0])), 4*len(values))
}

// runEval は、評価文の全区切りで GPU 実装と基準を比べ、分野ごとと全体の結果を出します。
func runEval(opts options) error {
	data, err := loadEvalChunks(opts.tokensPath)
	if err != nil {
		return err
	}
	chunkTokens := len(data.Chunks[0])
	scoreFrom := chunkTokens / 2
	model, err := openGPU(opts)
	if err != nil {
		return err
	}
	defer model.Close()
	vocab := model.Config().VocabSize

	baseFile, err := os.Open(opts.baseLogits)
	if err != nil {
		return err
	}
	defer baseFile.Close()
	rowsPerChunk := chunkTokens - 1 - scoreFrom
	stat, err := baseFile.Stat()
	if err != nil {
		return err
	}
	if want := int64(len(data.Chunks)) * int64(rowsPerChunk) * int64(vocab) * 4; stat.Size() != want {
		return fmt.Errorf("%s has %d bytes, want %d (%d chunks x %d rows x %d)", opts.baseLogits, stat.Size(), want,
			len(data.Chunks), rowsPerChunk, vocab)
	}
	reader := bufio.NewReaderSize(baseFile, 16<<20)

	baseRow := make([]float32, vocab)
	baseLog := make([]float64, vocab)
	testLog := make([]float64, vocab)
	total := &klStats{}
	domains := map[string]*klStats{}
	var order []string
	started := time.Now()
	for index, chunk := range data.Chunks {
		domain := data.Domains[index]
		if domains[domain] == nil {
			domains[domain] = &klStats{}
			order = append(order, domain)
		}
		if err := model.Reset(); err != nil {
			return err
		}
		if err := model.Prefill(chunk[:scoreFrom]); err != nil {
			return err
		}
		// 位置 p の logits は、トークン p+1 を予測する分布です。
		for position := scoreFrom; position < chunkTokens-1; position++ {
			logits, err := model.Forward(chunk[position])
			if err != nil {
				return err
			}
			if _, err := io.ReadFull(reader, float32Bytes(baseRow)); err != nil {
				return fmt.Errorf("reading base logits of chunk %d position %d: %w", index, position, err)
			}
			logSoftmax(baseRow, baseLog)
			logSoftmax(logits, testLog)
			var kl float64
			for token, value := range baseLog {
				kl += math.Exp(value) * (value - testLog[token])
			}
			next := chunk[position+1]
			sameTop := argmax(baseRow) == argmax(logits)
			total.add(kl, sameTop, -baseLog[next], -testLog[next])
			domains[domain].add(kl, sameTop, -baseLog[next], -testLog[next])
		}
	}
	for _, domain := range order {
		emit(domains[domain].summary("eval_domain", domain, opts.label))
	}
	result := total.summary("eval_summary", "all", opts.label)
	result["pq4"] = opts.pq4Path
	result["chunks"] = len(data.Chunks)
	result["gpu_bytes"] = model.AllocatedBytes()
	result["seconds"] = time.Since(started).Seconds()
	emit(result)
	return nil
}
