package lfm3gpu

import (
	"fmt"
	"sort"

	"github.com/cardy0828/pascallm/internal/cuda"
)

// shareWarmupTokens は、各計測の前に 1 トークン分の Graph を再生する回数です。
// GPU は処理の合間にクロックを下げるため（P5 では SM 139 MHz）、計測の直前まで動かしておきます。
const shareWarmupTokens = 50

// initialWarmupTokens は、計測を始める前に 1 トークン分の Graph を再生する回数です（約 3 秒）。
// 生成中のクロック（P2、SM 1911 MHz）に上がるまで 1〜2 秒かかったため、それより長く動かします。
// 再生ごとに位置が進まないよう、ここでは argmax の後の位置を毎回戻します。
const initialWarmupTokens = 1000

// KernelShare は、1 種類のカーネル（名前と形が同じもの）が 1 トークンの時間のうちどれだけを占めるかです。
//
// Microseconds は「全カーネルの Graph」と「その種類だけを抜いた Graph」の 1 トークンあたりの時間の差です。
// 単独で続けて起動して測る方法と違い、実際の並び（前後のカーネルとの重なり・起動の間隔）での負担を表します。
// 抜いた Graph は計算結果が正しくありませんが、時間の計測には影響しません。
// 例: {Label: "residual_norm_quantize", PerToken: 49, Microseconds: 600}
type KernelShare struct {
	Label        string
	PerToken     int
	Microseconds float64
}

// countLaunches は、1 トークン分の起動をラベルごとに数えます（何も積まない）。
func (m *Model) countLaunches() (map[string]int, error) {
	counts := map[string]int{}
	if err := m.enqueueTokenWith(&launcher{stream: m.stream, counts: counts}); err != nil {
		return nil, err
	}
	return counts, nil
}

// recordSkipping は、skip のラベルの起動を抜いた 1 トークン分の処理を CUDA Graph に記録し、実行可能な形にして返します。
func (m *Model) recordSkipping(skip map[string]bool) (*cuda.GraphExec, error) {
	if err := m.stream.BeginCapture(); err != nil {
		return nil, err
	}
	enqueueError := m.enqueueTokenWith(&launcher{stream: m.stream, skip: skip})
	graph, endError := m.stream.EndCapture()
	if enqueueError != nil {
		if graph != nil {
			graph.Close()
		}
		return nil, enqueueError
	}
	if endError != nil {
		return nil, endError
	}
	defer graph.Close()
	return graph.Instantiate()
}

// timeGraph は、executable を replays 回続けて再生したときの 1 回あたりの GPU 時間（µs）を返します。
// 計測の前に、位置を startPosition に戻し、通常の Graph を shareWarmupTokens 回再生してクロックを上げます。
// 位置を戻すのは、attention の時間が文脈の長さで変わるため、どの計測も同じ位置から始めるためです。
func (m *Model) timeGraph(executable *cuda.GraphExec, replays, startPosition int, start, end *cuda.Event) (float64, error) {
	for warmup := 0; warmup < shareWarmupTokens; warmup++ {
		if err := m.graph.Launch(m.stream); err != nil {
			return 0, err
		}
	}
	m.position = startPosition
	if err := m.writeInput(0); err != nil {
		return 0, err
	}
	if err := start.Record(m.stream); err != nil {
		return 0, err
	}
	for replay := 0; replay < replays; replay++ {
		if err := executable.Launch(m.stream); err != nil {
			return 0, err
		}
	}
	if err := end.Record(m.stream); err != nil {
		return 0, err
	}
	if err := end.Synchronize(); err != nil {
		return 0, err
	}
	milliseconds, err := cuda.ElapsedMilliseconds(start, end)
	if err != nil {
		return 0, err
	}
	return milliseconds * 1e3 / float64(replays), nil
}

// MeasureKernelShares は、1 トークンの GPU 時間（µs）と、カーネルの種類ごとの負担（長い順）を測ります。
// 現在の位置から replays 個のトークンを処理する時間で測るので、何トークンか処理してから呼びます。
// 例: replays = 200 なら、現在の位置から 200 位置分の平均
//
// 計測のために畳み込みの履歴・KV キャッシュ・状態を書き換えるため、終わった後で Reset します（位置は 0 に戻る）。
func (m *Model) MeasureKernelShares(replays int) (float64, []KernelShare, error) {
	startPosition := m.position
	// 計測の再生と暖機で位置が replays + shareWarmupTokens だけ進むため、KV キャッシュに収まるか確かめる。
	if startPosition == 0 || startPosition+replays+shareWarmupTokens > m.maxContext {
		return 0, nil, fmt.Errorf("lfm3gpu: MeasureKernelShares needs 0 < position and position + %d <= MaxContext %d",
			replays+shareWarmupTokens, m.maxContext)
	}
	counts, err := m.countLaunches()
	if err != nil {
		return 0, nil, err
	}
	start, err := cuda.NewEvent()
	if err != nil {
		return 0, nil, err
	}
	defer start.Close()
	end, err := cuda.NewEvent()
	if err != nil {
		return 0, nil, err
	}
	defer end.Close()

	if err := m.warmClock(initialWarmupTokens, startPosition); err != nil {
		return 0, nil, err
	}
	labels := make([]string, 0, len(counts))
	for label := range counts {
		labels = append(labels, label)
	}
	sort.Strings(labels)
	// クロックの揺れの影響を減らすため、抜いた Graph を測るたびに、直前に全カーネルの Graph も測って差を取る。
	var shares []KernelShare
	var fullSum float64
	for _, label := range labels {
		executable, err := m.recordSkipping(map[string]bool{label: true})
		if err != nil {
			return 0, nil, err
		}
		full, err := m.timeGraph(m.graph, replays, startPosition, start, end)
		if err != nil {
			executable.Close()
			return 0, nil, err
		}
		without, err := m.timeGraph(executable, replays, startPosition, start, end)
		executable.Close()
		if err != nil {
			return 0, nil, fmt.Errorf("lfm3gpu: measuring without %s: %w", label, err)
		}
		fullSum += full
		shares = append(shares, KernelShare{Label: label, PerToken: counts[label], Microseconds: full - without})
	}
	sort.Slice(shares, func(a, b int) bool { return shares[a].Microseconds > shares[b].Microseconds })
	return fullSum / float64(len(labels)), shares, m.Reset()
}

// warmClock は、1 トークン分の Graph を tokens 回再生して GPU のクロックを上げます。
// 位置が KV キャッシュを超えないよう、shareWarmupTokens 回ごとに位置を position に戻します。
func (m *Model) warmClock(tokens, position int) error {
	for done := 0; done < tokens; done += shareWarmupTokens {
		m.position = position
		if err := m.writeInput(0); err != nil {
			return err
		}
		for step := 0; step < shareWarmupTokens; step++ {
			if err := m.graph.Launch(m.stream); err != nil {
				return err
			}
		}
	}
	m.position = position
	return m.stream.Synchronize()
}

// gemvLabel は、GEMV を形ごとに分けて集計するためのラベルです。例: "gemv_pq4 7680x2048"
func gemvLabel(matrix deviceMatrix) string {
	return fmt.Sprintf("gemv_pq4 %dx%d %s", matrix.rows, matrix.cols, matrix.format())
}
