package gemma4gpu

import "fmt"

const mtpContextBuckets = 4
const mtpPolicyExploreInterval = 16
const mtpPolicySmoothing = 0.25
const mtpAcceptanceDecay = 0.98

// MTPDistribution は候補数別/文脈別の分布と時間です。AcceptedLengths[a] は a 候補を受理した回数です。
// ProposedAtDepth[d] と AcceptedAtDepth[d] の比は、候補列の先頭から d+1 個が受理される確率です。
type MTPDistribution struct {
	Batches            int                    `json:"batches"`
	Drafted            int                    `json:"drafted"`
	Accepted           int                    `json:"accepted"`
	AcceptedLengths    [mtpVerifyTokens]int   `json:"accepted_lengths"`
	ProposedAtDepth    [mtpMaxDraftTokens]int `json:"proposed_at_depth"`
	AcceptedAtDepth    [mtpMaxDraftTokens]int `json:"accepted_at_depth"`
	DraftMilliseconds  float64                `json:"draft_gpu_ms"`
	VerifyMilliseconds float64                `json:"verify_gpu_ms"`
	WallMilliseconds   float64                `json:"batch_wall_ms"`
}

// MTPStatistics は候補列の検証結果です。受理は本体 argmax との一致で、API が返したトークン数とは異なります。
// ByContext は順に <1024 / 1024..4095 / 4096..8191 / >=8192 の生成位置です。
type MTPStatistics struct {
	SelectionDriven    bool `json:"selection_driven"`
	SelectionRejected  int  `json:"selection_rejected"`
	SelectionTruncated int  `json:"selection_truncated"`
	MTPDistribution
	ByDraftCount [mtpVerifyTokens]MTPDistribution   `json:"by_draft_count"`
	ByContext    [mtpContextBuckets]MTPDistribution `json:"by_context"`
	Adaptive     bool                               `json:"adaptive"`
	DraftLimit   int                                `json:"draft_limit"`
}

// mtpPolicy は長さごとの実測コストと深さ別の条件付き受理確率を追います。入力内容を保存しません。
type mtpPolicy struct {
	batches     int
	exploration int
	samples     [mtpVerifyTokens]int
	cost        [mtpVerifyTokens]float64
	reached     [mtpMaxDraftTokens]float64
	successes   [mtpMaxDraftTokens]float64
}

// mtpContextBucket は意味のある文脈範囲を明示的に分類します。例: 4095 -> 1、4096 -> 2。
func mtpContextBucket(position int) int {
	switch {
	case position < 1024:
		return 0
	case position < 4096:
		return 1
	case position < 8192:
		return 2
	default:
		return 3
	}
}

// ConfigureMTP は固定候補数または自動調整の上限を指定します。0 は従来の 3、範囲は 1..7 です。
// 自動調整では 1/2/3/7 のうち上限以下を比較します。4 行の行列積タイルに合う長さを優先します。
func (m *Model) ConfigureMTP(limit int, adaptive bool) error {
	if limit == 0 {
		limit = mtpDraftTokens
	}
	if limit < 1 || limit > mtpMaxDraftTokens {
		return fmt.Errorf("MTP draft count must be between 1 and %d", mtpMaxDraftTokens)
	}
	if m.mtp == nil {
		return fmt.Errorf("MTP assistant is not loaded")
	}
	m.mtp.fixedDraft = limit
	m.mtp.adaptive = adaptive
	m.mtp.policy = [mtpContextBuckets]mtpPolicy{}
	return nil
}

// chooseDraftCount は実測 wall 時間/期待受理トークン数が小さい長さを選びます。
// 各長さを最初に 1 回試し、以後 16 バッチごとに再探索して内容変化に追従します。
func (mt *mtpState) chooseDraftCount(bucket int) int {
	if !mt.adaptive {
		return mt.fixedDraft
	}
	p := &mt.policy[bucket]
	choices := []int{}
	for _, count := range []int{3, 1, 2, 7} {
		if count <= mt.fixedDraft {
			choices = append(choices, count)
		}
	}
	for _, count := range choices {
		if p.samples[count] == 0 {
			return count
		}
	}
	if p.batches%mtpPolicyExploreInterval == 0 {
		count := choices[p.exploration%len(choices)]
		p.exploration++
		return count
	}
	best := choices[0]
	for _, count := range choices[1:] {
		if p.cost[count]/p.expectedGain(count) < p.cost[best]/p.expectedGain(best) {
			best = count
		}
	}
	return best
}

// observe は 1 検証バッチを分布へ加算します。例: count=3、accepted=2 は受理長 2 の頻度を 1 増やします。
func (d *MTPDistribution) observe(count, accepted int, draftMS, verifyMS, wallMS float64) {
	d.Batches++
	d.Drafted += count
	d.Accepted += accepted
	d.AcceptedLengths[accepted]++
	for depth := 0; depth < count; depth++ {
		d.ProposedAtDepth[depth]++
		if depth < accepted {
			d.AcceptedAtDepth[depth]++
		}
	}
	d.DraftMilliseconds += draftMS
	d.VerifyMilliseconds += verifyMS
	d.WallMilliseconds += wallMS
}

// observeBatch は分布と自動調整用の移動平均を更新します。上限付近で短縮したバッチも実際の長さで数えます。
func (mt *mtpState) observeBatch(bucket, count, accepted int, draftMS, verifyMS, wallMS float64) {
	mt.stats.MTPDistribution.observe(count, accepted, draftMS, verifyMS, wallMS)
	mt.stats.ByDraftCount[count].observe(count, accepted, draftMS, verifyMS, wallMS)
	mt.stats.ByContext[bucket].observe(count, accepted, draftMS, verifyMS, wallMS)
	p := &mt.policy[bucket]
	if p.samples[count] == 0 {
		p.cost[count] = wallMS
	} else {
		p.cost[count] = (1-mtpPolicySmoothing)*p.cost[count] + mtpPolicySmoothing*wallMS
	}
	// 先行候補が拒否された後の候補は、正しい prefix での観測ではないため母数へ入れない。
	// 全候補数で深さの観測を共有し、最初の 1 バッチだけで長さの良し悪しを決めない。
	for depth := 0; depth < mtpMaxDraftTokens; depth++ {
		p.reached[depth] *= mtpAcceptanceDecay
		p.successes[depth] *= mtpAcceptanceDecay
		if depth < count && depth <= accepted {
			p.reached[depth]++
			if depth < accepted {
				p.successes[depth]++
			}
		}
	}
	p.samples[count]++
	p.batches++
}

// ResetGenerationMetrics は要求単位の統計と調整履歴をリセットします。未消費候補を破棄し、論理位置までの KV は保持します。
func (m *Model) ResetGenerationMetrics() {
	if m.mtp != nil {
		m.invalidateMTP()
		m.mtp.stats = MTPStatistics{}
		m.mtp.policy = [mtpContextBuckets]mtpPolicy{}
	}
}

// GenerationMetrics は要求終了時に呼び、残る未消費バッチを打ち切りとして確定して診断情報を返します。MTP 無効モデルでは nil。
func (m *Model) GenerationMetrics() any {
	if m.mtp == nil || !m.mtp.enabled {
		return nil
	}
	m.mtp.finishSelectionBatch(true)
	return m.MTPStats()
}

// expectedGain は条件付き受理率の積から 1+期待受理数を求めます。
// 未観測の深さは Beta(1,1) の事前分布で過信を避け、深い候補ほど受理率が増えないようにします。
func (p *mtpPolicy) expectedGain(count int) float64 {
	gain := 1.0
	survival := 1.0
	for depth := 0; depth < count; depth++ {
		conditional := (p.successes[depth] + 1) / (p.reached[depth] + 2)
		survival *= conditional
		gain += survival
	}
	return gain
}
