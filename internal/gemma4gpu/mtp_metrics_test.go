package gemma4gpu

import "testing"

// TestMTPPolicyOptimizesTime は受理率が高い長さより、受理トークンあたりの時間が短い長さを選ぶことを確認します。
func TestMTPPolicyOptimizesTime(t *testing.T) {
	mt := &mtpState{fixedDraft: 7, adaptive: true}
	// 7 候補は全受理でも 100 ms、3 候補は 2 受理で 18 ms。率ではなく速度なら 3 を選ぶ。
	for _, sample := range []struct {
		count, accepted int
		cost            float64
	}{{3, 2, 18}, {1, 1, 17}, {2, 1, 18}, {7, 7, 100}} {
		got := mt.chooseDraftCount(0)
		if got != sample.count {
			t.Fatalf("exploration got %d want %d", got, sample.count)
		}
		mt.observeBatch(0, sample.count, sample.accepted, 1, sample.cost-1, sample.cost)
	}
	if got := mt.chooseDraftCount(0); got != 3 {
		t.Fatalf("selected %d, want 3", got)
	}
	// 入力傾向が変化し、3 候補が連続で外れると、より短い長さへ切り替える。
	for i := 0; i < 8; i++ {
		mt.observeBatch(0, 3, 0, 1, 40, 41)
	}
	if got := mt.chooseDraftCount(0); got == 3 {
		t.Fatal("policy did not adapt to repeated rejection")
	}
}

// TestMTPMetricsDistribution は受理長と深さ別分布の分母を、拒否を含む複数バッチで検査します。
func TestMTPMetricsDistribution(t *testing.T) {
	mt := &mtpState{fixedDraft: 3}
	mt.observeBatch(1, 3, 0, 1, 10, 12)
	mt.observeBatch(1, 3, 2, 1, 10, 12)
	mt.observeBatch(1, 3, 3, 1, 10, 12)
	stats := mt.stats
	if stats.Batches != 3 || stats.Drafted != 9 || stats.Accepted != 5 {
		t.Fatal(stats)
	}
	if stats.AcceptedLengths[0] != 1 || stats.AcceptedLengths[2] != 1 || stats.AcceptedLengths[3] != 1 {
		t.Fatal(stats.AcceptedLengths)
	}
	if stats.ProposedAtDepth[2] != 3 || stats.AcceptedAtDepth[2] != 1 || stats.AcceptedAtDepth[1] != 2 {
		t.Fatal(stats)
	}
	if stats.ByContext[1].Batches != 3 || stats.ByDraftCount[3].Batches != 3 {
		t.Fatal(stats)
	}
}

// TestSelectionMetricsFinalizeOnce は消費済み候補だけを集計し、要求終了後の再集計を防ぎます。
func TestSelectionMetricsFinalizeOnce(t *testing.T) {
	mt := &mtpState{selectionBatch: &mtpSelectionBatch{count: 3, accepted: 1, bucket: 0, draftMS: 2, verifyMS: 20, wallMS: 23}}
	mt.finishSelectionBatch(true)
	mt.finishSelectionBatch(true)
	if mt.stats.Batches != 1 || mt.stats.Accepted != 1 || mt.stats.SelectionTruncated != 1 {
		t.Fatal(mt.stats)
	}
}
