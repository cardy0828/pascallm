package gemma4gpu

import (
	"fmt"
	"time"

	"github.com/cardy0828/pascallm/internal/cuda"
	"github.com/cardy0828/pascallm/internal/gpucore"
)

// mtpState は assistant と一括検証用バッファです。メモリ所有者は target Model です。
// pending は検証済みでも呼出元がまだ入力していないトークンです。論理位置は Feed ごとに 1 だけ進めます。
type mtpState struct {
	integrated     *cuda.GraphExec
	dynamicPrepare *cuda.Function
	dynamicLocal   *cuda.Function
	dynamicGlobal  *cuda.Function
	dynamicCombine *cuda.Function
	compact        *cuda.Function
	jointResult    cuda.DevicePtr
	capturing      bool
	lookupLimit    int
	history        []int

	selectionDriven bool
	greedyOnly      bool
	selectionBatch  *mtpSelectionBatch
	draft           *Model
	enabled         bool
	seedValid       bool
	expected        int

	pre           gpucore.Matrix
	post          gpucore.Matrix
	seed          cuda.DevicePtr
	partial       cuda.DevicePtr
	logits        cuda.DevicePtr
	states        cuda.DevicePtr
	partialStride int

	normalize     *cuda.Function
	concat        *cuda.Function
	verifyLocal   *cuda.Function
	verifyGlobal  *cuda.Function
	verifyCombine *cuda.Function
	graphs        [mtpVerifyTokens]*cuda.GraphExec
	chainStart    *cuda.Function
	chainNext     *cuda.Function
	hostDraft     *cuda.HostBuffer
	draftStart    *cuda.Event
	verifyStart   *cuda.Event
	verifyEnd     *cuda.Event
	fixedDraft    int
	adaptive      bool
	policy        [mtpContextBuckets]mtpPolicy
	host          *cuda.HostBuffer

	pendingInputs  []int
	pendingOutputs []int
	pendingRows    []int
	logitsRow      int
	highWater      int
	stats          MTPStatistics
}

// prepare は assistant の Graph と最大 8 トークンの検証領域を作ります。
func (mt *mtpState) prepare(m *Model) error {
	var err error
	d := mt.draft
	entries := []namedKernel{{&mt.chainStart, "g4_mtp_chain_start"}, {&mt.chainNext, "g4_mtp_chain_next"}, {&mt.normalize, "g4_mtp_hidden"}, {&mt.concat, "g4_mtp_concat"},
		{&mt.verifyLocal, "g4_verify_local"}, {&mt.verifyGlobal, "g4_verify_global"}, {&mt.verifyCombine, "g4_verify_combine"},
		{&d.kernels.attentionPrepare, "g4_mtp_prepare"}, {&d.kernels.attentionLocal, "g4_mtp_local"},
		{&d.kernels.attentionGlobal, "g4_mtp_global"}, {&d.kernels.attentionCombine, "g4_mtp_combine"}}
	if err = loadNamed(m.module, entries); err != nil {
		return err
	}
	for i := range m.layers {
		mt.partialStride = max(mt.partialStride, m.layers[i].partialFloats(m.config.NumHeads))
	}
	for _, item := range []struct {
		dst  *cuda.DevicePtr
		size int
	}{
		{&mt.seed, float32Bytes * m.config.HiddenSize}, {&mt.partial, float32Bytes * mtpVerifyTokens * mt.partialStride},
		{&mt.logits, float32Bytes * mtpVerifyTokens * m.config.VocabSize}, {&mt.states, int32Bytes * stateCount * mtpVerifyTokens}} {
		if *item.dst, err = m.memory.AllocZeroed(item.size); err != nil {
			return err
		}
	}
	if mt.host, err = cuda.AllocHost(int32Bytes * stateCount * mtpVerifyTokens); err != nil {
		return err
	}
	if d.hostState, err = cuda.AllocHost(int32Bytes * stateCount); err != nil {
		return err
	}
	if mt.hostDraft, err = cuda.AllocHost(int32Bytes * mtpVerifyTokens); err != nil {
		return err
	}
	for _, event := range []**cuda.Event{&mt.draftStart, &mt.verifyStart, &mt.verifyEnd} {
		if *event, err = cuda.NewEvent(); err != nil {
			return err
		}
	}
	// 候補数ごとの Graph を初期化時に作る。実行中に記録やメモリ確保は行わない。
	for count := 1; count <= mtpMaxDraftTokens; count++ {
		if err = m.stream.BeginCapture(); err != nil {
			return err
		}
		launcher := &gpucore.Launcher{Stream: m.stream}
		mt.enqueueChain(m, launcher, count)
		graph, endErr := m.stream.EndCapture()
		if launcher.Err != nil {
			if graph != nil {
				graph.Close()
			}
			return launcher.Err
		}
		if endErr != nil {
			return endErr
		}
		executable, instantiateErr := graph.Instantiate()
		graph.Close()
		if instantiateErr != nil {
			return instantiateErr
		}
		mt.graphs[count] = executable
	}
	return nil
}

// close は assistant が所有する Graph と固定メモリだけを解放します。共有 KV や stream は解放しません。
func (mt *mtpState) close() error {
	var first error
	keep := func(err error) {
		if first == nil {
			first = err
		}
	}
	if mt.integrated != nil {
		keep(mt.integrated.Close())
		mt.integrated = nil
	}
	for i, graph := range mt.graphs {
		if graph != nil {
			keep(graph.Close())
			mt.graphs[i] = nil
		}
	}
	for _, event := range []*cuda.Event{mt.draftStart, mt.verifyStart, mt.verifyEnd} {
		if event != nil {
			keep(event.Close())
		}
	}
	if mt.hostDraft != nil {
		keep(mt.hostDraft.Close())
		mt.hostDraft = nil
	}
	if mt.host != nil {
		keep(mt.host.Close())
		mt.host = nil
	}
	if mt.draft.hostState != nil {
		keep(mt.draft.hostState.Close())
		mt.draft.hostState = nil
	}
	return first
}

// enqueueDraft は連結射影、4 層、語彙出力、次の再帰特徴量を同一 Graph に積みます。
func (mt *mtpState) enqueueDraft(m *Model, l *gpucore.Launcher) {
	d := mt.draft
	b := &d.buffers
	l.Launch(mt.concat, gpucore.Dim1(gpucore.CeilDiv(2*m.config.HiddenSize, threadsPerBlock)), gpucore.Dim1(threadsPerBlock),
		new(cuda.KernelArgs).Pointer(m.embedding.Packed).Pointer(m.embedding.Scales).Pointer(b.state).Pointer(mt.seed).
			Int32(int32(m.config.HiddenSize)).Float32(m.config.EmbeddingScale()).Pointer(b.actQ).Pointer(b.actInfo))
	d.gemv(l, mt.pre, b.hidden)
	d.residualNorm(l, 0, 0, d.layers[0].attnNorm, noScale)
	for i := range d.layers {
		layer := &d.layers[i]
		d.gemv(l, layer.qkv, b.qkv)
		d.attend(l, layer)
		d.gemv(l, layer.o, b.delta)
		d.residualNorm(l, b.delta, layer.postAttnNorm, layer.ffnNorm, noScale)
		d.gemv(l, layer.gateUp, b.gateUp)
		d.geluMul(l)
		d.gemv(l, layer.down, b.delta)
		weight := d.outputNorm
		if i+1 < len(d.layers) {
			weight = d.layers[i+1].attnNorm
		}
		d.residualNorm(l, b.delta, layer.postFFNNorm, weight, layer.outScale)
	}
	d.gemv(l, d.embedding, b.logits)
	d.argmax(l)
	d.gemv(l, mt.post, mt.seed)
}

// seedFromHidden は受理した target の残差を最終正規化し、assistant の開始特徴量を作ります。
func (mt *mtpState) seedFromHidden(m *Model, hidden cuda.DevicePtr) error {
	args := new(cuda.KernelArgs).Pointer(hidden).Pointer(m.outputNorm).Float32(float32(m.config.NormEps)).
		Int32(int32(m.config.HiddenSize)).Pointer(mt.seed)
	return mt.normalize.Launch(gpucore.Dim1(1), gpucore.Dim1(normThreads), 0, m.stream, args)
}

// enqueueChain は候補列を GPU 上だけで作ります。中間の token・位置・特徴量を CPU へ戻しません。
func (mt *mtpState) enqueueChain(m *Model, l *gpucore.Launcher, count int) {
	l.Launch(mt.chainStart, gpucore.Dim1(1), gpucore.Dim1(1), new(cuda.KernelArgs).Pointer(mt.draft.buffers.state).Pointer(m.prefill.tokens))
	for index := 1; index <= count; index++ {
		mt.enqueueDraft(m, l)
		l.Launch(mt.chainNext, gpucore.Dim1(1), gpucore.Dim1(1), new(cuda.KernelArgs).Pointer(mt.draft.buffers.state).Pointer(m.prefill.tokens).Int32(int32(index)))
	}
}

// draftAndVerify は候補列を直接 target の検証入力へ渡し、最後に 1 回だけ同期します。
// GPU 時間は候補生成と検証を分け、wall 時間にはホストの起動・転送・同期も含めます。
func (mt *mtpState) draftAndVerify(m *Model, token, count int) ([]int, []int, float64, float64, float64, error) {
	started := time.Now()
	d := mt.draft
	d.position = m.position
	if err := d.writeInput(token); err != nil {
		return nil, nil, 0, 0, 0, err
	}
	if err := mt.draftStart.Record(m.stream); err != nil {
		return nil, nil, 0, 0, 0, err
	}
	if m.useGraph {
		if err := mt.graphs[count].Launch(m.stream); err != nil {
			return nil, nil, 0, 0, 0, err
		}
	} else {
		l := &gpucore.Launcher{Stream: m.stream}
		mt.enqueueChain(m, l, count)
		if l.Err != nil {
			return nil, nil, 0, 0, 0, l.Err
		}
	}
	if err := mt.verifyStart.Record(m.stream); err != nil {
		return nil, nil, 0, 0, 0, err
	}
	if err := mt.hostDraft.CopyFromDeviceAsync(m.stream, m.prefill.tokens); err != nil {
		return nil, nil, 0, 0, 0, err
	}
	outputs, err := m.verifyMTPDevice(count+1, true)
	if err != nil {
		return nil, nil, 0, 0, 0, err
	}
	wallMS := float64(time.Since(started)) / float64(time.Millisecond)
	draftMS, err := cuda.ElapsedMilliseconds(mt.draftStart, mt.verifyStart)
	if err != nil {
		return nil, nil, 0, 0, 0, err
	}
	verifyMS, err := cuda.ElapsedMilliseconds(mt.verifyStart, mt.verifyEnd)
	if err != nil {
		return nil, nil, 0, 0, 0, err
	}
	inputs := make([]int, count+1)
	for i := range inputs {
		inputs[i] = int(mt.hostDraft.Int32s()[i])
	}
	return inputs, outputs, draftMS, verifyMS, wallMS, nil
}

// invalidateMTP は先読みを破棄します。未確定の KV は可視範囲から外れ、次回入力で上書きされます。
func (m *Model) invalidateMTP() {
	if m.mtp == nil {
		return
	}
	mt := m.mtp
	mt.finishSelectionBatch(true)
	mt.pendingInputs = nil
	mt.pendingOutputs = nil
	mt.pendingRows = nil
	mt.seedValid = false
	mt.logitsRow = -1
}

// SetMTPEnabled は読み込み済み assistant の使用を切り替えます。比較測定で false -> true と切り替えられます。
func (m *Model) SetMTPEnabled(enabled bool) error {
	if m.mtp == nil {
		if enabled {
			return fmt.Errorf("MTP assistant is not loaded")
		}
		return nil
	}
	// 途中で無効化しても CopyLogits と ContinueGreedy が直前の論理位置を参照できるようにする。
	if m.mtp.logitsRow >= 0 {
		source := m.mtp.logits + cuda.DevicePtr(m.mtp.logitsRow*float32Bytes*m.config.VocabSize)
		if err := cuda.CopyDeviceAsync(m.stream, m.buffers.logits, source, float32Bytes*m.config.VocabSize); err != nil {
			return err
		}
	}
	if m.mtp.seedValid {
		if err := m.writeInput(m.mtp.expected); err != nil {
			return err
		}
	}
	m.invalidateMTP()
	m.mtp.enabled = enabled
	return nil
}

// MTPStats は確定済みバッチの累積候補・受理数を返します。選択駆動の未消費バッチは GenerationMetrics で終了時に確定します。
func (m *Model) MTPStats() MTPStatistics {
	if m.mtp == nil {
		return MTPStatistics{}
	}
	stats := m.mtp.stats
	stats.SelectionDriven = m.mtp.selectionDriven
	stats.Adaptive = m.mtp.adaptive
	stats.DraftLimit = m.mtp.fixedDraft
	return stats
}

// feedMTP は候補列を先読みし、貪欲または実際の選択トークンとの一致で検証行を再利用します。呼出元には通常の Feed と同じ 1 入力・1 出力・1 位置更新を見せます。
// 貪欲モードは本体 argmax、選択駆動モードは次回に渡された実トークンと候補を比較します。
func (m *Model) feedMTP(token int) (int, error) {
	mt := m.mtp
	if mt.history != nil {
		mt.history = append(mt.history, token)
	}
	if len(mt.pendingInputs) > 0 {
		if token == mt.pendingInputs[0] {
			next, row := mt.pendingOutputs[0], mt.pendingRows[0]
			mt.pendingInputs = mt.pendingInputs[1:]
			mt.pendingOutputs = mt.pendingOutputs[1:]
			mt.pendingRows = mt.pendingRows[1:]
			if mt.selectionBatch != nil {
				mt.selectionBatch.accepted++
			}
			m.position++
			mt.logitsRow = row
			mt.expected = next
			if len(mt.pendingInputs) == 0 {
				mt.finishSelectionBatch(false)
				if err := mt.seedFromHidden(m, m.prefill.hidden+cuda.DevicePtr(row*float32Bytes*m.config.HiddenSize)); err != nil {
					return 0, err
				}
			}
			return next, nil
		}
		if mt.selectionBatch != nil {
			mt.stats.SelectionRejected++
			mt.finishSelectionBatch(false)
		}
		m.invalidateMTP()
	}
	if !mt.seedValid || (!mt.selectionDriven && token != mt.expected) || m.maxContext-m.position < 2 {
		m.invalidateMTP()
		next, err := m.feedSingle(token)
		if err != nil {
			return 0, err
		}
		if err = mt.seedFromHidden(m, m.buffers.hidden); err != nil {
			return 0, err
		}
		mt.seedValid = true
		mt.expected = next
		mt.highWater = max(mt.highWater, m.position)
		return next, nil
	}
	contextBucket := mtpContextBucket(m.position)
	count := min(mt.chooseDraftCount(contextBucket), m.maxContext-m.position-1)
	var inputs, outputs []int
	var accepted int
	var draftMS, verifyMS, wallMS float64
	var err error
	var proposal []int
	if mt.history != nil && !mt.selectionDriven && !mt.adaptive {
		proposal = historyCandidates(mt.history, mt.lookupLimit, m.maxContext-m.position-1)
	}
	integrated := len(proposal) == 0 && mt.integrated != nil && m.useGraph && !mt.selectionDriven && !mt.adaptive && count == mtpDraftTokens
	if len(proposal) > 0 {
		count = len(proposal)
		inputs = append([]int{token}, proposal...)
		started := time.Now()
		outputs, err = m.verifyMTP(inputs)
		wallMS = time.Since(started).Seconds() * 1000
	} else if integrated {
		inputs, outputs, accepted, wallMS, err = mt.runIntegrated(m, token)
	} else {
		inputs, outputs, draftMS, verifyMS, wallMS, err = mt.draftAndVerify(m, token, count)
	}
	if err != nil {
		return 0, err
	}
	if len(proposal) > 0 {
		mt.stats.LookupBatches++
	}
	if !integrated {
		for accepted < count && inputs[accepted+1] == outputs[accepted] {
			accepted++
		}
	}
	retainedRows := accepted
	if mt.selectionDriven {
		// 本体 argmax で受理を決めない。呼出元が実際に選んだ token と次回入力時に比較する。
		mt.selectionBatch = &mtpSelectionBatch{bucket: contextBucket, count: count, draftMS: draftMS, verifyMS: verifyMS, wallMS: wallMS}
		retainedRows = count
	} else {
		mt.observeBatch(contextBucket, count, accepted, draftMS, verifyMS, wallMS)
	}
	mt.highWater = max(mt.highWater, m.position+len(inputs))
	for row := 1; row <= retainedRows; row++ {
		mt.pendingInputs = append(mt.pendingInputs, inputs[row])
		mt.pendingOutputs = append(mt.pendingOutputs, outputs[row])
		mt.pendingRows = append(mt.pendingRows, row)
	}
	m.position++
	mt.logitsRow = 0
	mt.expected = outputs[0]
	mt.seedValid = true
	if retainedRows == 0 {
		if err := mt.seedFromHidden(m, m.prefill.hidden); err != nil {
			return 0, err
		}
	}
	return outputs[0], nil
}

// prepareIntegrated は固定3候補から本体4行の検証、受理判定までを1本のGraphに記録する。
func (m *Model) prepareIntegrated() error {
	mt := m.mtp
	err := loadNamed(m.module, []namedKernel{{&mt.dynamicPrepare, "g4_attention_prepare_rows_dynamic"}, {&mt.dynamicLocal, "g4_verify_local_dynamic"}, {&mt.dynamicGlobal, "g4_verify_global_dynamic"}, {&mt.dynamicCombine, "g4_verify_combine_dynamic"}, {&mt.compact, "g4_accept_three"}})
	if err != nil {
		return err
	}
	mt.jointResult, err = m.memory.AllocZeroed(int32Bytes * stateCount * mtpVerifyTokens)
	if err != nil {
		return err
	}
	if err = m.stream.BeginCapture(); err != nil {
		return err
	}
	l := &gpucore.Launcher{Stream: m.stream}
	mt.enqueueChain(m, l, mtpDraftTokens)
	mt.capturing = true
	m.enqueueVerification(l, mtpDraftTokens+1)
	mt.capturing = false
	l.Launch(mt.compact, gpucore.Dim1(1), gpucore.Dim1(1), new(cuda.KernelArgs).Pointer(m.prefill.tokens).Pointer(mt.states).Int32(stateCount).Int32(stateArgmax).Pointer(mt.jointResult))
	graph, err := m.stream.EndCapture()
	if graph != nil {
		defer graph.Close()
	}
	if l.Err != nil {
		return l.Err
	}
	if err != nil {
		return err
	}
	mt.integrated, err = graph.Instantiate()
	return err
}

// runIntegrated は既知のトークンと位置を転送し、Graphの結果を1回だけ同期して読む。
func (mt *mtpState) runIntegrated(m *Model, token int) ([]int, []int, int, float64, error) {
	start := time.Now()
	mt.draft.position = m.position
	if err := mt.draft.writeInput(token); err != nil {
		return nil, nil, 0, 0, err
	}
	if err := mt.integrated.Launch(m.stream); err != nil {
		return nil, nil, 0, 0, err
	}
	if err := mt.host.CopyFromDeviceAsync(m.stream, mt.jointResult); err != nil {
		return nil, nil, 0, 0, err
	}
	if err := m.stream.Synchronize(); err != nil {
		return nil, nil, 0, 0, err
	}
	v := mt.host.Int32s()
	const rows = mtpDraftTokens + 1
	inputs, outputs := make([]int, rows), make([]int, rows)
	for row := 0; row < rows; row++ {
		inputs[row] = int(v[row])
		outputs[row] = int(v[rows+row])
	}
	mt.stats.IntegratedBatches++
	return inputs, outputs, int(v[2*rows]), time.Since(start).Seconds() * 1000, nil
}
