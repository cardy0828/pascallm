package gemma4gpu

import (
	"fmt"

	"github.com/cardy0828/pascallm/internal/cuda"
	"github.com/cardy0828/pascallm/internal/gpucore"
)

// attendVerify は target の各候補位置を通常生成と同じ split attention で検証します。
// KV は行ごとの位置へ書きますが、各 Q が見るのは自身の位置までなので未来の候補は参照しません。
func (m *Model) attendVerify(l *gpucore.Launcher, layer *deviceLayer, start, count int) {
	p := &m.prefill
	mt := m.mtp
	c := m.config
	hd := layer.config.HeadDim
	kv := layer.config.KVHeads
	prepare := new(cuda.KernelArgs).Pointer(p.qkv).Pointer(layer.qNorm).Pointer(layer.kNorm).Pointer(layer.ropeInverse).
		Float32(float32(c.NormEps)).Int32(int32(start)).Int32(int32(hd)).Int32(int32(c.NumHeads)).Int32(int32(kv)).
		Int32(int32(layer.valueOffset)).Int32(int32(layer.qkvWidth)).Pointer(p.query).Pointer(layer.keyCache).Pointer(layer.valueCache).Int32(int32(layer.ringSize))
	l.Launch(m.prefillKernels.prepareRows, gpucore.Dim2(m.prepareVectorBlocks(layer), count), gpucore.Dim1(threadsPerBlock), prepare)
	kernel := mt.verifyGlobal
	if layer.config.Sliding {
		kernel = mt.verifyLocal
	}
	body := new(cuda.KernelArgs).Pointer(p.query).Pointer(layer.keyCache).Pointer(layer.valueCache).Int32(int32(start)).
		Int32(int32(layer.window)).Int32(int32(layer.ringSize)).Int32(int32(kv)).Int32(int32(c.NumHeads / kv)).Float32(attentionScale).
		Pointer(mt.partial).Int32(int32(c.NumHeads)).Int32(int32(mt.partialStride))
	grid := cuda.Dim3{X: uint32(c.NumHeads / layer.groupSize), Y: uint32(layer.maxSplits), Z: uint32(count)}
	l.Launch(kernel, grid, gpucore.Dim1(threadsPerBlock), body)
	combine := new(cuda.KernelArgs).Pointer(mt.partial).Int32(int32(start)).Int32(int32(layer.window)).Int32(int32(layer.maxSplits)).
		Int32(int32(hd)).Int32(int32(layer.groupSize)).Int32(int32(c.NumHeads)).Pointer(p.actQ).Pointer(p.actInfo).Int32(int32(mt.partialStride))
	l.Launch(mt.verifyCombine, gpucore.Dim2(layer.queryWidth(c.NumHeads)/threadsPerBlock, count), gpucore.Dim1(threadsPerBlock), combine)
}

// verifyMTP は最大 8 入力の本体 argmax を返します。論理位置は変更せず、feedMTP が受理済み行だけ公開します。
// 行列積・attention・正規化の加算順を 1 トークン生成にそろえます。
func (m *Model) verifyMTP(tokens []int) ([]int, error) {
	if len(tokens) < 1 || len(tokens) > mtpVerifyTokens || m.position+len(tokens) > m.maxContext {
		return nil, fmt.Errorf("MTP invalid verification batch")
	}
	for i, token := range tokens {
		if token < 0 || token >= m.config.VocabSize {
			return nil, fmt.Errorf("MTP token outside vocabulary")
		}
		m.hostPrefillTokens.Int32s()[i] = int32(token)
	}
	if err := m.hostPrefillTokens.CopyPrefixToDeviceAsync(m.stream, m.prefill.tokens, int32Bytes*len(tokens)); err != nil {
		return nil, err
	}
	return m.verifyMTPDevice(len(tokens), false)
}

// verifyMTPDevice は GPU 上の prefill.tokens をそのまま検証します。位置・語彙の検査は呼出元で済ませます。
// timed が true の場合は検証終了を CUDA event に記録し、候補列の読み戻しも含めて最後に同期します。
func (m *Model) verifyMTPDevice(count int, timed bool) ([]int, error) {
	l := &gpucore.Launcher{Stream: m.stream}
	m.enqueueRows(l, count, len(m.layers), true)
	last := &m.layers[len(m.layers)-1]
	m.residualNormRows(l, count, m.prefill.delta, last.postFFNNorm, m.outputNorm, last.outScale)
	m.gemmRows(l, m.embedding, m.mtp.logits, count)
	l.Launch(m.kernels.softcap, gpucore.Dim1(gpucore.CeilDiv(count*m.config.VocabSize, threadsPerBlock)), gpucore.Dim1(threadsPerBlock),
		new(cuda.KernelArgs).Pointer(m.mtp.logits).Int32(int32(count*m.config.VocabSize)).Float32(float32(m.config.FinalLogitSoftcap)))
	// argmax の作業領域は同じ stream 内で順に使い回します。各行の結果だけ別の状態に残します。
	original := m.buffers
	for row := 0; row < count; row++ {
		m.buffers.logits = m.mtp.logits + cuda.DevicePtr(row*float32Bytes*m.config.VocabSize)
		m.buffers.state = m.mtp.states + cuda.DevicePtr(row*int32Bytes*stateCount)
		m.argmax(l)
	}
	m.buffers = original
	if l.Err != nil {
		return nil, l.Err
	}
	if timed {
		if err := m.mtp.verifyEnd.Record(m.stream); err != nil {
			return nil, err
		}
	}
	if err := m.mtp.host.CopyFromDeviceAsync(m.stream, m.mtp.states); err != nil {
		return nil, err
	}
	if err := m.stream.Synchronize(); err != nil {
		return nil, err
	}
	out := make([]int, count)
	for row := range out {
		out[row] = int(m.mtp.host.Int32s()[row*stateCount+stateArgmax])
	}
	return out, nil
}

// DebugMTPVerify は検証経路の全 logits を返します。通常生成との一致を検査する診断専用です。
// 論理位置は変わりませんが、指定位置の KV を上書きするため、検査後は Reset/Prefill してください。
func (m *Model) DebugMTPVerify(tokens []int) ([][]float32, error) {
	if m.mtp == nil {
		return nil, fmt.Errorf("MTP not loaded")
	}
	if _, err := m.verifyMTP(tokens); err != nil {
		return nil, err
	}
	values := make([]float32, len(tokens)*m.config.VocabSize)
	if err := cuda.CopyFromDevice(m.stream, values, m.mtp.logits); err != nil {
		return nil, err
	}
	rows := make([][]float32, len(tokens))
	for i := range rows {
		rows[i] = values[i*m.config.VocabSize : (i+1)*m.config.VocabSize]
	}
	return rows, nil
}
