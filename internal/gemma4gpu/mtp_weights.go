package gemma4gpu

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/cardy0828/pascallm/internal/cuda"
	"github.com/cardy0828/pascallm/internal/gemma4"
	"github.com/cardy0828/pascallm/internal/gguf"
	"github.com/cardy0828/pascallm/internal/gpucore"
	"github.com/cardy0828/pascallm/internal/pq4"
)

const assistantHidden = 1024
const assistantFFN = 8192
const assistantLayers = 4
const mtpDraftTokens = 3
const mtpMaxDraftTokens = 7
const mtpVerifyTokens = mtpMaxDraftTokens + 1

// assistantLoader は検証済み Q4_0/F32 GGUF を GPU 用 PQ4 に値を変えず詰め替えます。
type assistantLoader struct {
	file   *gguf.File
	upload layerUploader
	err    error
}

// tensor は名前・型・寸法が一致するテンソルだけを返します。例: matrix [7680,1024]。
func (a *assistantLoader) tensor(name string, kind gguf.TensorType, dims ...int) []byte {
	if a.err != nil {
		return nil
	}
	t, err := a.file.Tensor(name)
	if err == nil && (t.Type != kind || len(t.Dims) != len(dims)) {
		err = fmt.Errorf("MTP tensor %s: incompatible type or dimensions", name)
	}
	if err == nil {
		for i, n := range dims {
			if t.Dims[i] != n {
				err = fmt.Errorf("MTP tensor %s: got %v, want %v", name, t.Dims, dims)
				break
			}
		}
	}
	var data []byte
	if err == nil {
		data, err = a.file.Bytes(t)
	}
	a.err = err
	return data
}

// vector は F32 ベクトルを読みます。例: output_norm.weight は 1024 要素です。
func (a *assistantLoader) vector(name string, n int) []float32 {
	data := a.tensor(name, gguf.TypeF32, n)
	if a.err != nil {
		return nil
	}
	out := make([]float32, n)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return out
}

// matrix は Q4_0 の係数・4bit 値を PQ4 配置へ移します。再量子化はしません。
func (a *assistantLoader) matrix(name string, rows, cols int) *pq4.Matrix {
	data := a.tensor(name, gguf.TypeQ4_0, cols, rows)
	if a.err != nil {
		return nil
	}
	result, err := pq4.FromQ4_0(rows, cols, data)
	a.err = err
	return result
}

// uploadMatrix は検証後の行列を転送します。前のエラーがあれば何もしません。
func (a *assistantLoader) uploadMatrix(name string, rows, cols int) gpucore.Matrix {
	matrix := a.matrix(name, rows, cols)
	if a.err != nil {
		return gpucore.Matrix{}
	}
	return a.upload.matrices(name, matrix)
}

// uploadVector は検証後の F32 ベクトルを転送します。
func (a *assistantLoader) uploadVector(name string, n int) cuda.DevicePtr {
	values := a.vector(name, n)
	if a.err != nil {
		return 0
	}
	return a.upload.vector(values)
}

// loadAssistant は Gemma4 12B 用の 4 層 assistant を読み、target の最後の SWA/global KV を共有します。
// 誤った組合せを黙って使わないよう、メタデータと全行列の形を確認します。
func (m *Model) loadAssistant(path string) (*mtpState, error) {
	f, err := gguf.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	arch, err := f.String("general.architecture")
	if err != nil {
		return nil, err
	}
	if arch != "gemma4-assistant" || m.config.HiddenSize != 3840 || len(m.layers) != 48 || m.config.VocabSize != 262144 {
		return nil, fmt.Errorf("MTP requires Gemma4 12B target and gemma4-assistant")
	}
	for key, want := range map[string]int64{"block_count": 4, "embedding_length": 1024, "embedding_length_out": 3840, "feed_forward_length": 8192, "attention.head_count": 16, "attention.sliding_window": 1024, "attention.shared_kv_layers": 4} {
		got, err := f.Int(arch + "." + key)
		if err != nil {
			return nil, err
		}
		if got != want {
			return nil, fmt.Errorf("MTP %s=%d, want %d", key, got, want)
		}
	}
	sliding, err := f.BoolArray(arch + ".attention.sliding_window_pattern")
	if err != nil {
		return nil, err
	}
	kv, err := f.IntArray(arch + ".attention.head_count_kv")
	if err != nil {
		return nil, err
	}
	if len(sliding) != assistantLayers || len(kv) != assistantLayers {
		return nil, fmt.Errorf("MTP invalid layer metadata")
	}
	for key, want := range map[string]float64{"attention.layer_norm_rms_epsilon": m.config.NormEps, "rope.freq_base": m.config.RopeBaseFull, "rope.freq_base_swa": m.config.RopeBaseSliding} {
		got, err := f.Float(arch + "." + key)
		if err != nil {
			return nil, err
		}
		if got != want {
			return nil, fmt.Errorf("MTP %s differs from target", key)
		}
	}
	config := *m.config
	config.HiddenSize = assistantHidden
	config.FeedForwardSize = assistantFFN
	config.FinalLogitSoftcap = 0
	config.Layers = make([]gemma4.LayerConfig, assistantLayers)
	d := &Model{config: &config, maxContext: m.maxContext, stream: m.stream, module: m.module, memory: m.memory, kernels: m.kernels}
	a := &assistantLoader{file: f, upload: layerUploader{memory: m.memory}}
	mt := &mtpState{draft: d, enabled: true, logitsRow: -1}
	mt.pre = a.uploadMatrix("nextn.pre_projection.weight", assistantHidden, 2*m.config.HiddenSize)
	mt.post = a.uploadMatrix("nextn.post_projection.weight", m.config.HiddenSize, assistantHidden)
	d.embedding = a.uploadMatrix("token_embd.weight", m.config.VocabSize, assistantHidden)
	d.outputNorm = a.uploadVector("output_norm.weight", assistantHidden)
	d.layers = make([]deviceLayer, assistantLayers)
	for i := range d.layers {
		isSliding := i < assistantLayers-1
		source := len(m.layers) - 1
		if isSliding {
			source--
		}
		target := m.layers[source]
		if sliding[i] != isSliding || int(kv[i]) != target.config.KVHeads || target.config.Sliding != isSliding {
			return nil, fmt.Errorf("MTP layer %d is incompatible with shared target KV", i)
		}
		layer := target
		base := config.RopeBaseSliding
		var factors []float32
		if !isSliding {
			base = config.RopeBaseFull
			factors = a.vector("rope_freqs.weight", layer.config.HeadDim/2)
		}
		inverse := make([]float32, layer.config.HeadDim/2)
		theta := float32(math.Pow(float64(float32(base)), -2/float64(layer.config.HeadDim)))
		for j := range inverse {
			inverse[j] = float32(math.Pow(float64(theta), float64(j)))
			if factors != nil {
				inverse[j] /= factors[j]
			}
		}
		layer.ropeInverse = a.upload.vector(inverse)
		config.Layers[i] = target.config
		name := func(suffix string) string { return fmt.Sprintf("blk.%d.%s", i, suffix) }
		layer.attnNorm = a.uploadVector(name("attn_norm.weight"), assistantHidden)
		layer.postAttnNorm = a.uploadVector(name("post_attention_norm.weight"), assistantHidden)
		layer.ffnNorm = a.uploadVector(name("ffn_norm.weight"), assistantHidden)
		layer.postFFNNorm = a.uploadVector(name("post_ffw_norm.weight"), assistantHidden)
		layer.qNorm = a.uploadVector(name("attn_q_norm.weight"), layer.config.HeadDim)
		layer.qkvWidth = layer.queryWidth(config.NumHeads)
		layer.qkv = a.uploadMatrix(name("attn_q.weight"), layer.qkvWidth, assistantHidden)
		layer.o = a.uploadMatrix(name("attn_output.weight"), assistantHidden, layer.qkvWidth)
		gate := a.matrix(name("ffn_gate.weight"), assistantFFN, assistantHidden)
		up := a.matrix(name("ffn_up.weight"), assistantFFN, assistantHidden)
		if a.err == nil {
			layer.gateUp = a.upload.matrices("mtp_gate_up", gate, up)
		}
		layer.down = a.uploadMatrix(name("ffn_down.weight"), assistantHidden, assistantFFN)
		values := a.vector(name("layer_output_scale.weight"), 1)
		if a.err == nil {
			layer.outScale = values[0]
		}
		d.layers[i] = layer
	}
	if a.err != nil {
		return nil, a.err
	}
	if a.upload.err != nil {
		return nil, a.upload.err
	}
	if err = d.allocateActivations(); err != nil {
		return nil, err
	}
	if err = mt.prepare(m); err != nil {
		mt.close()
		return nil, err
	}
	return mt, nil
}
