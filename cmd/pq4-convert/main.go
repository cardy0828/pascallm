// Command pq4-convert は、Hugging Face 形式の LFM3（config.json + model.safetensors）を PQ4 のモデルファイル（.pq4m）に変換します。
//
// 保存形式の方針（docs/ARCHITECTURE.md「4.3 混合精度」）:
//   - nn.Linear の重み: PQ4（ただしヘッドごとのゲート gate_proj は -gate-kind で選択、既定は F16）
//   - 埋め込み表: F16（1 トークン 1 行しか読まないため精度を落とさない）
//   - 語彙出力: 埋め込み表を -lm-head-kind の形式（既定は PQ4）にしたもの（lm_head.weight として保存）
//   - 正規化の重み・畳み込みの係数: F32
//
// M5 で加えた設定（docs/ARCHITECTURE.md「4.4 第 2 段階の候補」）:
//   - -method search: 係数の候補を細かく試し、割り当てごとに最適な係数を求め直す
//   - -imatrix: 入力の列ごとの重要度（tools/reference/m5_reference.py imatrix の出力）で誤差を重み付けする（search のとき）
//   - -emulate-pq4s: 係数を PQ4S の形（8 ブロックごとの共通の係数 × 6bit の整数）に制限して品質を測る
//   - -pq8: 名前が正規表現に合う重みを PQ8（約 8bit、混合精度）にする。例: 'feed_forward\.w2|lm_head'
//     GPU は q・k・v と W1・W3 をつないで計算するため、つなぐ重みは同じ形式にする必要がある
//
// 実行例:
//
//	go run ./cmd/pq4-convert -model-dir ./models/lfm3-1.7b-training \
//	    -out ./data/lfm3-1.7b-pq4-search-imatrix.pq4m \
//	    -imatrix ./data/m5/imatrix.safetensors
//
// 進み具合と、テンソルごとの量子化誤差を 1 行 1 件の JSON で標準出力に書きます。
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/cardy0828/pascallm/internal/lfm3"
	"github.com/cardy0828/pascallm/internal/numeric"
	"github.com/cardy0828/pascallm/internal/pq4"
	"github.com/cardy0828/pascallm/internal/safetensors"
)

// options はコマンドライン引数です。
type options struct {
	modelDir   string
	out        string
	method     pq4.Method
	gateKind   pq4.TensorKind
	lmHeadKind pq4.TensorKind
	// imatrixPath は重要度のファイル（safetensors、重みの名前ごとに入力の列の数の float32）です。空なら使いません。
	imatrixPath string
	emulatePQ4S bool
	pq8Pattern  *regexp.Regexp // PQ8 にする重みの名前（nil なら使わない）
}

// parseMatrixKind は、行列の保存形式の引数（f16 または pq4）を確かめます。
// flagName はエラーに出す引数名です。例: parseMatrixKind("-gate-kind", "f16")
func parseMatrixKind(flagName, text string) (pq4.TensorKind, error) {
	kind := pq4.TensorKind(text)
	if kind != pq4.KindF16 && kind != pq4.KindPQ4 {
		return "", fmt.Errorf("%s must be f16 or pq4, got %q", flagName, text)
	}
	return kind, nil
}

// parseOptions はコマンドライン引数を読み、値を確かめます。
func parseOptions() (options, error) {
	modelDir := flag.String("model-dir", "", "変換元のディレクトリ（config.json と model.safetensors がある場所）")
	out := flag.String("out", "", "出力する .pq4m ファイルのパス")
	// 既定は search: 自然な文章の評価で mse より KL が小さかった（imatrix なしで 0.127、ありで 0.099。
	// results/20261117-m5/README.md）。
	method := flag.String("method", string(pq4.MethodSearch), "係数の決め方: search（候補を細かく試し係数を求め直す）、mse（二乗誤差が最小の候補）、absmax（Q4_0 と同じ）")
	gateKind := flag.String("gate-kind", string(pq4.KindF16), "attention のゲート（gate_proj）の保存形式: f16 または pq4")
	lmHeadKind := flag.String("lm-head-kind", string(pq4.KindPQ4), "語彙出力（lm_head）の保存形式: pq4 または f16")
	imatrixPath := flag.String("imatrix", "", "重要度（imatrix）の safetensors。-method search のときに誤差の重みに使う")
	emulatePQ4S := flag.Bool("emulate-pq4s", false, "係数を PQ4S の形（8 ブロックごとの共通の係数 × 6bit の整数）に制限する")
	pq8Pattern := flag.String("pq8", "", "PQ8（約 8bit）にする重みの名前の正規表現（空なら使わない）。例: 'feed_forward\\.w2|lm_head'")
	flag.Parse()
	if *modelDir == "" || *out == "" {
		return options{}, fmt.Errorf("-model-dir and -out are required")
	}
	opts := options{modelDir: *modelDir, out: *out, imatrixPath: *imatrixPath, emulatePQ4S: *emulatePQ4S}
	var err error
	if opts.method, err = pq4.ParseMethod(*method); err != nil {
		return options{}, err
	}
	if opts.gateKind, err = parseMatrixKind("-gate-kind", *gateKind); err != nil {
		return options{}, err
	}
	if opts.lmHeadKind, err = parseMatrixKind("-lm-head-kind", *lmHeadKind); err != nil {
		return options{}, err
	}
	if opts.imatrixPath != "" && opts.method != pq4.MethodSearch {
		return options{}, fmt.Errorf("-imatrix needs -method search")
	}
	if *pq8Pattern != "" {
		if opts.pq8Pattern, err = regexp.Compile(*pq8Pattern); err != nil {
			return options{}, fmt.Errorf("-pq8: %w", err)
		}
	}
	return opts, nil
}

// emit は 1 件の記録を JSON の 1 行として標準出力に書きます。
func emit(record map[string]any) {
	line, err := json.Marshal(record)
	if err != nil {
		fmt.Fprintf(os.Stderr, "json encode error: %v\n", err)
		return
	}
	fmt.Println(string(line))
}

// fileSHA256 はファイルの SHA-256 を 16 進文字列で返します（変換元の記録用）。
func fileSHA256(path string) (string, error) {
	handle, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer handle.Close()
	digest := sha256.New()
	if _, err := io.Copy(digest, handle); err != nil {
		return "", err
	}
	return hex.EncodeToString(digest.Sum(nil)), nil
}

// relativeError は、元の重みと PQ4 で復元した重みの相対 RMS 誤差 sqrt(Σ(w-ŵ)² / Σw²) を返します。
// 例: 0.1 は、誤差の大きさが重みの大きさの 10% であることを表す。
func relativeError(matrix *pq4.Matrix, original []float32) float64 {
	row := make([]float32, matrix.Cols())
	var errorSquares, valueSquares float64
	for index := 0; index < matrix.Rows(); index++ {
		matrix.DequantizeRow(index, row)
		for col, restored := range row {
			value := float64(original[index*matrix.Cols()+col])
			difference := value - float64(restored)
			errorSquares += difference * difference
			valueSquares += value * value
		}
	}
	if valueSquares == 0 {
		return 0
	}
	return math.Sqrt(errorSquares / valueSquares)
}

// patternText は正規表現の文字列を返します（nil なら空）。
func patternText(pattern *regexp.Regexp) string {
	if pattern == nil {
		return ""
	}
	return pattern.String()
}

// isGateProjection は、ヘッドごとのゲートの重みかを名前で判定します。
func isGateProjection(name string) bool {
	return strings.HasSuffix(name, ".self_attn.gate_proj.weight")
}

// converter は、変換中の状態（出力するテンソルと集計）をまとめます。
type converter struct {
	opts        options
	source      *lfm3.SafetensorsSource
	importance  *safetensors.File // -imatrix のファイル（なければ nil）
	tensors     []pq4.TensorData
	pq4Elements int64
	pq4Bytes    int64
	pq8Elements int64
	pq8Bytes    int64
	totalBytes  int64
}

// add は 1 テンソルを出力に加え、サイズを集計します。
func (c *converter) add(tensor pq4.TensorData) {
	c.tensors = append(c.tensors, tensor)
	switch tensor.Kind {
	case pq4.KindPQ4:
		bytes := int64(len(tensor.PQ4.Packed()) + 2*len(tensor.PQ4.Scales()))
		c.pq4Elements += int64(tensor.PQ4.Rows() * tensor.PQ4.Cols())
		c.pq4Bytes += bytes
		c.totalBytes += bytes
	case pq4.KindPQ8:
		bytes := int64(len(tensor.PQ4.Packed()) + 2*len(tensor.PQ4.Scales()))
		c.pq8Elements += int64(tensor.PQ4.Rows() * tensor.PQ4.Cols())
		c.pq8Bytes += bytes
		c.totalBytes += bytes
	case pq4.KindF16:
		c.totalBytes += int64(2 * len(tensor.F16))
	case pq4.KindF32:
		c.totalBytes += int64(4 * len(tensor.F32))
	}
}

// importanceOf は、重み name の入力の列ごとの重要度（長さ cols）を返します。-imatrix がなければ nil です。
// 重要度のファイルにない重みはエラーにします（較正の漏れに気付くため）。
func (c *converter) importanceOf(name string, cols int) ([]float32, error) {
	if c.importance == nil {
		return nil, nil
	}
	values, err := c.importance.Float32(name, []int{cols})
	if err != nil {
		return nil, fmt.Errorf("imatrix: %w", err)
	}
	return values, nil
}

// addQuantized は 2 次元の重みを PQ4（-pq8 に合えば PQ8）に量子化して加え、誤差を出力します。
func (c *converter) addQuantized(name string, rows, cols int, values []float32) error {
	if c.opts.pq8Pattern != nil && c.opts.pq8Pattern.MatchString(name) {
		matrix, err := pq4.QuantizePQ8Matrix(rows, cols, values)
		if err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
		c.add(pq4.TensorData{Name: name, Kind: pq4.KindPQ8, Shape: []int{rows, cols}, PQ4: matrix})
		emit(map[string]any{"kind": "tensor", "name": name, "format": "pq8", "shape": []int{rows, cols},
			"relative_rms_error": relativeError(matrix, values)})
		return nil
	}
	importance, err := c.importanceOf(name, cols)
	if err != nil {
		return err
	}
	matrix, err := pq4.QuantizeMatrixWith(rows, cols, values, pq4.QuantizeOptions{
		Method: c.opts.method, Importance: importance, EmulatePQ4S: c.opts.emulatePQ4S})
	if err != nil {
		return fmt.Errorf("%s: %w", name, err)
	}
	c.add(pq4.TensorData{Name: name, Kind: pq4.KindPQ4, Shape: []int{rows, cols}, PQ4: matrix})
	emit(map[string]any{"kind": "tensor", "name": name, "format": "pq4", "shape": []int{rows, cols},
		"relative_rms_error": relativeError(matrix, values), "imatrix": importance != nil})
	return nil
}

// addFloat16 は重みを FP16 に丸めて加えます。
func (c *converter) addFloat16(name string, shape []int, values []float32) {
	bits := make([]uint16, len(values))
	for index, value := range values {
		bits[index] = numeric.Float32ToFloat16(value)
	}
	c.add(pq4.TensorData{Name: name, Kind: pq4.KindF16, Shape: shape, F16: bits})
	emit(map[string]any{"kind": "tensor", "name": name, "format": "f16", "shape": shape})
}

// convertSpec は、モデルが必要とする 1 テンソルを読み、方針に従った形式で加えます。
func (c *converter) convertSpec(spec lfm3.TensorSpec, embeddingName string) error {
	values, err := c.source.Vector(spec.Name, spec.Shape)
	if err != nil {
		return err
	}
	switch {
	case spec.Name == embeddingName:
		// 埋め込み表は F16 で保存し、同じ値から語彙出力（-lm-head-kind の形式）を作る。
		c.addFloat16(spec.Name, spec.Shape, values)
		if c.opts.lmHeadKind == pq4.KindF16 {
			c.addFloat16(lfm3.LMHeadTensorName, spec.Shape, values)
			return nil
		}
		return c.addQuantized(lfm3.LMHeadTensorName, spec.Shape[0], spec.Shape[1], values)
	case spec.Linear && isGateProjection(spec.Name) && c.opts.gateKind == pq4.KindF16:
		c.addFloat16(spec.Name, spec.Shape, values)
		return nil
	case spec.Linear:
		return c.addQuantized(spec.Name, spec.Shape[0], spec.Shape[1], values)
	default:
		c.add(pq4.TensorData{Name: spec.Name, Kind: pq4.KindF32, Shape: spec.Shape, F32: values})
		return nil
	}
}

// run は変換全体を実行します。
func run(opts options) error {
	started := time.Now()
	config, source, err := lfm3.OpenSafetensorsModel(opts.modelDir)
	if err != nil {
		return err
	}
	defer source.File.Close()
	rawConfig, err := os.ReadFile(filepath.Join(opts.modelDir, "config.json"))
	if err != nil {
		return err
	}
	safetensorsPath := filepath.Join(opts.modelDir, "model.safetensors")
	sourceHash, err := fileSHA256(safetensorsPath)
	if err != nil {
		return err
	}
	specs, err := lfm3.TensorSpecs(config)
	if err != nil {
		return err
	}

	c := &converter{opts: opts, source: source}
	imatrixHash := ""
	if opts.imatrixPath != "" {
		if c.importance, err = safetensors.Open(opts.imatrixPath); err != nil {
			return err
		}
		defer c.importance.Close()
		if imatrixHash, err = fileSHA256(opts.imatrixPath); err != nil {
			return err
		}
	}
	for _, spec := range specs {
		if err := c.convertSpec(spec, specs[0].Name); err != nil {
			return err
		}
	}

	header := pq4.FileHeader{
		ModelConfig: json.RawMessage(rawConfig),
		Source: map[string]string{
			"model_dir":          opts.modelDir,
			"safetensors_sha256": sourceHash,
		},
		Converter: map[string]string{
			"tool":           "pascal-engine/cmd/pq4-convert",
			"method":         string(opts.method),
			"gate_kind":      string(opts.gateKind),
			"embedding":      string(pq4.KindF16),
			"lm_head":        string(opts.lmHeadKind),
			"imatrix":        opts.imatrixPath,
			"imatrix_sha256": imatrixHash,
			"emulate_pq4s":   fmt.Sprint(opts.emulatePQ4S),
			"pq8_pattern":    patternText(opts.pq8Pattern),
			"created_at":     time.Now().UTC().Format(time.RFC3339),
		},
	}
	if err := pq4.WriteFile(opts.out, header, c.tensors); err != nil {
		return err
	}
	stat, err := os.Stat(opts.out)
	if err != nil {
		return err
	}
	emit(map[string]any{
		"kind":                "summary",
		"out":                 opts.out,
		"file_bytes":          stat.Size(),
		"tensor_bytes":        c.totalBytes,
		"pq4_elements":        c.pq4Elements,
		"pq4_bytes":           c.pq4Bytes,
		"pq4_bits_per_weight": float64(8*c.pq4Bytes) / float64(c.pq4Elements),
		"method":              opts.method,
		"imatrix":             opts.imatrixPath,
		"emulate_pq4s":        opts.emulatePQ4S,
		"pq8_pattern":         patternText(opts.pq8Pattern),
		"pq8_elements":        c.pq8Elements,
		"pq8_bytes":           c.pq8Bytes,
		"bits_per_weight":     float64(8*(c.pq4Bytes+c.pq8Bytes)) / float64(c.pq4Elements+c.pq8Elements),
		"gate_kind":           opts.gateKind,
		"seconds":             time.Since(started).Seconds(),
	})
	return nil
}

func main() {
	opts, err := parseOptions()
	if err == nil {
		err = run(opts)
	}
	if err != nil {
		emit(map[string]any{"kind": "result", "ok": false, "error": err.Error()})
		os.Exit(1)
	}
	emit(map[string]any{"kind": "result", "ok": true})
}
