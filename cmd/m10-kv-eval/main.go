// Command m10-kv-eval は同じトークン列の末尾を teacher forcing で評価し、KV 精度だけの差を集計します。
package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"github.com/cardy0828/pascallm/internal/compare"
	"github.com/cardy0828/pascallm/internal/gemma4gpu"
)

// evaluationCase は固定トークン列です。例: name=japanese、tokens=[2,...]。
type evaluationCase struct {
	Name   string `json:"name"`
	Tokens []int  `json:"tokens"`
}

// run は参照 logits の保存または比較を行います。参照は同じ量子化の K/V FP32 で、元モデルの完全精度ではありません。
func run() error {
	casesPath := flag.String("cases", "", "固定入力 JSON")
	directory := flag.String("reference", "", "参照 logits の保存先/読込先")
	save := flag.Bool("save", false, "参照を保存する")
	format := flag.String("kv-cache", "fp16", "fp16、k-fp32、fp32")
	tail := flag.Int("tail", 64, "末尾で測る位置数")
	flag.Parse()
	if *tail < 1 || *directory == "" {
		return fmt.Errorf("tail > 0 and reference are required")
	}
	if *save && *format != "fp32" {
		return fmt.Errorf("reference must use fp32 KV")
	}
	data, err := os.ReadFile(*casesPath)
	if err != nil {
		return err
	}
	var cases []evaluationCase
	if err = json.Unmarshal(data, &cases); err != nil {
		return err
	}
	if len(cases) == 0 {
		return fmt.Errorf("empty cases")
	}
	if err = os.MkdirAll(*directory, 0755); err != nil {
		return err
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	model, err := gemma4gpu.Open("./data/gemma4-12b-qat.pq4m", gemma4gpu.Options{DeviceUUID: "", MaxContext: 8192, KVCache: *format})
	if err != nil {
		return err
	}
	defer model.Close()
	for _, c := range cases {
		if filepath.Base(c.Name) != c.Name || len(c.Tokens) <= *tail || len(c.Tokens) > model.MaxContext() {
			return fmt.Errorf("invalid case %q", c.Name)
		}
		if err = model.Reset(); err != nil {
			return err
		}
		start := len(c.Tokens) - *tail - 1
		begun := time.Now()
		if err = model.Prefill(c.Tokens[:start]); err != nil {
			return err
		}
		prefillSeconds := time.Since(begun).Seconds()
		path := filepath.Join(*directory, c.Name+".f32")
		var file *os.File
		if *save {
			file, err = os.Create(path)
		} else {
			file, err = os.Open(path)
		}
		if err != nil {
			return err
		}
		accumulator := new(compare.Accumulator)
		reference := make([]float32, model.Config().VocabSize)
		logProbs := make([]float64, len(reference))
		for position := start; position < len(c.Tokens)-1; position++ {
			logits, forwardErr := model.Forward(c.Tokens[position])
			if forwardErr != nil {
				file.Close()
				return forwardErr
			}
			if *save {
				err = binary.Write(file, binary.LittleEndian, logits)
			} else {
				err = binary.Read(file, binary.LittleEndian, reference)
				if err == nil {
					compare.LogSoftmax(reference, logProbs)
					for i, v := range logProbs {
						reference[i] = float32(v)
					}
					accumulator.Add(reference, float32(math.Inf(-1)), logits, c.Tokens[position+1])
				}
			}
			if err != nil {
				file.Close()
				return err
			}
		}
		if err = file.Close(); err != nil {
			return err
		}
		if err = json.NewEncoder(os.Stdout).Encode(map[string]any{"case": c.Name, "kv_cache": *format, "tokens": len(c.Tokens), "prefill_seconds": prefillSeconds, "gpu_bytes": model.AllocatedBytes(), "saved_reference": *save, "comparison": accumulator.Summary()}); err != nil {
			return err
		}
	}
	return nil
}

// main は失敗を終了コード 1 として呼出元へ返します。
func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
