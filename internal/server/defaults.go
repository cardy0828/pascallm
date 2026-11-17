package server

import (
	"encoding/json"
	"fmt"
	"os"
)

// Defaults は、要求で指定がないときに使う値です。
type Defaults struct {
	Temperature       float64 // 例: 0.1
	TopK              int     // 例: 50（0 は制限なし）
	TopP              float64 // 例: 1（制限なし）
	MinP              float64 // 例: 0（制限なし）
	RepetitionPenalty float64 // 例: 1.1
	EnableThinking    bool    // chat_template_kwargs.enable_thinking の既定値（テンプレートの既定は true）
}

// BaseDefaults は、generation_config.json に項目がないときの値です（Hugging Face の generate の既定値と同じ）。
var BaseDefaults = Defaults{Temperature: 1, TopK: 0, TopP: 1, MinP: 0, RepetitionPenalty: 1, EnableThinking: true}

// LoadGenerationConfig は Hugging Face の generation_config.json を読み、base のうち書かれている項目を置き換えます。
// do_sample が false なら temperature を 0（最大値を選ぶ）にします。
// 例: LFM3 の generation_config.json → temperature 0.1、top_k 50、repetition_penalty 1.1
func LoadGenerationConfig(path string, base Defaults) (Defaults, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return base, err
	}
	var file struct {
		DoSample          *bool    `json:"do_sample"`
		Temperature       *float64 `json:"temperature"`
		TopK              *int     `json:"top_k"`
		TopP              *float64 `json:"top_p"`
		MinP              *float64 `json:"min_p"`
		RepetitionPenalty *float64 `json:"repetition_penalty"`
	}
	if err := json.Unmarshal(data, &file); err != nil {
		return base, fmt.Errorf("%s: %w", path, err)
	}
	defaults := base
	defaults.Temperature = valueOr(file.Temperature, defaults.Temperature)
	defaults.TopK = valueOr(file.TopK, defaults.TopK)
	defaults.TopP = valueOr(file.TopP, defaults.TopP)
	defaults.MinP = valueOr(file.MinP, defaults.MinP)
	defaults.RepetitionPenalty = valueOr(file.RepetitionPenalty, defaults.RepetitionPenalty)
	if file.DoSample != nil && !*file.DoSample {
		defaults.Temperature = 0
	}
	return defaults, nil
}
