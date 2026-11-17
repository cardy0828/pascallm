// tokenizer-check は、Go のトークナイザーとチャットテンプレート（M3、ツールは M6）が Hugging Face の実装と
// 同じ結果になるかを、tools/reference/tokenizer_golden.py・m6_template_golden.py の出力と突き合わせて確かめます。
// 会話は API サーバーと同じ変換（server.RenderChat）を通してから比べます。
// 一致しない件があれば最初の数件を表示し、終了コード 1 で終わります。
//
// 実行例（Linux）:
//
//	go run ./cmd/tokenizer-check -tokenizer ./models/lfm3-1.7b-training/tokenizer.json \
//	    -golden ./data/tokenizer-golden.jsonl
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/cardy0828/pascallm/internal/server"
	"github.com/cardy0828/pascallm/internal/tokenizer"
)

// maxReportedMismatches は、表示する不一致の最大件数です。
const maxReportedMismatches = 10

// maxLineBytes は、基準の 1 行の最大の長さです（コーパス全体を 1 件にした行が入る大きさ）。
const maxLineBytes = 64 << 20

// goldenRecord は基準の 1 件です（kind が "text" または "chat"）。
type goldenRecord struct {
	Kind           string          `json:"kind"`
	Text           string          `json:"text"`
	IDs            []int           `json:"ids"`
	Decoded        string          `json:"decoded"`
	Messages       json.RawMessage `json:"messages"` // OpenAI の API の形の messages
	Tools          json.RawMessage `json:"tools"`    // OpenAI の API の形の tools（なければ null）
	EnableThinking bool            `json:"enable_thinking"`
	Rendered       string          `json:"rendered"`
}

// summary は突き合わせの結果の集計です。
type summary struct {
	Texts, Chats     int
	TextIDMismatches int
	DecodeMismatches int
	RenderMismatches int
	ChatIDMismatches int
	TextTokens       int
	EncodeSeconds    float64
}

// main は引数を読み、突き合わせを行います。
func main() {
	tokenizerPath := flag.String("tokenizer", "", "tokenizer.json")
	goldenPath := flag.String("golden", "", "tools/reference/tokenizer_golden.py の出力")
	flag.Parse()
	result, err := run(*tokenizerPath, *goldenPath)
	encoded, _ := json.Marshal(result)
	fmt.Println(string(encoded))
	if err != nil {
		fmt.Fprintln(os.Stderr, "tokenizer-check:", err)
		os.Exit(1)
	}
}

// run は基準の全件を突き合わせ、集計を返します。不一致があればエラーを返します。
func run(tokenizerPath, goldenPath string) (summary, error) {
	var result summary
	tok, err := tokenizer.Load(tokenizerPath)
	if err != nil {
		return result, err
	}
	file, err := os.Open(goldenPath)
	if err != nil {
		return result, err
	}
	defer file.Close()

	reported := 0
	report := func(format string, args ...any) {
		if reported < maxReportedMismatches {
			fmt.Fprintf(os.Stderr, format+"\n", args...)
		}
		reported++
	}
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 0, 1<<20), maxLineBytes)
	for scanner.Scan() {
		var record goldenRecord
		if err := json.Unmarshal(scanner.Bytes(), &record); err != nil {
			return result, err
		}
		switch record.Kind {
		case "text":
			if err := checkText(tok, record, &result, report); err != nil {
				return result, err
			}
		case "chat":
			if err := checkChat(tok, record, &result, report); err != nil {
				return result, err
			}
		default:
			return result, fmt.Errorf("unknown kind %q", record.Kind)
		}
	}
	if err := scanner.Err(); err != nil {
		return result, err
	}
	if reported > 0 {
		return result, fmt.Errorf("%d mismatches", reported)
	}
	return result, nil
}

// checkText は、本文としての分割（特殊トークンを認識しない）と復元を基準と比べます。
func checkText(tok *tokenizer.Tokenizer, record goldenRecord, result *summary, report func(string, ...any)) error {
	result.Texts++
	start := time.Now()
	ids, err := tok.EncodeText(record.Text)
	result.EncodeSeconds += time.Since(start).Seconds()
	if err != nil {
		return err
	}
	result.TextTokens += len(ids)
	if !slices.Equal(ids, record.IDs) {
		result.TextIDMismatches++
		report("text ids differ for %q:\n  go: %v\n  hf: %v", shorten(record.Text), shortenIDs(ids), shortenIDs(record.IDs))
	}
	decoded, err := tok.Decode(record.IDs, false)
	if err != nil {
		return err
	}
	if decoded != record.Decoded {
		result.DecodeMismatches++
		report("decode differs:\n  go: %q\n  hf: %q", shorten(decoded), shorten(record.Decoded))
	}
	return nil
}

// checkChat は、テンプレートを当てはめた文字列とトークン列を基準と比べます。
func checkChat(tok *tokenizer.Tokenizer, record goldenRecord, result *summary, report func(string, ...any)) error {
	result.Chats++
	segments, err := server.RenderChatForModel(tokenizerArchitecture(tok), record.Messages, record.Tools, record.EnableThinking)
	if err != nil {
		return err
	}
	var rendered strings.Builder
	for _, segment := range segments {
		rendered.WriteString(segment.Text)
	}
	if rendered.String() != record.Rendered {
		result.RenderMismatches++
		report("rendered differs (thinking=%v):\n  go: %q\n  hf: %q", record.EnableThinking, rendered.String(), record.Rendered)
	}
	ids, err := tok.Encode(segments...)
	if err != nil {
		return err
	}
	if !slices.Equal(ids, record.IDs) {
		result.ChatIDMismatches++
		report("chat ids differ (thinking=%v):\n  go: %v\n  hf: %v", record.EnableThinking, ids, record.IDs)
	}
	return nil
}

// shortenLimit は、不一致の表示で文字列やトークン列を切り詰める長さです。
const shortenLimit = 200

// shorten は長い文字列を先頭 shortenLimit 文字に切り詰めます。例: 1000 文字 → 200 文字 + "…"
func shorten(text string) string {
	runes := []rune(text)
	if len(runes) <= shortenLimit {
		return text
	}
	return string(runes[:shortenLimit]) + "…"
}

// shortenIDs は長いトークン列を先頭 shortenLimit 個に切り詰めます。
func shortenIDs(ids []int) []int {
	if len(ids) <= shortenLimit {
		return ids
	}
	return ids[:shortenLimit]
}

// tokenizerArchitecture は読み込んだトークナイザーに対応するモデル名を返します。
func tokenizerArchitecture(tok *tokenizer.Tokenizer) string {
	if tok.IsGemma() {
		return "gemma4"
	}
	return "lfm3"
}
