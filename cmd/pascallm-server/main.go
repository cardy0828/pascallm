// pascallm-server は、LFM3 または Gemma 4（PQ4、GTX 1080 Ti）を OpenAI 互換の API で提供します（M3）。
// ループバックのアドレス（127.0.0.1・::1）だけで待ち受け、それ以外を指定したら起動しません。
//
// 実行例（Linux）:
//
//	CGO_ENABLED=0 go build -o ./data/bin/pascallm-server ./cmd/pascallm-server
//	./data/bin/pascallm-server \
//	    -tokenizer-dir ./models/lfm3-1.7b-training
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"syscall"
	"time"

	"github.com/cardy0828/pascallm/internal/chat"
	"github.com/cardy0828/pascallm/internal/gemma4gpu"
	"github.com/cardy0828/pascallm/internal/generate"
	"github.com/cardy0828/pascallm/internal/lfm3gpu"
	"github.com/cardy0828/pascallm/internal/server"
	"github.com/cardy0828/pascallm/internal/tokenizer"
)

// defaultDeviceUUID は空なら PASCA_DEVICE_UUID を使用します。GPUは明示的に選択します。
const defaultDeviceUUID = ""

// defaultModelPath は既定のモデルです。M5 の search + imatrix で変換したもの
// （KL 0.099、mse 版の 0.129 より小さく、生成速度は同じ。開発時の内部測定）。
const defaultModelPath = "./data/lfm3-1.7b-pq4-search-imatrix.pq4m"

// サーバーの時間の設定です。
const (
	readHeaderTimeout = 10 * time.Second // 要求のヘッダーを読み終えるまでの上限
	shutdownTimeout   = 30 * time.Second // 終了時に処理中の応答を待つ上限
)

// endOfTextToken は、<|im_end|> のほかに生成を止めるトークンです。
const endOfTextToken = "<|endoftext|>"

// maxPrefixCacheMiB は起動指定の上限（1 TiB）です。単位変換のoverflowも防ぎます。
const maxPrefixCacheMiB int64 = 1024 * 1024

// options はコマンドの引数です。
type options struct {
	mtpLookup          int
	mtpIntegrated      bool
	compactVRAM        bool
	prefillChunk       int
	prefixCacheMiB     int64
	prefixCacheEntries int
	mtpGreedyOnly      bool
	kvCache            string
	mtpDraft           int
	mtpAdaptive        bool
	mtpPath            string
	architecture       string
	thinkingSet        bool // 利用者が enable-thinking を明示したか。
	pq4Path            string
	tokenizerDir       string
	addr               string
	modelName          string
	deviceUUID         string
	maxContext         int
	enableThinking     bool
	gpuCandidates      bool
	batchPrefill       bool
}

// main は引数を読み、サーバーを動かします。失敗したら理由を表示して終了コード 1 で終わります。
func main() {
	var opts options
	flag.Int64Var(&opts.prefixCacheMiB, "prefix-cache-mib", 0, "Gemma4 ホストKVキャッシュ上限MiB（0は無効）")
	flag.IntVar(&opts.prefixCacheEntries, "prefix-cache-entries", 8, "ホストKVキャッシュの最大保存点数")
	flag.StringVar(&opts.architecture, "architecture", "lfm3", "モデルの種類（lfm3 または gemma4）")
	flag.StringVar(&opts.pq4Path, "pq4", "", "PQ4 のモデルファイル（.pq4m）")
	flag.StringVar(&opts.tokenizerDir, "tokenizer-dir", "", "tokenizer.json と generation_config.json があるディレクトリ")
	flag.StringVar(&opts.addr, "addr", "127.0.0.1:18120", "待ち受けるアドレス（ループバックだけ）")
	flag.StringVar(&opts.modelName, "model-name", "", "API に出すモデルの名前")
	flag.StringVar(&opts.deviceUUID, "device-uuid", defaultDeviceUUID, "使う GPU の UUID（省略時 PASCA_DEVICE_UUID）")
	flag.IntVar(&opts.mtpLookup, "mtp-lookup", 0, "貪欲生成の要求履歴から提案する最大候補数（0・3・7）")
	flag.BoolVar(&opts.mtpIntegrated, "mtp-integrated", false, "固定3候補・FP16 KV の貪欲 MTP を1本の CUDA Graph で実行する")
	flag.BoolVar(&opts.compactVRAM, "compact-vram", false, "Gemma4 専用。重みと KV を一括確保し、保存位置の復元幅を入力チャンクに合わせる")
	flag.IntVar(&opts.prefillChunk, "prefill-chunk", 0, "Gemma4 の入力を区切るトークン数（1〜1024、0 は既定）")
	flag.IntVar(&opts.maxContext, "max-context", 4096, "プロンプトと生成を合わせた最大のトークン数（KV キャッシュの大きさ）")
	flag.BoolVar(&opts.enableThinking, "enable-thinking", true, "chat_template_kwargs.enable_thinking の既定値")
	flag.BoolVar(&opts.gpuCandidates, "gpu-candidates", true, "選び方の候補を GPU で絞る（false なら毎回 logits 全体を CPU に移す。比較用）")
	flag.BoolVar(&opts.batchPrefill, "batch-prefill", true, "プロンプトをまとめて処理する（false なら 1 トークンずつ。比較用）")
	flag.StringVar(&opts.kvCache, "kv-cache", "fp16", "Gemma4 KV: fp16、k-fp32、fp32（診断用）")
	flag.BoolVar(&opts.mtpGreedyOnly, "mtp-greedy-only", false, "MTP を従来の貪欲生成だけに限定（比較用）")
	flag.StringVar(&opts.mtpPath, "mtp", "", "Gemma4 12B assistant Q4_0 GGUF（MTP 候補検証、空なら無効）")
	flag.IntVar(&opts.mtpDraft, "mtp-draft", 3, "MTP 候補数 1..7（自動調整時は上限）")
	flag.BoolVar(&opts.mtpAdaptive, "mtp-auto", false, "MTP の候補数を実測時間と受理数から調整")
	flag.Parse()
	flag.Visit(func(f *flag.Flag) {
		if f.Name == "enable-thinking" {
			opts.thinkingSet = true
		}
	})
	if err := run(opts); err != nil {
		fmt.Fprintln(os.Stderr, "pascallm-server:", err)
		os.Exit(1)
	}
}

// run はトークナイザーとモデルを読み込み、HTTP サーバーを起動し、SIGINT・SIGTERM まで生成を処理します。
// CUDA のコンテキストのため、このゴルーチンを OS スレッドに固定し、生成はこのゴルーチンで行います。
func run(opts options) error {
	if err := resolveOptions(&opts); err != nil {
		return err
	}
	runtime.LockOSThread()
	if opts.pq4Path == "" || opts.tokenizerDir == "" {
		return errors.New("-pq4 and -tokenizer-dir must not be empty")
	}
	if err := checkLoopback(opts.addr); err != nil {
		return err
	}

	tok, err := tokenizer.Load(filepath.Join(opts.tokenizerDir, "tokenizer.json"))
	if err != nil {
		return err
	}
	if tok.IsGemma() != (opts.architecture == "gemma4") {
		return errors.New("tokenizer does not match -architecture")
	}
	defaults, err := server.LoadGenerationConfig(filepath.Join(opts.tokenizerDir, "generation_config.json"), server.BaseDefaults)
	if err != nil {
		return err
	}
	defaults.EnableThinking = opts.enableThinking
	stopNames := []string{chat.MessageEnd, endOfTextToken}
	if opts.architecture == "gemma4" {
		stopNames = []string{"<eos>", chat.GemmaTurnEnd, chat.GemmaResponseStart}
	}
	stopTokens, err := lookupTokens(tok, stopNames...)
	if err != nil {
		return err
	}

	var model generate.Model
	if opts.architecture == "gemma4" {
		model, err = gemma4gpu.Open(opts.pq4Path, gemma4gpu.Options{
			MTPLookup:      opts.mtpLookup,
			MTPIntegrated:  opts.mtpIntegrated,
			CompactVRAM:    opts.compactVRAM,
			PrefillChunk:   opts.prefillChunk,
			MTPGreedyOnly:  opts.mtpGreedyOnly,
			KVCache:        opts.kvCache,
			DeviceUUID:     opts.deviceUUID,
			MaxContext:     opts.maxContext,
			MTPPath:        opts.mtpPath,
			MTPDraftTokens: opts.mtpDraft,
			MTPAdaptive:    opts.mtpAdaptive,
		})
	} else {
		model, err = lfm3gpu.Open(opts.pq4Path, lfm3gpu.Options{DeviceUUID: opts.deviceUUID, MaxContext: opts.maxContext})
	}
	if err != nil {
		return err
	}
	defer func() {
		if err := model.Close(); err != nil {
			log.Printf("close model: %v", err)
		}
	}()
	if opts.mtpPath != "" {
		log.Printf("Gemma4 MTP enabled: target-selected verification; greedy-only=%t", opts.mtpGreedyOnly)
	}
	engine, err := generate.New(model, tok.Size(), stopTokens)
	if err != nil {
		return err
	}
	if opts.prefixCacheMiB < 0 || opts.prefixCacheMiB > maxPrefixCacheMiB {
		return fmt.Errorf("prefix-cache-mib must be between 0 and 1048576")
	}
	if err := engine.SetPrefixCache(opts.prefixCacheMiB*1024*1024, opts.prefixCacheEntries); err != nil {
		return err
	}
	engine.SetGPUCandidates(opts.gpuCandidates)
	engine.SetBatchPrefill(opts.batchPrefill)

	api := server.New(server.Config{Architecture: opts.architecture, ModelName: opts.modelName, Tokenizer: tok, Defaults: defaults, MaxContext: opts.maxContext, StopTokens: stopTokens})
	listener, err := net.Listen("tcp", opts.addr)
	if err != nil {
		return err
	}
	httpServer := &http.Server{Handler: api.Handler(), ReadHeaderTimeout: readHeaderTimeout}
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	serveErr := make(chan error, 1)
	go func() {
		serveErr <- httpServer.Serve(listener)
		stop()
	}()
	log.Printf("listening on http://%s/v1 (model %s, max context %d, defaults %+v, gpu candidates %v, batch prefill %v, GPU memory %.2f GB)",
		listener.Addr(), opts.modelName, opts.maxContext, defaults, opts.gpuCandidates, opts.batchPrefill,
		float64(model.AllocatedBytes())/1e9)

	api.RunWorker(ctx, engine)

	shutdownCtx, cancel := context.WithTimeout(context.Background(), shutdownTimeout)
	defer cancel()
	if err := httpServer.Shutdown(shutdownCtx); err != nil {
		return err
	}
	if err := <-serveErr; err != nil && !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}

// checkLoopback は、待ち受けるアドレスがループバック（127.0.0.0/8 または ::1）かを確かめます。
// 例: "127.0.0.1:18120" → nil、"0.0.0.0:18120" → エラー、"localhost:18120" → エラー（名前は使わない）
func checkLoopback(addr string) error {
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		return fmt.Errorf("-addr %q: %w", addr, err)
	}
	ip := net.ParseIP(host)
	if ip == nil || !ip.IsLoopback() {
		return fmt.Errorf("-addr %q must be a loopback IP address such as 127.0.0.1", addr)
	}
	return nil
}

// lookupTokens は、文字列のトークン番号をまとめて引きます。例: lookupTokens(tok, "<|im_end|>") → [124900]
func lookupTokens(tok *tokenizer.Tokenizer, contents ...string) ([]int, error) {
	ids := make([]int, 0, len(contents))
	for _, content := range contents {
		id, ok := tok.TokenID(content)
		if !ok {
			return nil, fmt.Errorf("tokenizer has no %q token", content)
		}
		ids = append(ids, id)
	}
	return ids, nil
}

// resolveOptions はモデルごとの既定値を補います。明示したパス・名前・思考設定は保持します。
// 例: architecture=gemma4、pq4未指定 → Gemma 12B の成果物を選ぶ。
func resolveOptions(opts *options) error {
	if opts.architecture == "" {
		opts.architecture = "lfm3"
	}
	if opts.architecture != "gemma4" && (opts.mtpLookup != 0 || opts.mtpIntegrated) {
		return fmt.Errorf("MTP options require gemma4")
	}
	if opts.architecture != "gemma4" && (opts.prefillChunk != 0 || opts.compactVRAM) {
		return fmt.Errorf("-prefill-chunk and -compact-vram require -architecture gemma4")
	}
	if opts.mtpPath != "" && opts.architecture != "gemma4" {
		return fmt.Errorf("-mtp requires -architecture gemma4")
	}
	if opts.architecture != "lfm3" && opts.architecture != "gemma4" {
		return fmt.Errorf("unsupported -architecture %q", opts.architecture)
	}
	if opts.pq4Path == "" {
		opts.pq4Path = defaultModelPath
		if opts.architecture == "gemma4" {
			opts.pq4Path = "./data/gemma4-12b-qat.pq4m"
		}
	}
	if opts.modelName == "" {
		opts.modelName = "lfm3-1.7b-pq4"
		if opts.architecture == "gemma4" {
			opts.modelName = "gemma4-12b-qat-pq4"
		}
	}
	if !opts.thinkingSet {
		opts.enableThinking = opts.architecture == "lfm3"
	}
	return nil
}
