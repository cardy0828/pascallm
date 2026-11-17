# 導入・変換・検証

## 配置

すべてリポジトリのルートから実行します。実行例は Linux の bash です。
`data/` は変換済み重み・評価資産、`models/` は取得した原本、`external/` は比較用ツールの任意配置先です。
これらは Git の対象外です。CLIの `-pq4` 等で別の絶対パスを指定できます。
GPUは `-device-uuid`、または環境変数 `PASCA_DEVICE_UUID` で必ず指定します。空ならエラーになり、自動選択しません。

## ビルド

```bash
go version
CGO_ENABLED=0 go build -o bin/pascallm-server ./cmd/pascallm-server
CGO_ENABLED=0 go build -o bin/gguf-to-pq4 ./cmd/gguf-to-pq4
CGO_ENABLED=0 go build -o bin/pq4-convert ./cmd/pq4-convert
CGO_ENABLED=0 go build -o bin/m1a-probe ./cmd/m1a-probe
export PASCA_DEVICE_UUID='GPU-対象GPUのUUID'
./bin/m1a-probe
```

通常のGoビルドは保存済みcubinを埋め込みます。カーネル変更時は、sm_61を扱えるnvccを指定して再生成します。

```bash
sha256sum -c internal/kernels/cubin.sha256
NVCC=/path/to/cuda/bin/nvcc bash kernels/build.sh
git diff -- internal/kernels/
```

再生成前のSHA検査は保存済み資産との照合です。build.shはハッシュ記録も更新するため、再生成後はcubinと記録のGit差分をレビューします。コンパイラ版が異なるとバイナリが変わる場合があります。

## Gemma 4

変換入力は、実装が対応する Gemma 4 構成で、対象行列が Q4_0 の GGUF です。
名前にQ4と付く任意の混合量子化GGUFを受け付ける意味ではありません。変換時にテンソル形式を検査します。

```bash
mkdir -p data
./bin/gguf-to-pq4 -gguf /path/to/compatible-gemma4.gguf -out ./data/gemma4-12b-qat.pq4m
python3 tools/reference/m7_prepare_tokenizer.py   --gguf /path/to/compatible-gemma4.gguf   --tokenizer /path/to/matching/tokenizer.json   --gguf-python /path/to/llama.cpp/gguf-py   --out ./data/m7-api/tokenizer
```

tokenizer準備はGGUFと全語彙・全mergeの一致を検査します。任意のGemma tokenizerの流用はできません。
Pythonの準備・参照スクリプト群は、用途に応じて numpy、torch、transformers、safetensors、tokenizers、llama.cppのgguf-pyを使います。すべてがtokenizer準備だけに必要という意味ではありません。依存バージョンの固定と、クリーン環境でのモデル取得からの再現は今後の整備項目です。
API実行時にはPythonを使いません。

## MTP assistant（任意）

MTPを使わない場合、assistantは不要です。使う場合は、本体のGemma4構成に対応するQ4_0 assistant GGUFを別途用意し、-mtp /path/to/assistant.gguf を指定します。
assistantにも提供元の独立した利用条件が適用され、本体のMITには含まれません。
本体GGUF・tokenizer・assistantの取得元、対応リビジョンとSHA-256を固定した一覧は未整備です。現在の手順だけで資産取得からの完全再現を保証しません。

## LFM3

対応する LFM3 1.7B の safetensors と設定を `models/lfm3-1.7b-training` に置くか、任意のパスを指定します。

```bash
./bin/pq4-convert -model-dir /path/to/lfm3-model -out ./data/lfm3.pq4m
./bin/pascallm-server -architecture lfm3 -pq4 ./data/lfm3.pq4m -tokenizer-dir /path/to/lfm3-model
```

imatrixや混合PQ8は追加設定です。まず変換CLIの `-h` を確認してください。

## 検査

```bash
CGO_ENABLED=0 go test ./...
CGO_ENABLED=0 go vet ./...
CGO_ENABLED=1 go test -race ./internal/server ./internal/generate
```

race検査は任意で、cgoと対応するCコンパイラ（例: gcc）が必要です。通常ビルドのCGO_ENABLED=0とは別の検査です。race検査は通常のビルド検査とは別に実施してください。

GPU検査は既定でskipします。対応する実機・モデル・KLD評価資産を用意した場合だけ有効化します。
`PASCA_DATA_DIR`、`PASCA_MODELS_DIR` はGemma4のMTP／snapshotテスト用資産のルートを変更できます。

```bash
export PASCA_DATA_DIR=/absolute/path/to/data
export PASCA_MODELS_DIR=/absolute/path/to/models
PASCA_MTP_GPU_TEST=1 go test ./internal/gemma4gpu -run TestMTPSelectionGPU -count=1 -v
PASCA_PREFIX_GPU_TEST=1 go test ./internal/gemma4gpu -run TestPrefixSnapshotGPU -count=1 -v
```

`tools/m3-api-check.py`、`m6-api-check.py`、`m7-api-check.py` は起動済みAPIの回帰検査です。
M8〜M12の測定スクリプトは旧実験の再実行補助で、`data/m*/` の外部資産配置を前提にします。
公開版の一般的なベンチマークCLIへの統合は未実施です。
