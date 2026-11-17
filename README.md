# PascaLLM

Pascal（sm_61）向けの LLM 推論エンジンと、量子化形式 PQ4 の開発プロジェクトです。
Go と CUDA Driver API を使い、GTX 1080 Ti で LFM3 1.7B と Gemma 4 12B を検証しています。

**実験的なソース公開です。対応環境・モデルと制約を確認して利用してください。**
対応していないモデルを自動変換して実行する汎用エンジンではありません。

## 主な機能

- PQ4/PQ8、DP4A による行列積、CUDA Graph、まとめたプロンプト処理。
- LFM3 1.7B の safetensors 変換と、Gemma 4 の対応 Q4_0 GGUF から PQ4 への詰め替え。
- OpenAI 互換のチャット API、SSE、temperature、seed、ペナルティ、ツール呼び出し、JSON Schema。
- Gemma 4 の MTP（貪欲・サンプリング・制約付き生成）、任意の K FP32 / V FP16。
- 任意のホスト KV キャッシュによる会話切り替えと prefix 再利用、上限16件の FIFO。

Paged KV、continuous batching、画像・音声入力、複数モデル同時実行は未対応です。
1プロセスにモデル1個、GPUでの生成は1件ずつです。APIはループバックだけで待ち受けます。

## 動作環境と導入

以下は [導入手順](docs/GETTING_STARTED.md) に従って重みの変換とtokenizer準備を済ませた後の起動例です。対応資産の取得元・リビジョン一覧は今後の整備項目です。

Linux amd64、NVIDIA Driver、Go 1.27.1 以上を使用します。Windows ネイティブ実行は未対応です。
確認済みGPUは GTX 1080 Ti。埋め込み cubin は sm_61 向けです。
CUDA Toolkit はカーネルの再生成時だけ必要です。保存済み cubin のビルド記録は `internal/kernels/nvcc-version.txt` にあります。

```bash
git clone https://github.com/cardy0828/pascallm.git
cd pascallm
go build -o bin/pascallm-server ./cmd/pascallm-server
nvidia-smi -L
export PASCA_DEVICE_UUID='GPU-対象GPUのUUID'
./bin/pascallm-server   -architecture gemma4   -pq4 ./data/gemma4-12b-qat.pq4m   -tokenizer-dir ./data/m7-api/tokenizer   -addr 127.0.0.1:18120 -max-context 8192
```

重み・tokenizer は同梱しません。モデルの取得・利用条件を確認し、対応する資産を用意してください。
具体的な変換・tokenizer準備は [導入手順](docs/GETTING_STARTED.md) を参照してください。

## ドキュメント

- [導入・変換・検証](docs/GETTING_STARTED.md)
- [APIと実用機能](docs/API.md)
- [実装構成](docs/ARCHITECTURE.md)
- [測定結果と制約](docs/BENCHMARKS.md)
- [Gemma 4 12B: llama.cppとのMTP比較](docs/GEMMA4_LLAMA_MTP_COMPARISON.md)
- [依存物とモデル資産](THIRD_PARTY.md)

開発マイルストーンの生ログ・機器固有の運用手順は、公開リポジトリから除いています。
測定ごとの対象バージョン・条件・制約は、各比較資料に記載しています。

## ライセンス

本体の独自コード・文書は [MIT License](LICENSE) で提供します。
purego自体はApache-2.0、モデル重み・tokenizer・assistant・評価データ・外部ツールはそれぞれ別のライセンス／利用条件に従います。
量子化・変換後の重みも本体のMITに変更されません。詳細は [第三者の資産と適用範囲](THIRD_PARTY.md) を参照してください。
