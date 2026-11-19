# 実装構成

[English](ARCHITECTURE.en.md)

| 場所 | 役割 |
| --- | --- |
| cmd/pascallm-server | APIサーバー起動・設定 |
| cmd/gguf-to-pq4、cmd/pq4-convert | 重み変換 |
| internal/cuda | purego経由のCUDA Driver API |
| kernels、internal/kernels | CUDAソースと埋め込みsm_61 cubin |
| internal/pq4、internal/gguf | 量子化形式・モデルファイル |
| internal/lfm3gpu、internal/gemma4gpu | モデル別GPU実行 |
| internal/generate | サンプラー、生成ループ、prefix管理 |
| internal/chat、internal/tokenizer、internal/grammar | テンプレート、トークン化、制約 |
| internal/server | HTTP/SSE、FIFO、応答 |
| tools/reference | 評価・モデル資産の準備 |

モデルとCUDA contextは専用のOS threadで扱います。HTTPハンドラーはworkerへ仕事を渡します。
PQ4は係数と4bitデータを分離して配置し、量子化活性値とのDP4A行列積を用います。
Gemma4では局所attentionを輪状KVで保持し、全attentionは文脈上限まで保持します。
MTPの未消費候補は論理KV位置に含めません。prefix snapshotは確定したprompt境界でのみ取得し、復元時に候補とlogitsを無効化します。
文法・乱数・ペナルティは要求ごとに再構築します。
