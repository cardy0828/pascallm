# API と設定

[English](API.en.md)

`GET /health`、`GET /v1/models`、`POST /v1/chat/completions` に対応します。
既定は `127.0.0.1:18120`。認証・TLS・インターネット公開の仕組みは含みません。

```bash
curl http://127.0.0.1:18120/v1/chat/completions   -H 'Content-Type: application/json'   -d '{"messages":[{"role":"user","content":"こんにちは"}],"temperature":0.7,"seed":42,"max_tokens":128}'
```

temperature、top_k、top_p、min_p、repetition_penalty、presence_penalty、frequency_penalty、stop、streamを扱います。
tools/tool_choice、parallel_tool_calls、response_format（json_object/json_schema）にも対応します。
ツールは実行せず、呼び出しを返すだけです。画像・音声、n>1は対象外です。
JSON Schema は対応する構造・型を制約しますが、数値範囲、文字列長、pattern、format等の完全なJSON Schema準拠は保証しません。

## Gemma4 の追加設定

- `-prefill-chunk 512` は入力の区切り幅（1〜1,024）。`-compact-vram` で一括確保と局所 KV 復元幅の縮小を有効化。[設定別の測定](GEMMA4_PREFILL.md)
- `-mtp /path/to/assistant.gguf`: 対応するQ4_0 assistantでMTPを有効化。
- `-mtp-draft 3`: 固定候補数。1〜7。`-mtp-auto` は自動調整。
- `-mtp-greedy-only`: 比較用。サンプリング・制約付き生成でMTPを使わない。
- `-kv-cache fp16`: 既定。比較用に k-fp32、診断用に fp32。
- `-prefix-cache-mib 1024 -prefix-cache-entries 8`: 任意のホストKV保持。既定は無効。

MTPは決定的なassistant候補と本体サンプラーの実選択を照合します。一般の確率的assistant分布のp/q受理法ではありません。
開発者の内部検査で同じ選択経路・seedの一致を確認しています。公開環境で資産取得から再現する手順は未整備です。GPU候補選択と全logits選択の経路間では浮動小数点処理が異なります。

## 複数会話

毎回、完全なmessagesを渡します。サーバー側で会話履歴を暗黙追加しません。
ホストKV保持は同じモデルインスタンス内でトークン列の一致を調べ、最長の保存点を復元します。
保存点は先頭256位置と生成直前の安定したprompt境界。任意境界のPaged KVではありません。
上限は保持KV payloadとtoken配列の合計で、プロセスRSSの上限ではありません。上限超過はLRU破棄、単独で超過する状態は保存しません。
初回保存に時間がかかるため、再訪の少ない用途では遅くなる場合があります。

生成は1件ずつ。受付済みの最大16件をFIFOで処理します。
満杯時は通常応答503、開始済みSSEではエラーイベントです。生成中の要求の横取りはしません。
`timings.queue_ms` はworker待ち時間、`prompt_ms` は保存・復元を含む入力処理時間です。

[貪欲 MTP の追加設定と用途別の実測](GEMMA4_MTP_TUNING.md)
