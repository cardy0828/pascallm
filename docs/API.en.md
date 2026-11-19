# Chat API and runtime settings

[日本語](API.md)

The server listens on loopback (default `127.0.0.1:18120`). Use `GET /health`, `GET /v1/models`, or `POST /v1/chat/completions`. Authentication and TLS are outside the server.

```bash
curl http://127.0.0.1:18120/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"temperature":0.7,"seed":42,"max_tokens":128}'
```

Requests support temperature, top_k, top_p, min_p, repetition/presence/frequency penalties, stop sequences, and SSE streaming. Tool choice, parallel tool call output, JSON objects, and a subset of JSON Schema are supported. The caller executes tools. Numeric ranges, string lengths, patterns, and formats are among the limits of schema enforcement. Use one completion per request.

| Gemma option | Effect |
|---|---|
| `-mtp FILE` | Load a matching Q4_0 assistant. |
| `-mtp-draft 3` | Propose up to three tokens. Allowed range 1–7. Use the server's `-mtp-auto` for automatic adjustment. |
| `-mtp-greedy-only` | Restrict MTP to greedy decoding. |
| `-kv-cache fp16` | Default KV type. Alternatives are k-fp32 and diagnostic fp32. |
| `-prefill-chunk 512` | Prompt tokens per chunk, range 1–1,024. |
| `-compact-vram` | Pack allocations and size local rollback space for the selected chunk. |
| `-prefix-cache-mib 1024 -prefix-cache-entries 8` | Enable host prefix snapshots with size and entry limits. Disabled by default. |

MTP checks deterministic draft candidates against the target sampler's actual selections. Random state, grammar, and penalties are request-local. Floating-point details can differ between GPU candidate selection and full-logits sampling.

Send the complete messages on every request. When host caching is enabled with a positive prefix-cache budget, the engine can restore the longest matching token prefix within that model instance. It saves a boundary at 256 tokens and a stable boundary before generation when eligible. Least recently used snapshots are evicted. The byte budget covers saved KV payload and tokens. Saving a new snapshot costs time.

One GPU worker processes accepted requests FIFO, with a queue of up to 16. A full queue returns HTTP 503 or an SSE error if streaming has begun. `timings.queue_ms` measures time waiting for the worker. `prompt_ms` includes cache save and restore.

[Greedy MTP options and workload benchmarks](GEMMA4_MTP_TUNING.en.md)
