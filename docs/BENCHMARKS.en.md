# Benchmarks

[日本語](BENCHMARKS.md)

For prompt processing, including chunk size and maximum context, see the [Gemma 4 prefill measurements](GEMMA4_PREFILL.en.md). For decode on the same GPU, see the [llama.cpp MTP comparison](GEMMA4_LLAMA_MTP_COMPARISON.en.md).

These development measurements used Linux, a GTX 1080 Ti, Gemma 4 12B PQ4, FP16 KV, and three draft candidates. The table measures decode only, up to 128 output tokens. Some settings were measured once.

| Workload | Standard (tok/s) | MTP (tok/s) |
|---|---:|---:|
| Text, temperature 0.7 and penalties | 45.86 | 65.60 |
| Required tool, temperature 0.7 | 29.79 | 51.69 |
| JSON Schema, temperature 0.7 | 27.87 | 45.61 |
| Full vocabulary, temperature 1.5 | 16.18 | 15.95 |

FP32 keys improved a long-context diagnostic KL metric but reduced speed; evaluation did not establish higher answer accuracy. FP16 remains the default.

| Host prefix cache workload | Disabled (ms) | Enabled (ms) |
|---|---:|---:|
| First visit to A (567 tokens) | 980.25 | 1161.78 |
| Return to A after B | 977.49 | 121.88 |
| New question with the same system message | 980.37 | 757.52 |

The cache used 1,024 MiB and eight snapshot slots. Prompt time includes saving and restoring KV. The first visit pays a save cost; revisiting a cached prompt can substantially reduce latency.

[Greedy MTP options and workload benchmarks](GEMMA4_MTP_TUNING.en.md)
