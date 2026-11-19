# Training-free MTP decode options for Gemma 4 12B

[日本語](GEMMA4_MTP_TUNING.md)

When copying or editing a passage, the correct continuation often appears earlier in the prompt. The server option `-mtp-lookup 3` proposes up to three previously seen tokens and verifies them with the target. If the current suffix has no match of at least four tokens in the last 4,096 tokens, the assistant proposes its usual three candidates. A limit of seven permits longer proposals only for matches of eight or more tokens. Each request starts with its own confirmed prompt history, including after cancellation or cache reuse.

For greedy decoding, `-mtp-integrated` captures the three assistant proposals, four-row target verification and accepted-prefix decision in a single CUDA Graph. It requires FP16 KV and is available on both the server and gemma4-gpu. When both options are enabled, the graph handles eligible batches without a history proposal. Near the context limit, shorter batches use the existing path.

The following example targets copying tasks. For rows where only the amounts or times change, an incorrect continuation can trigger extra target passes. Three-token lookup was about 25–27% slower on those cases.

Load a compatible assistant and use a fixed draft count of three. Both options are off by default. Sampling and grammar-driven token selection use the existing verification path. The server's generation_metrics logs count integrated_batches and lookup_batches when used. These batches contribute to batch_wall_ms but have no separate draft_gpu_ms or verify_gpu_ms measurement.

```bash
./bin/pascallm-server -architecture gemma4 -pq4 data/gemma4-12b-qat.pq4m \
  -tokenizer-dir data/m7-api/tokenizer -mtp /path/to/assistant.gguf \
  -kv-cache fp16 -mtp-draft 3 -mtp-integrated -mtp-lookup 3
```

## Development measurements

One GTX 1080 Ti, Gemma 4 12B IT QAT repacked as PQ4, four-layer assistant, FP16 KV, greedy decoding. Throughput is generated tokens per second, excluding startup and prefill. The history benchmark warmed each configuration, varied execution order and generated up to 128 tokens. The first two rows are medians of three runs and the last two are medians of four. Every generated ID and output length matched target-only decoding.

| Workload | Existing MTP | Lookup up to 3 | Lookup up to 7 |
|---|---:|---:|---:|
| Copy a notice | 134.36 | 155.45 | 164.37 |
| Change one color in a notice | 138.78 | 152.47 | 145.90 |
| CSV with different times per row | 142.95 | 107.45 | Unmeasured |
| Invoices with changing amounts and days | 139.04 | 102.06 | Unmeasured |

In a separate CUDA Graph experiment, the baseline ran at 124.67 / 99.01 / 110.26 tok/s for KLD prompts of 24 / 1,024 / 4,092 tokens. The integrated graph reached 124.97 / 99.32 / 110.60 tok/s (four paired runs per length, 128 generated tokens). The observed difference of about 0.2–0.3% based on the rounded medians includes GPU acceptance, consolidated readback and omitted stage timing. These runs did not measure a combined speedup for both options.

A four-row branching prototype yielded speed ratios (total serial time divided by total branching time) of 1.015× across six prompts and 1.007× on four additional prompts. Always branching yielded 0.888× on the first set. Reducing the assistant's output vocabulary improved one CSV case but slowed Japanese explanation and Go code generation. The public runtime does not expose those prototypes.
