# Gemma 4 12B prompt processing

[日本語](GEMMA4_PREFILL.md)

Processing a 65,536-token prompt took 307.39 seconds on one GTX 1080 Ti. The measurements below were collected on November 19, 2026 in development using the kernels shipped here. The model was Gemma 4 12B IT QAT, with Q4_0 weights repacked as PQ4, FP16 KV, and all layers on the GPU. The model stayed loaded, MTP was disabled, and each prompt started with an empty cache.

## What improved

1. **Matrix multiplication and the last layer.** Eligible matrix products use 128 output rows by 64 prompt tokens per tile. The last prefill layer computes only the keys and values needed for subsequent generation. For a 4,096-token prompt, prefill fell from 8.416 to 7.989 seconds (four alternating runs per variant, medians).
2. **Global and local attention.** Eight query heads share K/V in global attention. Local attention uses a bit mask when its ring capacity is a power of two. On 8,191 tokens, time to the first output ID fell from 18.254 to 17.043 seconds against the preceding version (two runs per variant, medians, excluding tokenization and HTTP).
3. **Long-context global attention.** Aligned shared-memory writes for keys and unrolling value accumulation over eight history positions reduced the time shown below. The baseline already includes changes 1 and 2. Each stage was measured separately.

| Prompt tokens | Before (s) | After (s) | Before (tok/s) | After (tok/s) | llama.cpp (tok/s) |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 7.566 | 7.526 | 541.23 | 544.15 | 564.52 |
| 16,384 | 40.458 | 40.097 | 404.94 | 408.58 | 408.49 |
| 65,536 | 312.461 | 307.389 | 209.74 | 213.20 | 200.55 |

PascaLLM used 512-token chunks. It feeds the final prompt token separately, so throughput counts prompt length minus one. The 16K values are medians of two runs per variant with the order reversed. Each 4K and 64K value is from one run, so run-to-run variability at these lengths is unknown. For each input length, the full-vocabulary logits hash at the prompt boundary and the next greedy token matched across both versions.

The llama.cpp reference used the same GPU and source GGUF with Flash Attention. We ran llama-bench once per length with `-ngl 99 -fa 1 -b 512 -ub 512 -p 4096,16384,65536 -n 0 -r 1 -t 6`. The benchmark uses a different prompt and includes the last input token in its timing. [llama.cpp comparison build](https://github.com/tvall43/llama.cpp/commit/e98d01d92d789147d33ac1caf89b08c4416bf504)

## Choosing a chunk size and context limit

The API server and gemma4-gpu accept `-prefill-chunk` from 1 to 1,024 (default 512). For 4,096 tokens, increasing the chunk from 512 to 1,024 reduced median time to first ID from 7.580 to 7.502 seconds over three runs per size, and increased observed peak GPU usage by 134 MiB.

On the 11 GiB card, `-compact-vram -prefill-chunk 256 -max-context 262144` packed the weights, KV cache, and work buffers into one allocation. The local KV rollback allowance follows the chunk size, with a minimum of eight positions. Host prefix snapshots are managed separately.

Before the final long-context optimization, a full-model run used 262,128 prompt tokens and 16 generation positions to reach position 262,144. Prefill took 3,831.07 seconds and observed peak GPU memory was 11,105 MiB. The MTP assistant did not fit at that limit. With the final kernels, the longest full-model benchmark used 65,536 tokens. A separate synthetic attention test covered the last 128 positions at the model limit and produced matching outputs in 463.34 vs. 447.72 ms (two runs per variant, medians).
