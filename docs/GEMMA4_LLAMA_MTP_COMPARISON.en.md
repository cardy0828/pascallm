# Gemma 4 12B decode with llama.cpp and MTP

[日本語・original Japanese prompts](GEMMA4_LLAMA_MTP_COMPARISON.md)

On November 17, 2026, PascaLLM with MTP generated 1.21–1.51 times as many tokens per second as the tested llama.cpp fork with MTP on four 128-token completions. This benchmark predates the [long-prompt optimizations](GEMMA4_PREFILL.en.md).

## Conditions

PascaLLM used sources corresponding to public commit `8c0178b`. The comparison used the existing CUDA build of the [tvall43 llama.cpp fork at e98d01d](https://github.com/tvall43/llama.cpp/commit/e98d01d92d789147d33ac1caf89b08c4416bf504). Both ran on one GTX 1080 Ti under Linux, all layers on GPU, FP16 KV, context limit 8,192, one request at a time. Both used Gemma 4 12B IT QAT from `gemma-4-12B-it-qat-UD-Q4_K_XL.gguf`; the target matrices are Q4_0. PascaLLM repacked the same values as PQ4.

Both MTP engines used `mtp-gemma-4-12B-it.gguf` with up to three candidates. PascaLLM used a fixed candidate count. The llama.cpp build used Flash Attention, batch and ubatch 512, six CPU threads, and `--spec-type draft-mtp`. Its logs showed CUDA Graph reuse.

Thinking was disabled. Sampling used seed 42, top_k 40, top_p 0.95, min_p 0.05, repetition penalty 1, and zero presence and frequency penalties. Each mode processed five prompts in three rounds after warmup (60 requests). Prompt lengths matched and cached tokens were zero. Mode order was fixed: llama.cpp standard, llama.cpp MTP, PascaLLM standard, PascaLLM MTP. GPU clocks were unlocked.

## Results

Median server-reported decode speed over three runs, excluding prompt processing and HTTP. Temperature was zero except for the story (0.7). Parentheses show the observed MTP range.

| Prompt | llama.cpp standard | llama.cpp MTP | PascaLLM standard | PascaLLM MTP |
|---|---:|---:|---:|---:|
| Photosynthesis in Japanese | 34.92 | 58.39 (58.38–58.62) | 45.96 | 88.40 (88.35–88.58) |
| Remove duplicates in Go | 34.93 | 68.50 (68.45–68.62) | 45.89 | 99.46 (99.38–99.76) |
| Explain a change calculation | 34.89 | 84.05 (84.04–84.06) | 45.77 | 122.76 (122.75–123.12) |
| Short library story | 34.79 | 58.16 (58.10–58.20) | 45.75 | 70.35 (70.33–70.35) |

Median timings with MTP, in milliseconds. Total latency was measured by a non-streaming HTTP client on the same host.

| Prompt | Input tokens | llama.cpp prompt | PascaLLM prompt | llama.cpp total | PascaLLM total |
|---|---:|---:|---:|---:|---:|
| Photosynthesis | 24 | 123.00 | 135.74 | 2300.02 | 1573.08 |
| Go | 32 | 142.35 | 171.44 | 1997.69 | 1449.13 |
| Change calculation | 52 | 167.10 | 206.20 | 1679.65 | 1241.25 |
| Story | 29 | 153.83 | 170.24 | 2341.25 | 1976.32 |
| Extract an identifier | 1647 | 3244.74 | 3025.70 | 3290.25 | 3057.82 |

All runs correctly extracted the identifier; llama.cpp generated five tokens and PascaLLM four. The other prompts reached the 128-token cap. The test did not assess complete code correctness, the final arithmetic answer, or overall writing quality. PascaLLM gave identical text with MTP on and off for all five prompts. llama.cpp gave matching text for the calculation and identifier, and different text for the other three. The reason for the differences at temperature zero is unresolved. Each mode repeated its own text across runs. Different output sequences can affect draft acceptance and throughput. The original Japanese prompts and identifier construction are in the linked Japanese edition.
