# Engine architecture

[日本語](ARCHITECTURE.md)

| Path | Responsibility |
|---|---|
| cmd/pascallm-server | Server startup and configuration |
| cmd/gguf-to-pq4, cmd/pq4-convert | Weight conversion |
| internal/cuda | CUDA Driver API bindings through purego |
| kernels, internal/kernels | CUDA sources and embedded sm_61 cubins |
| internal/pq4, internal/gguf | Quantized formats and model files |
| internal/lfm3gpu, internal/gemma4gpu | Model-specific GPU execution |
| internal/generate | Sampling, generation, and prefix management |
| internal/chat, internal/tokenizer, internal/grammar | Templates, tokenization, and constrained output |
| internal/server | HTTP, SSE, queue, and responses |
| tools/reference | Model preparation and evaluation helpers |

The GPU worker owns the CUDA context on a dedicated OS thread. PQ4 stores coefficients separately from packed four-bit values and uses DP4A with quantized activations. Gemma stores local attention in a KV ring and retains global KV through the configured context limit. Unconsumed draft tokens do not advance the logical KV position. Restoring a committed prompt snapshot invalidates pending candidates and logits. [Prefill kernels and memory layout](GEMMA4_PREFILL.en.md)
