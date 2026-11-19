# PascaLLM

[日本語](README.md)

PascaLLM runs LFM3 1.7B and Gemma 4 12B on NVIDIA Pascal sm_61 GPUs using Go, the CUDA Driver API, and DP4A integer matrix multiplication. Development and performance testing use the GTX 1080 Ti. The engine supports specific model layouts and includes tools to convert their weights to PQ4.

## Run Gemma 4

Use Linux amd64, an NVIDIA driver, and Go 1.27.1 or later. The build embeds sm_61 cubins. You only need the CUDA Toolkit to rebuild the kernels. Prepare the weights and matching tokenizer using the setup guide, then start the server.

```bash
git clone https://github.com/cardy0828/pascallm.git
cd pascallm
CGO_ENABLED=0 go build -o bin/pascallm-server ./cmd/pascallm-server
nvidia-smi -L
export PASCA_DEVICE_UUID='GPU-your-device-uuid'
./bin/pascallm-server -architecture gemma4 \
  -pq4 ./data/gemma4-12b-qat.pq4m -tokenizer-dir ./data/m7-api/tokenizer \
  -addr 127.0.0.1:18120 -max-context 8192
```

The server runs one model per process and generates one response at a time. It supports streaming chat, sampling, tool calls, JSON Schema constraints, optional Gemma MTP, and host-side prefix caching. It accepts text input. Paged KV and continuous batching are outside the current implementation.

## Documentation

- [Greedy MTP options and workload benchmarks](docs/GEMMA4_MTP_TUNING.en.md)
- [Build, convert weights, and validate the installation](docs/GETTING_STARTED.en.md)
- [API, generation settings, and conversation reuse](docs/API.en.md)
- [Engine architecture](docs/ARCHITECTURE.en.md)
- [Benchmarks and limitations](docs/BENCHMARKS.en.md)
- [Gemma 4 prefill and maximum context](docs/GEMMA4_PREFILL.en.md)
- [Gemma 4 decode with llama.cpp and MTP](docs/GEMMA4_LLAMA_MTP_COMPARISON.en.md)

Original engine code and documentation are MIT licensed. The purego dependency uses Apache-2.0. Model weights, tokenizers, MTP assistants, datasets, and external tools retain their respective terms, including after conversion. [MIT license](LICENSE) · [Third-party notices](THIRD_PARTY.en.md)
