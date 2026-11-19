# Build, conversion, and validation

[日本語](GETTING_STARTED.md)

Run the commands from the checkout root on Linux amd64. Store original models in `models/` and prepared assets in `data/`, or pass explicit paths. Use `nvidia-smi -L` to identify the intended GPU, then set `PASCA_DEVICE_UUID` or pass `-device-uuid`.

```bash
mkdir -p data bin
export PASCA_DEVICE_UUID='GPU-your-device-uuid'
CGO_ENABLED=0 go build -o bin/ ./cmd/...
./bin/m1a-probe
sha256sum -c internal/kernels/cubin.sha256
```

A standard build embeds the supplied cubins. After editing CUDA source, use an nvcc toolchain with sm_61 support, run `NVCC=/path/to/nvcc bash kernels/build.sh`, and review the resulting cubins and hash manifest.

## Gemma 4 12B

Use a compatible GGUF whose target weight matrices are Q4_0. The converter checks the tensor types. Prepare the matching tokenizer locally.

```bash
./bin/gguf-to-pq4 -gguf /path/to/compatible-gemma4.gguf -out data/gemma4-12b-qat.pq4m
python3 tools/reference/m7_prepare_tokenizer.py \
  --gguf /path/to/compatible-gemma4.gguf --tokenizer /path/to/tokenizer.json \
  --gguf-python /path/to/llama.cpp/gguf-py --out data/m7-api/tokenizer
```

Preparation checks the vocabulary and merges against the GGUF. Reference and conversion scripts have individual Python dependencies. The server runs without Python. For MTP, supply a matching Q4_0 assistant GGUF with `-mtp /path/to/assistant.gguf`. Weights and tokenizers are supplied separately. A pinned inventory of upstream revisions, asset hashes, and Python dependencies is pending. [Server startup](../README.en.md)

## LFM3 1.7B

```bash
./bin/pq4-convert -model-dir /path/to/lfm3-model -out data/lfm3.pq4m
./bin/pascallm-server -architecture lfm3 -pq4 data/lfm3.pq4m -tokenizer-dir /path/to/lfm3-model
```

The converter's `-h` lists imatrix and mixed PQ8 options.

## Validation

```bash
CGO_ENABLED=0 go test ./...
CGO_ENABLED=0 go vet ./...
CGO_ENABLED=1 go test -race ./internal/server ./internal/generate
PASCA_KV_GPU_TEST=1 go test ./internal/gemma4gpu -run TestKVAttentionGPU -count=1
```

The optional race check requires a C compiler. GPU tests are opt-in. For model-dependent checks, set `PASCA_DATA_DIR` and `PASCA_MODELS_DIR` to prepared asset roots, then run the corresponding MTP or prefix test.

```bash
PASCA_MTP_GPU_TEST=1 go test ./internal/gemma4gpu -run TestMTPSelectionGPU -count=1
PASCA_PREFIX_GPU_TEST=1 go test ./internal/gemma4gpu -run TestPrefixSnapshotGPU -count=1
```
