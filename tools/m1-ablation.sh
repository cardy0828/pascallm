#!/usr/bin/env bash
# M1 の PQ4 品質の切り分け: 同じ 1024 トークンで PQ4 の各条件と llama.cpp の既存形式の KL を測る。
# 実行例: bash ./tools/m1-ablation.sh
# 出力: ./results/20261114-m1/kl-*.jsonl, llama-*.log
set -euo pipefail

repo=.
data=./data
results=$repo/results/20261114-m1
model_dir=./models/lfm3-1.7b-training
corpus=./data/kl/corpus.txt
python=python3
llama_bin=./external/llama.cpp/build/bin
gguf_dir=./data/baseline
bf16_gguf=./models/lfm3-1.7b-bf16/LFM3-1.7B-BF16.gguf
# 1080 Ti を UUID で固定する (nvidia-smi の番号と CUDA の番号が逆のため)。
gtx1080ti_uuid=${PASCA_DEVICE_UUID:?PASCA_DEVICE_UUID must name the target GPU}
context_tokens=1024

cd "$repo"
mkdir -p "$results" "$data/kl" "$data/bin"

echo "== tokenize"
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= "$python" -I tools/reference/tokenize_text.py \
    --model-dir "$model_dir" --text-file "$corpus" --tokens "$context_tokens" --out "$data/kl/tokens-1024.json"

echo "== build"
go build -o "$data/bin/pq4-convert" ./cmd/pq4-convert
go build -o "$data/bin/lfm3-kl" ./cmd/lfm3-kl

if [[ ! -f $data/lfm3-1.7b-pq4-absmax-lmf16.pq4m ]]; then
    echo "== convert absmax + lm_head f16"
    "$data/bin/pq4-convert" -model-dir "$model_dir" -method absmax -lm-head-kind f16 \
        -out "$data/lfm3-1.7b-pq4-absmax-lmf16.pq4m" > "$results/convert-absmax-lmf16.jsonl"
fi

# run_kl は 1 条件の KL を測ります。引数: 結果の名前、.pq4m のパス、追加の引数。
run_kl() {
    local label=$1 model=$2
    shift 2
    echo "== kl $label"
    /usr/bin/time -f "elapsed=%e maxrss_kb=%M" -o "$results/kl-$label.time" \
        "$data/bin/lfm3-kl" -tokens "$data/kl/tokens-1024.json" -base-model-dir "$model_dir" \
        -base-logits "$data/kl/base-fp32-1024.f32" -pq4 "$model" -label "$label" "$@" > "$results/kl-$label.jsonl"
    tail -2 "$results/kl-$label.jsonl"
}

run_kl absmax "$data/lfm3-1.7b-pq4-absmax.pq4m"
run_kl absmax-lmf16 "$data/lfm3-1.7b-pq4-absmax-lmf16.pq4m"
run_kl absmax-lmf16-dequant "$data/lfm3-1.7b-pq4-absmax-lmf16.pq4m" -dequantize
run_kl absmax-dequant "$data/lfm3-1.7b-pq4-absmax.pq4m" -dequantize
run_kl mse "$data/lfm3-1.7b-pq4-mse.pq4m"

echo "== llama.cpp (-c $context_tokens --chunks 1)"
export LD_LIBRARY_PATH=${CUDA_HOME:-/usr/local/cuda}/lib64:$llama_bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
CUDA_VISIBLE_DEVICES=$gtx1080ti_uuid "$llama_bin/llama-perplexity" -m "$bf16_gguf" -f "$corpus" \
    -c "$context_tokens" --chunks 1 -ngl 99 --kl-divergence-base "$data/kl/llama-base-bf16-1024.kld" \
    > "$results/llama-bf16.log" 2>&1
for quant in Q4_0 Q4_K_M IQ4_XS Q8_0; do
    CUDA_VISIBLE_DEVICES=$gtx1080ti_uuid "$llama_bin/llama-perplexity" -m "$gguf_dir/LFM3-1.7B-$quant.gguf" \
        -c "$context_tokens" --kl-divergence-base "$data/kl/llama-base-bf16-1024.kld" --kl-divergence -ngl 99 \
        > "$results/llama-$quant.log" 2>&1
    echo "== llama $quant"
    grep -E "Mean    KLD|Median  KLD|99.0%   KLD|Maximum KLD|Same top p|Mean PPL\(Q\)/PPL\(base\)" "$results/llama-$quant.log" || true
done
echo "== done"
