#!/usr/bin/env bash
# kernels/*.cu を GTX 1080 Ti（sm_61）向けの cubin にコンパイルし、internal/kernels/ に置く。
# Go のビルドは cubin を go:embed で埋め込むだけなので、nvcc が必要なのはこのスクリプトだけ。
# 実行例（リポジトリのルートで）: bash kernels/build.sh
set -euo pipefail

repository_root=$(cd "$(dirname "$0")/.." && pwd)
nvcc_path=${NVCC:-nvcc}
target_architecture=sm_61
output_dir=$repository_root/internal/kernels

cd "$repository_root"
for source in kernels/*.cu; do
    name=$(basename "$source" .cu)
    output=$output_dir/${name}_${target_architecture}.cubin
    "$nvcc_path" -cubin -O3 -arch="$target_architecture" -Wno-deprecated-gpu-targets \
        -o "$output" "$source"
    echo "built $output"
done
# どの nvcc で作った cubin かを記録する（cubin はリポジトリに含めるため）。
"$nvcc_path" --version | tail -n 2 > "$output_dir/nvcc-version.txt"
sha256sum "$output_dir"/*.cubin | sed "s|$repository_root/||" > "$output_dir/cubin.sha256"
