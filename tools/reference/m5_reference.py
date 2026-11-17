"""M5 の基準 logits と重要度行列（imatrix）を、Hugging Face 実装（FP32）で作る一時ツール。

GTX 1080 Ti（sm_61）は venv の PyTorch が対応していないため、同じホストの RTX 3060 を CUDA_VISIBLE_DEVICES の
UUID で指定して使います（CPU でも動くが遅い）。TF32 は使いません（FP32 の結果を Go の参照実装とそろえるため）。
既存の venv を読み取り専用で使い、パッケージの追加・更新はしません。

サブコマンド:
  check     既存の基準（Go の FP32 参照実装で作った tokens-1024 の logits）と、この実装の logits を比べます。
  base      評価の区切り（eval-chunks.json）ごとに 1 回で計算し、採点する位置（score_from 〜 最後の 1 個前）の
            logits を float32 で書きます（区切りの順、位置の順。1 行 vocab 個）。
  imatrix   較正の区切り（calibration-chunks.json）を入れ、各 nn.Linear の入力の 2 乗の平均（入力の列ごと）を
            safetensors に書きます。名前は重みの名前（例: model.layers.0.feed_forward.w1.weight）です。

実行例（Linux）:
    export CUDA_VISIBLE_DEVICES=GPU-対象GPUのUUID   # RTX 3060
    PY="python3 -I"
    HF_HUB_OFFLINE=1 $PY tools/reference/m5_reference.py check --model-dir ./models/lfm3-1.7b-training \
        --tokens ./data/kl/tokens-1024.json --base-logits ./data/kl/base-fp32-1024.f32
    HF_HUB_OFFLINE=1 $PY tools/reference/m5_reference.py base --model-dir ... --chunks eval-chunks.json --out base-eval.f32
    HF_HUB_OFFLINE=1 $PY tools/reference/m5_reference.py imatrix --model-dir ... --chunks calibration-chunks.json --out imatrix.safetensors
"""

import argparse
import json
import time

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import AutoModelForCausalLM

# imatrix を集めるとき、一度に入れる区切りの数（RTX 3060 の 12 GB に収まる大きさ）。
IMATRIX_BATCH_CHUNKS = 4


def load_model(model_dir: str):
    """FP32 のモデルを読み込み、GPU があれば GPU に置きます。戻り値は (model, device)。"""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, dtype=torch.float32,
                                                 attn_implementation="eager")
    model.to(device).eval()
    return model, device


def chunk_logits(model, device, tokens: list[int]) -> np.ndarray:
    """1 区切りを 1 回で計算し、全位置の logits（[位置][vocab] の float32）を返します。"""
    with torch.no_grad():
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        return model(input_ids=ids).logits[0].float().cpu().numpy()


def log_softmax(rows: np.ndarray) -> np.ndarray:
    """行ごとの log(softmax) を float64 で返します。"""
    rows = rows.astype(np.float64)
    rows -= rows.max(axis=1, keepdims=True)
    return rows - np.log(np.exp(rows).sum(axis=1, keepdims=True))


def run_check(args) -> None:
    """tokens-1024 の全位置で、既存の基準（Go の FP32）とこの実装の logits の差・KL を求めます。"""
    tokens = json.load(open(args.tokens, encoding="utf-8"))["token_ids"]
    model, device = load_model(args.model_dir)
    logits = chunk_logits(model, device, tokens)
    base = np.fromfile(args.base_logits, dtype=np.float32).reshape(len(tokens), -1)
    base_log, test_log = log_softmax(base), log_softmax(logits)
    kl = (np.exp(base_log) * (base_log - test_log)).sum(axis=1)
    print(json.dumps({"kind": "check", "device": str(device), "tokens": len(tokens),
                      "max_abs_diff": float(np.abs(base - logits).max()), "mean_kl": float(kl.mean()),
                      "max_kl": float(kl.max()),
                      "same_top1": float((base.argmax(axis=1) == logits.argmax(axis=1)).mean())}))


def run_base(args) -> None:
    """評価の各区切りの、採点する位置の logits を書きます。"""
    data = json.load(open(args.chunks, encoding="utf-8"))
    model, device = load_model(args.model_dir)
    started = time.time()
    rows = 0
    with open(args.out, "wb") as handle:
        for tokens in data["chunks"]:
            score_from = len(tokens) // 2
            logits = chunk_logits(model, device, tokens)
            scored = np.ascontiguousarray(logits[score_from : len(tokens) - 1], dtype=np.float32)
            handle.write(scored.tobytes())
            rows += scored.shape[0]
    print(json.dumps({"kind": "base", "device": str(device), "chunks": len(data["chunks"]), "rows": rows,
                      "vocab": int(logits.shape[1]), "seconds": time.time() - started}))


def run_imatrix(args) -> None:
    """較正の区切りを入れ、各 nn.Linear の入力の列ごとの 2 乗の平均を集めます。"""
    data = json.load(open(args.chunks, encoding="utf-8"))
    model, device = load_model(args.model_dir)
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}

    def make_hook(name: str):
        """名前 name の nn.Linear の入力を集める関数を返します。"""
        def hook(_module, inputs):
            x = inputs[0].detach().to(torch.float64).reshape(-1, inputs[0].shape[-1])
            total = (x * x).sum(dim=0)
            sums[name] = sums[name] + total if name in sums else total
            counts[name] = counts.get(name, 0) + x.shape[0]
        return hook

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_pre_hook(make_hook(name + ".weight")))
    started = time.time()
    chunks = data["chunks"]
    with torch.no_grad():
        for start in range(0, len(chunks), IMATRIX_BATCH_CHUNKS):
            ids = torch.tensor(chunks[start : start + IMATRIX_BATCH_CHUNKS], dtype=torch.long, device=device)
            model(input_ids=ids)
    for handle in handles:
        handle.remove()
    tensors = {name: (total / counts[name]).to(torch.float32).cpu().numpy() for name, total in sums.items()}
    tokens = len(chunks) * len(chunks[0])
    save_file(tensors, args.out, metadata={"tokens": str(tokens), "chunks": str(len(chunks)),
                                           "statistic": "mean of squared inputs per input column"})
    print(json.dumps({"kind": "imatrix", "device": str(device), "tensors": len(tensors), "tokens": tokens,
                      "seconds": time.time() - started}))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check")
    check.add_argument("--model-dir", required=True)
    check.add_argument("--tokens", required=True)
    check.add_argument("--base-logits", required=True)
    for name in ("base", "imatrix"):
        command = sub.add_parser(name)
        command.add_argument("--model-dir", required=True)
        command.add_argument("--chunks", required=True)
        command.add_argument("--out", required=True)
    args = parser.parse_args()
    {"check": run_check, "base": run_base, "imatrix": run_imatrix}[args.command](args)


if __name__ == "__main__":
    main()
