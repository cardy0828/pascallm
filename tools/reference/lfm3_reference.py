"""LFM3 の基準 logits を Hugging Face 実装（FP32、CPU、eager attention）で作る一時ツール。

Go の参照実装（cmd/lfm3-ref）と突き合わせるための正解データを出力する。
既存の venv を読み取り専用で使い、パッケージの追加・更新はしない。

実行例（Linux）:
    HF_HUB_OFFLINE=1 python3 -I \
        tools/reference/lfm3_reference.py \
        --model-dir ./models/lfm3-1.7b-training \
        --out-dir ./data/reference-lfm3-1.7b

出力:
    reference.json      トークン列、全位置の上位 10 候補（ID と logit）、全語彙を保存した位置の一覧、環境情報
    logits_full.f32     全語彙の logits（float32 リトルエンディアン、positions_full の順に 1 行ずつ）
"""

import argparse
import hashlib
import json
import os
import pathlib
import platform

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# 上位候補として保存する数。Go 側の比較で上位 1 位・上位 5 位の一致を確かめる。
TOP_K = 10
# 窓付き attention（幅 256）の境界をまたぐよう、入力はこのトークン数以上にする。
MIN_TOKENS = 300


def build_prompt(tokenizer) -> list[int]:
    """チャットテンプレートで日本語と英語を混ぜた入力を作り、MIN_TOKENS 以上になるまで本文を足す。

    戻り値はトークン ID の列（例: [124894, 124896, ...]）。
    """
    paragraph = (
        "春の朝、港町の市場には新鮮な魚と野菜が並び、商人たちが威勢よく声を張り上げていた。"
        "The lighthouse keeper wrote down the tide table, the wind speed, and the number of ships that passed. "
        "計算問題: 1 個 120 円のりんごを 3 個と、1 本 80 円の牛乳を 2 本買うと、合計はいくらですか。"
    )
    body = paragraph
    while True:
        messages = [{"role": "user", "content": body + "\n上の文章を要約し、計算問題にも答えてください。"}]
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        if isinstance(ids, dict) or hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        if len(ids) >= MIN_TOKENS:
            return list(ids)
        body += paragraph


def file_sha256(path: pathlib.Path) -> str:
    """ファイルの SHA-256 を 16 進文字列で返す（例: "ab12..."）。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args()

    model_dir = pathlib.Path(args.model_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()

    token_ids = build_prompt(tokenizer)
    count = len(token_ids)
    with torch.no_grad():
        output = model(input_ids=torch.tensor([token_ids]), use_cache=False)
    logits = output.logits[0].float().numpy()  # [count, vocab]

    # 全語彙を保存する位置: 先頭付近、窓幅 256 の前後、最後。
    candidates = [0, 1, 2, 127, 254, 255, 256, 257, 258, count - 2, count - 1]
    positions_full = sorted({p for p in candidates if 0 <= p < count})
    logits[positions_full].astype("<f4").tofile(out_dir / "logits_full.f32")

    top_ids = np.argsort(-logits, axis=1)[:, :TOP_K]
    top_values = np.take_along_axis(logits, top_ids, axis=1)

    record = {
        "model_dir": str(model_dir),
        "safetensors_sha256": file_sha256(model_dir / "model.safetensors"),
        "token_ids": token_ids,
        "text": tokenizer.decode(token_ids),
        "vocab_size": int(logits.shape[1]),
        "positions_full": positions_full,
        "top_k": TOP_K,
        "top_ids": top_ids.tolist(),
        "top_values": top_values.astype(float).tolist(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "dtype": "float32",
            "device": "cpu",
            "attn_implementation": "eager",
            "threads": args.threads,
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE", ""),
        },
    }
    with (out_dir / "reference.json").open("w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False)
    print(json.dumps({"tokens": count, "positions_full": positions_full, "out_dir": str(out_dir)}))


if __name__ == "__main__":
    main()
