"""テキストを Hugging Face のトークナイザーでトークン列にする一時ツール（cmd/lfm3-kl の入力用）。

llama-perplexity と同じ列になるよう、先頭に BOS を 1 個置き、本文は特殊トークンを解釈せずに分割する。
既存の venv を読み取り専用で使い、パッケージの追加・更新はしない。

実行例:
    HF_HUB_OFFLINE=1 python3 -I tools/reference/tokenize_text.py \
        --model-dir ./models/lfm3-1.7b-training --text-file corpus.txt --tokens 1024 --out tokens.json
"""

import argparse
import json

from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--tokens", type=int, required=True, help="BOS を含めたトークン数")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    with open(args.text_file, encoding="utf-8") as handle:
        text = handle.read()
    # split_special_tokens=True: 本文中の "<|im_start|>" などを特殊トークンにせず、普通の文字列として分割する。
    body = tokenizer(text, add_special_tokens=False, split_special_tokens=True)["input_ids"]
    if len(body) + 1 < args.tokens:
        raise SystemExit(f"text has only {len(body)} tokens")
    token_ids = [tokenizer.bos_token_id] + body[: args.tokens - 1]
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"token_ids": token_ids, "text_file": args.text_file, "bos_token_id": tokenizer.bos_token_id}, handle)
    print(json.dumps({"tokens": len(token_ids), "body_tokens": len(body)}))


if __name__ == "__main__":
    main()
