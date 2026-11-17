"""API を通した生成速度を測ります（pascallm-server の timings を記録する）。

M3 の api-check の速度の項目と同じ要求（日本語の長い文章を 512 トークン、思考なし）を、
既定の選び方（seed 11）と temperature 0・ペナルティなし（GPU の argmax だけ）で、それぞれ --runs 回送ります。
--prompt-file を指定すると、長いプロンプトの処理速度（M4）も測ります。ファイルの先頭 --prompt-chars 文字を
1 つの user の発言にし（チャットテンプレートの印 <|im_start|> などは除く）、32 トークンだけ生成します。
1 行に 1 回分の JSON を標準出力に書きます。

実行例（Linux、サーバーを起動した後）:
    python3 tools/api-speed.py --base http://127.0.0.1:18120 --label gpu_candidates > speed.jsonl
"""

import argparse
import hashlib
import json
import time
import urllib.request

ESSAY = "日本の四季について、それぞれの季節の行事や食べ物を交えて、できるだけ詳しく長い文章で説明してください。"
MAX_TOKENS = 512
LONG_PROMPT_MAX_TOKENS = 32  # 長いプロンプトの測定で生成するトークン数（プロンプトの処理を測るため少なくする）
TEMPLATE_MARKERS = ["<|im_start|>user", "<|im_start|>assistant", "<|im_start|>", "<|im_end|>"]
MODES = [
    ("default_sampling", {"seed": 11}),
    ("greedy_gpu", {"temperature": 0, "repetition_penalty": 1.0}),
]


def chat(base, options, content=ESSAY, max_tokens=MAX_TOKENS):
    """chat/completions に 1 回要求し、(応答の本体, 経過秒) を返します。例: chat("http://127.0.0.1:18120", {"seed": 11})"""
    payload = {
        "model": "lfm3-1.7b-pq4",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        **options,
    }
    request = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request) as response:
        body = json.loads(response.read())
    return body, time.perf_counter() - started


def load_long_prompt(path, chars):
    """長いプロンプトの文章を読みます。テンプレートの印を除き、先頭 chars 文字を返します。例: load_long_prompt("corpus.txt", 6000)"""
    with open(path, encoding="utf-8") as source:
        text = source.read()
    for marker in TEMPLATE_MARKERS:
        text = text.replace(marker, "")
    return text[:chars]


def main():
    """引数を読み、各モードを --runs 回ずつ測って JSON の行を書きます。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:18120")
    parser.add_argument("--label", required=True, help="記録に付ける名前（サーバーの設定など）")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--prompt-file", help="長いプロンプトの文章のファイル（省略すると測らない）")
    parser.add_argument("--prompt-chars", type=int, default=6000, help="長いプロンプトに使う文字数")
    args = parser.parse_args()
    long_prompt = load_long_prompt(args.prompt_file, args.prompt_chars) if args.prompt_file else None
    for run in range(args.runs):
        for mode, options in MODES:
            body, elapsed = chat(args.base, options)
            print(json.dumps({
                "label": args.label, "run": run, "mode": mode, "wall_seconds": round(elapsed, 3),
                "completion_tokens": body["usage"]["completion_tokens"],
                "predicted_per_second": body["timings"]["predicted_per_second"],
                "prompt_per_second": body["timings"]["prompt_per_second"],
                # 設定を変えても同じ文章になったかを比べるための、本文の短いハッシュ
                "content_sha1": hashlib.sha1(body["choices"][0]["message"]["content"].encode("utf-8")).hexdigest()[:12],
            }, ensure_ascii=False), flush=True)
        if long_prompt is None:
            continue
        # 長いプロンプト。直前の要求と先頭が違うため、KV キャッシュは使われず全部を処理し直す。
        body, elapsed = chat(args.base, {"temperature": 0, "repetition_penalty": 1.0}, content=long_prompt,
                             max_tokens=LONG_PROMPT_MAX_TOKENS)
        print(json.dumps({
            "label": args.label, "run": run, "mode": "long_prompt", "wall_seconds": round(elapsed, 3),
            "prompt_tokens": body["usage"]["prompt_tokens"],
            "cached_tokens": body["usage"].get("prompt_tokens_details", {}).get("cached_tokens"),
            "prompt_per_second": body["timings"]["prompt_per_second"],
            "completion_tokens": body["usage"]["completion_tokens"],
            "content_sha1": hashlib.sha1(body["choices"][0]["message"]["content"].encode("utf-8")).hexdigest()[:12],
        }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
