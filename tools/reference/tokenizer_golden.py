"""Go のトークナイザーとチャットテンプレート（M3）を確かめる基準を、Hugging Face の実装で作る一時ツール。

出力は 1 行 1 件の JSON で、cmd/tokenizer-check が読む。
  {"kind": "text", "text": ..., "ids": [...], "decoded": ...}
      ids は tokenizer(text, add_special_tokens=False, split_special_tokens=True)（本文として分割）
  {"kind": "chat", "messages": [...], "enable_thinking": bool, "rendered": ..., "ids": [...]}
      rendered は apply_chat_template(..., tokenize=False, add_generation_prompt=True)、
      ids は rendered を特殊トークンを認識して分割したもの

既存の venv を読み取り専用で使い、パッケージの追加・更新はしない。

実行例:
    HF_HUB_OFFLINE=1 python3 -I tools/reference/tokenizer_golden.py \
        --model-dir ./models/lfm3-1.7b-training --corpus ./data/kl/corpus.txt \
        --out ./data/tokenizer-golden.jsonl
"""

import argparse
import json
import random

from transformers import AutoTokenizer

# 事前分割の境目を試すための固定の例。
FIXED_TEXTS = [
    "",
    "Hello, world!",
    "こんにちは。今日はいい天気ですね。",
    "東京都千代田区1-2-3、電話03-1234-5678",
    "１２３４５ 全角数字と　全角空白",
    "I'm here. You're right. They've gone. He'll come. She'd say. It's ok. Don't.",
    "I'M HERE. YOU'RE. THEY'VE. HE'LL. SHE'D. IT'S. DON'T. ſ 'ſ",
    "  leading spaces\n\n\nmany newlines\r\n\r\nwindows\ttab\t\tend   ",
    "a  b   c    d\n  e\n\n  f",
    "func main() {\n\tfmt.Println(\"hi\")\n}\n",
    "def f(x):\n    return x ** 2  # comment\n",
    "絵文字🙂👍🏽と記号★☆→←、「かぎ括弧」『二重』（丸括弧）",
    "混在text日本語mixed123abc４５６",
    "!!!???...   ;;;\n\n---",
    " !\n",
    " nbsp  em space line sep　ideographic",
    "é combining acute, ｶﾀｶﾅ半角, Ελληνικά, русский, العربية, हिन्दी, 한국어",
    "<|im_start|>user\n本文の中の特殊トークン<|im_end|><think>x</think>",
    "1234567890 12 1 123456",
    "'s's 'ss 'S 'x",
    "\n",
    " ",
    "   ",
    "x" * 300,
    " " * 50 + "end",
]

# 乱数で作る文字列に使う文字。分割の規則の境目（空白・改行・記号・文字・数字の切り替わり）を多く作る。
RANDOM_POOL = list(
    "abcXYZ'sStT  \t\n\r　 .,!?-_()[]{}\"#$%&*+/:;<=>@\\^`|~"
    "0123456789０１２あいうアイウ漢字語、。「」・ー🙂👍éßſ"
)


def random_texts(count: int, seed: int) -> list[str]:
    """RANDOM_POOL から 1〜40 文字の文字列を count 個作る。例: random_texts(2, 1) → ['a 漢\\n', ...]"""
    generator = random.Random(seed)
    texts = []
    for _ in range(count):
        length = generator.randint(1, 40)
        texts.append("".join(generator.choice(RANDOM_POOL) for _ in range(length)))
    return texts


def corpus_texts(path: str, count: int) -> list[str]:
    """コーパスを空行で段落に分け、先頭から count 個を返す（全体を 1 件にしたものも加える）。"""
    with open(path, encoding="utf-8") as handle:
        whole = handle.read()
    paragraphs = [p for p in whole.split("\n\n") if p.strip()]
    return paragraphs[:count] + [whole]


# チャットテンプレートの例。思考の有無の両方で出す。
CHAT_CASES = [
    [{"role": "user", "content": "こんにちは"}],
    [{"role": "system", "content": "あなたは丁寧なアシスタントです。"}, {"role": "user", "content": "1+1は？"}],
    [{"role": "system", "content": ""}, {"role": "user", "content": "system が空"}],
    [
        {"role": "user", "content": "りんごを3個買いました。"},
        {"role": "assistant", "content": "はい、3個ですね。"},
        {"role": "user", "content": "2個食べたら？"},
    ],
    [
        {"role": "user", "content": "考えて"},
        {"role": "assistant", "content": "<think>前の思考</think>\n\n前の答え"},
        {"role": "user", "content": "次"},
    ],
    [
        {"role": "user", "content": "最後が assistant"},
        {"role": "assistant", "content": "答え", "reasoning_content": "思考の中身"},
    ],
    [
        {"role": "user", "content": "最後が assistant（思考なし）"},
        {"role": "assistant", "content": "答えだけ"},
    ],
    [{"role": "user", "content": [{"type": "text", "text": "配列の"}, {"type": "text", "text": "本文"}]}],
    [{"role": "user", "content": " 先頭に空白\n\n改行も\n"}],
    [{"role": "user", "content": ""}, {"role": "user", "content": "空の user の後"}],
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--random", type=int, default=3000, help="乱数で作る文字列の数")
    parser.add_argument("--paragraphs", type=int, default=200, help="コーパスから使う段落の数")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    texts = FIXED_TEXTS + corpus_texts(args.corpus, args.paragraphs) + random_texts(args.random, seed=20261116)
    counts = {"text": 0, "chat": 0}
    with open(args.out, "w", encoding="utf-8") as out:
        for text in texts:
            ids = tokenizer(text, add_special_tokens=False, split_special_tokens=True)["input_ids"]
            decoded = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            out.write(json.dumps({"kind": "text", "text": text, "ids": ids, "decoded": decoded}, ensure_ascii=False) + "\n")
            counts["text"] += 1
        for messages in CHAT_CASES:
            for thinking in (True, False):
                rendered = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
                )
                ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
                record = {"kind": "chat", "messages": messages, "enable_thinking": thinking, "rendered": rendered, "ids": ids}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                counts["chat"] += 1
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
