"""M5 の評価文と較正文（imatrix 用）を作る一時ツール。

日本語は、既存の日本語コーパス（./data/corpus/tokens、uint16 のトークン列、文書の区切りは
eos=2、トークナイザーは data/tokenizer/pure.json）から文書を選び、文字列に戻して使います。
英語は、ホストにある英語の文書（Go の仕様書などの HTML からタグを除いたもの、ライセンス文）を使います。

  - 評価文: 日本語は各分野の val から、英語は go_mem.html から作ります。1 個のテキストファイルにまとめ、全体を
    LFM3 のトークナイザーで分割して 1024 トークンずつに区切ります（先頭を BOS に置き換える。llama-perplexity に
    同じファイルを -c 1024 で渡したときと同じ区切り）。
  - 較正文: 日本語は各分野の train から、英語は評価文と別の文書から作ります。512 トークンずつ（先頭 BOS）に区切ります。

既存の venv を読み取り専用で使い、パッケージの追加・更新はしません。

実行例:
    HF_HUB_OFFLINE=1 python3 -I tools/reference/m5_prepare_text.py \
        --corpus-dir ./data/corpus \
        --model-dir ./models/lfm3-1.7b-training --out-dir ./data/m5
"""

import argparse
import hashlib
import html
import json
import random
import re
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

# 評価の区切り（1 区切り 1024 トークン）。分野ごとに EVAL_CHUNKS_PER_DOMAIN 個。
EVAL_JAPANESE_DOMAINS = ["wikipedia-other", "aozora", "egov-law"]
EVAL_ENGLISH_FILES = {"english-go-memory-model": ["/usr/local/go/doc/go_mem.html"]}
EVAL_CHUNKS_PER_DOMAIN = 2
EVAL_CHUNK_TOKENS = 1024

# 較正の区切り（1 区切り 512 トークン）の分野ごとの数。
CALIBRATION_JAPANESE_DOMAINS = {
    "wikipedia-other": 24,
    "wikipedia-person": 12,
    "wikipedia-work": 8,
    "aozora": 16,
    "egov-law": 8,
    "wikibooks": 4,
    "dialogue": 4,
}
CALIBRATION_ENGLISH_FILES = {
    "english-go-spec": (["/usr/local/go/doc/go_spec.html"], 24),
    "english-licenses": (["/usr/share/common-licenses/GPL-3", "/usr/share/common-licenses/Apache-2.0"], 12),
}
CALIBRATION_CHUNK_TOKENS = 512

# pure トークナイザーの特殊トークン（jpcorpus の tokenizer.py と同じ並び）。eos は 2。
PURE_SPECIALS = ["<pad>", "<s>", "</s>", "<unk>"]
PURE_BYTE_COUNT = 256
EOS = 2
MIN_DOCUMENT_CHARS = 200  # これより短い文書は使わない（見出しだけの文書などを除く）
MAX_DOCUMENT_CHARS = 2000  # 1 文書から使う最大の文字数（長い文書が分野を占めないようにする）
SEED = 20261117


class PureDecoder:
    """jpcorpus の pure トークナイザーの復号だけを行います（表層形をつなぐ。バイトのトークンは UTF-8 として読む）。"""

    def __init__(self, path: Path):
        data = json.loads(path.read_text(encoding="utf-8"))
        self.vocab = [("special", s) for s in PURE_SPECIALS]
        self.vocab += [("byte", f"<0x{b:02X}>") for b in range(PURE_BYTE_COUNT)]
        self.vocab += [("head", c) for c in data["chars"]] + [("inner", c) for c in data["chars"]]
        self.vocab += [("func", f) for f in data["funcs"]] + [("punct", p) for p in data["puncts"]]

    def decode(self, ids) -> str:
        """トークン番号の列を文字列に戻します。例: decode([▁花 の番号, 火 の番号]) → "花火" """
        out, buffer = [], bytearray()
        for index in ids:
            kind, text = self.vocab[index]
            if kind == "byte":
                buffer.append(int(text[3:5], 16))
                continue
            if buffer:
                out.append(buffer.decode("utf-8", "replace"))
                buffer = bytearray()
            if kind != "special":
                out.append(text)
        if buffer:
            out.append(buffer.decode("utf-8", "replace"))
        return "".join(out)


def split_documents(path: Path) -> list[np.ndarray]:
    """uint16 のトークン列を eos で文書に分けます。例: [5, 6, 2, 7, 2] → [[5, 6], [7]]"""
    tokens = np.memmap(path, dtype=np.uint16, mode="r")
    documents = []
    start = 0
    for end in np.flatnonzero(tokens == EOS):
        if end > start:
            documents.append(np.asarray(tokens[start:end]))
        start = end + 1
    return documents


def cut_text(text: str) -> str:
    """文書を MAX_DOCUMENT_CHARS 文字以内にし、できれば文の終わりで切ります。"""
    text = text.strip()
    if len(text) <= MAX_DOCUMENT_CHARS:
        return text
    head = text[:MAX_DOCUMENT_CHARS]
    end = max(head.rfind("。"), head.rfind(". "), head.rfind("\n"))
    return head[: end + 1] if end > MAX_DOCUMENT_CHARS // 2 else head


def count_tokens(lfm_tokenizer, text: str) -> int:
    """LFM3 のトークン数を返します（特殊トークンは解釈しない）。"""
    return len(lfm_tokenizer(text, add_special_tokens=False, split_special_tokens=True)["input_ids"])


def sample_japanese(decoder: PureDecoder, lfm_tokenizer, path: Path, min_tokens: int, rng: random.Random) -> str:
    """path の文書を乱数の順に並べ、LFM3 のトークン数が min_tokens 以上になるまでつなげた文字列を返します。"""
    documents = split_documents(path)
    order = list(range(len(documents)))
    rng.shuffle(order)
    parts: list[str] = []
    count = 0
    for index in order:
        text = cut_text(decoder.decode(documents[index].tolist()))
        if len(text) < MIN_DOCUMENT_CHARS:
            continue
        parts.append(text)
        count += count_tokens(lfm_tokenizer, text + "\n\n")
        if count >= min_tokens:
            return "\n\n".join(parts) + "\n\n"
    raise SystemExit(f"{path} does not have {min_tokens} tokens of usable documents")


def plain_text(path: str) -> str:
    """HTML ならタグと連続する空白を除き、それ以外はそのまま読みます。"""
    text = Path(path).read_text(encoding="utf-8")
    if path.endswith(".html"):
        text = re.sub(r"<(script|style|pre)[^>]*>.*?</\1>", " ", text, flags=re.S)
        text = html.unescape(re.sub(r"<[^>]+>", "", text))
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip() + "\n\n"


def english_text(lfm_tokenizer, paths: list[str], min_tokens: int) -> str:
    """paths の文書をつなげ、LFM3 のトークン数が min_tokens 以上あることを確かめて返します（長い分はそのまま）。"""
    text = "".join(plain_text(path) for path in paths)
    if count_tokens(lfm_tokenizer, text) < min_tokens:
        raise SystemExit(f"{paths} have fewer than {min_tokens} tokens")
    return text


def trim_to_tokens(lfm_tokenizer, text: str, tokens: int) -> str:
    """text の先頭から、LFM3 のトークン数がおよそ tokens になる位置（その後の最初の改行）までを返します。"""
    encoded = lfm_tokenizer(text, add_special_tokens=False, split_special_tokens=True, return_offsets_mapping=True)
    if len(encoded["input_ids"]) <= tokens:
        return text
    end = encoded["offset_mapping"][tokens][0]
    line = text.find("\n", end)
    return text if line < 0 else text[: line + 1] + "\n"


def make_chunks(token_ids: list[int], chunk_tokens: int, count: int, bos: int) -> list[list[int]]:
    """トークン列を chunk_tokens 個ずつに分け、各区切りの先頭を BOS に置き換えます（llama-perplexity と同じ）。"""
    chunks = []
    for index in range(count):
        chunk = list(token_ids[index * chunk_tokens : (index + 1) * chunk_tokens])
        if len(chunk) < chunk_tokens:
            raise SystemExit(f"only {len(token_ids)} tokens for {count} chunks of {chunk_tokens}")
        chunk[0] = bos
        chunks.append(chunk)
    return chunks


def build_set(parts: list[tuple[str, str]], lfm_tokenizer, chunk_tokens: int, counts: list[int], bos: int):
    """分野ごとの文章 parts（[(分野, 文章)]）をつなげ、全体を 1 回で分割して区切ります。
    区切りの分野は、区切りの 3/4 の位置の文字がどの分野の文章かで決めます（評価で採点するのは後半）。"""
    starts, offset = [], 0
    for _, text in parts:
        starts.append(offset)
        offset += len(text)
    whole = "".join(text for _, text in parts)
    encoded = lfm_tokenizer(whole, add_special_tokens=False, split_special_tokens=True, return_offsets_mapping=True)
    total = sum(counts)
    chunks = make_chunks(encoded["input_ids"], chunk_tokens, total, bos)
    domains = []
    probe = chunk_tokens * 3 // 4
    for index in range(total):
        char = encoded["offset_mapping"][index * chunk_tokens + probe][0]
        domains.append(parts[max(i for i, start in enumerate(starts) if start <= char)][0])
    return whole, chunks, domains


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    decoder = PureDecoder(corpus_dir / "tokenizer" / "pure.json")
    lfm_tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    bos = lfm_tokenizer.bos_token_id
    rng = random.Random(SEED)
    # 各分野の文章は「区切りの数 × 区切りのトークン数」に切りそろえます（分野の境目と区切りの境目をそろえる）。

    eval_parts, eval_counts = [], []
    for domain in EVAL_JAPANESE_DOMAINS:
        need = EVAL_CHUNKS_PER_DOMAIN * EVAL_CHUNK_TOKENS
        text = sample_japanese(decoder, lfm_tokenizer, corpus_dir / "tokens" / f"{domain}.val.bin", need, rng)
        eval_parts.append((domain, trim_to_tokens(lfm_tokenizer, text, need)))
        eval_counts.append(EVAL_CHUNKS_PER_DOMAIN)
    for domain, paths in EVAL_ENGLISH_FILES.items():
        need = EVAL_CHUNKS_PER_DOMAIN * EVAL_CHUNK_TOKENS
        text = trim_to_tokens(lfm_tokenizer, english_text(lfm_tokenizer, paths, need), need)
        eval_parts.append((domain, text))
        eval_counts.append(EVAL_CHUNKS_PER_DOMAIN)
    eval_text, eval_chunks, eval_domains = build_set(eval_parts, lfm_tokenizer, EVAL_CHUNK_TOKENS, eval_counts, bos)
    (out_dir / "eval.txt").write_text(eval_text, encoding="utf-8")
    with open(out_dir / "eval-chunks.json", "w", encoding="utf-8") as handle:
        json.dump({"chunks": eval_chunks, "domains": eval_domains, "chunk_tokens": EVAL_CHUNK_TOKENS,
                   "bos_token_id": bos, "seed": SEED, "text_file": str(out_dir / "eval.txt")}, handle)

    calibration_parts, calibration_counts = [], []
    for domain, count in CALIBRATION_JAPANESE_DOMAINS.items():
        need = count * CALIBRATION_CHUNK_TOKENS
        path = corpus_dir / "tokens" / f"{domain}.train.bin"
        text = sample_japanese(decoder, lfm_tokenizer, path, need, rng)
        calibration_parts.append((domain, trim_to_tokens(lfm_tokenizer, text, need)))
        calibration_counts.append(count)
    for domain, (paths, count) in CALIBRATION_ENGLISH_FILES.items():
        need = count * CALIBRATION_CHUNK_TOKENS
        text = trim_to_tokens(lfm_tokenizer, english_text(lfm_tokenizer, paths, need), need)
        calibration_parts.append((domain, text))
        calibration_counts.append(count)
    calibration_text, calibration_chunks, calibration_domains = build_set(
        calibration_parts, lfm_tokenizer, CALIBRATION_CHUNK_TOKENS, calibration_counts, bos)
    (out_dir / "calibration.txt").write_text(calibration_text, encoding="utf-8")
    with open(out_dir / "calibration-chunks.json", "w", encoding="utf-8") as handle:
        json.dump({"chunks": calibration_chunks, "domains": calibration_domains,
                   "chunk_tokens": CALIBRATION_CHUNK_TOKENS, "bos_token_id": bos, "seed": SEED}, handle)

    print(json.dumps({
        "eval_chunks": len(eval_chunks), "eval_domains": eval_domains, "eval_chars": len(eval_text),
        "eval_sha256": hashlib.sha256(eval_text.encode("utf-8")).hexdigest(),
        "calibration_chunks": len(calibration_chunks),
        "calibration_tokens": len(calibration_chunks) * CALIBRATION_CHUNK_TOKENS,
        "calibration_domains": {d: calibration_domains.count(d) for d in dict.fromkeys(calibration_domains)},
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
