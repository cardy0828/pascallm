"""Gemma 12B GGUF と既存 tokenizer.json の語彙・マージを照合して API 用アセットを用意する。

例: python m7_prepare_tokenizer.py --gguf model.gguf --tokenizer tokenizer.json --out /path/to/tokenizer
GGUF の Python モジュールを --gguf-python で指定する。元のファイルは変更しない。
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


def main():
    """全語彙・全マージ一致時だけコピーし、GGUF のテンプレートと由来を保存する。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--gguf', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--gguf-python', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.gguf_python)
    from gguf import GGUFReader
    source = Path(args.tokenizer)
    config = json.loads(source.read_text())
    reader = GGUFReader(args.gguf)
    tokens = reader.fields['tokenizer.ggml.tokens'].contents()
    merges = reader.fields['tokenizer.ggml.merges'].contents()
    vocab = config['model']['vocab']
    if len(tokens) != len(vocab) or any(vocab.get(token) != index for index, token in enumerate(tokens)):
        raise ValueError('GGUF と tokenizer.json の語彙が一致しない')
    if merges != [' '.join(pair) for pair in config['model']['merges']]:
        raise ValueError('GGUF と tokenizer.json のマージが一致しない')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, out / 'tokenizer.json')
    (out / 'chat_template.jinja').write_text(reader.fields['tokenizer.chat_template'].contents())
    (out / 'generation_config.json').write_text(json.dumps(dict(do_sample=True, temperature=1.0, top_k=64, top_p=0.95, repetition_penalty=1.0), indent=2))
    evidence = dict(gguf=args.gguf, tokenizer=str(source), vocab_size=len(tokens), merges=len(merges), all_equal=True, tokenizer_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    (out / 'source.json').write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence))


if __name__ == '__main__':
    main()
