"""GGUF 由来テンプレートと Hugging Face tokenizers で Go 実装の比較データを作る。
既存 venv の jinja2/tokenizers を CPU だけで使う。実行例: --assets /.../tokenizer --out /.../golden.jsonl
"""
import argparse
import copy
import json
import random
from pathlib import Path
import jinja2
from tokenizers import Tokenizer


def main():
    """日本語・空白・byte fallback・特殊タグと、ツールを含む対話の基準を保存する。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    assets = Path(args.assets)
    tok = Tokenizer.from_file(str(assets / 'tokenizer.json'))
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    template = env.from_string((assets / 'chat_template.jinja').read_text())
    texts = ['', 'こんにちは、世界！', ' a  b\n\n\tC ', '▁東京', '👨‍👩‍👧‍👦🚀🫠', 'e\u0301 café ＡＢＣ', '<bos><|turn>model\n<|think|>', 'quote "\\\n<|"|>', '𠮷野家𩸽\x00\x01']
    randomizer = random.Random(710)
    alphabet = list('日本語学習ABCxyz01239 \n\t,.;漢字▁') + ['🙂', '𠮷', '\u0301', '\r', '\x00']
    for _ in range(250):
        texts.append(''.join(randomizer.choices(alphabet, k=randomizer.randrange(1, 180))))
    corpus = Path('./data/m5/eval.txt').read_text()
    texts.extend(corpus.splitlines()[:120])
    records = []
    tok.encode_special_tokens = True
    for text in texts:
        ids = tok.encode(text, add_special_tokens=False).ids
        records.append(dict(kind='text', text=text, ids=ids, decoded=tok.decode(ids, skip_special_tokens=False)))
    tok.encode_special_tokens = False
    tools = [dict(type='function', function=dict(name='get_weather', description='天気を取得', parameters=dict(type='object', properties=dict(city=dict(type='string', description='都市'), unit=dict(type='string', enum=['celsius', 'fahrenheit'])), required=['city'])))]
    complex_tools = [dict(type='function', function=dict(name='save_data', description='保存', parameters=dict(type='object', properties=dict(z=dict(type='object', properties=dict(value=dict(type='number'), flag=dict(type='boolean')), required=['value']), items=dict(type='array', items=dict(type='object', properties=dict(name=dict(type='string')), required=['name'])), note=dict(type='string', nullable=True)), required=['z']), response=dict(type='object', description='結果')))]
    cases = [
        ([dict(role='user', content='こんにちは')], None),
        ([dict(role='system', content=' 日本語で回答。 '), dict(role='user', content=' 日本の首都は？ ')], None),
        ([dict(role='developer', content=[dict(type='text',text=' 日本語 '),dict(type='text',text=' 簡潔 ') ]), dict(role='user', content=[dict(type='text',text=' 東京 '),dict(type='text',text=' 天気 ')])], None),
        ([dict(role='user',content='質問'), dict(role='assistant',content=' 答え ',reasoning_content='古い思考'),dict(role='user',content='続けて')],None),
        ([dict(role='user',content='質問'),dict(role='assistant',content='<|channel>thought\n秘密<channel|>回答'),dict(role='user',content='続けて')],None),
        ([dict(role='user',content='東京の天気を調べて')],tools),
        ([dict(role='user',content='保存して')],complex_tools),
    ]
    call=dict(id='call_1',type='function',function=dict(name='get_weather',arguments=dict(city='東京')))
    call2=dict(id='call_2',type='function',function=dict(name='get_weather',arguments=dict(city='大阪',unit='celsius')))
    for parallel in (False, True):
        messages=[dict(role='user',content='天気は？'),dict(role='assistant',content=None,reasoning_content='調べます',tool_calls=[call,call2] if parallel else [call])]
        cases.append((copy.deepcopy(messages),tools))
        messages.append(dict(role='tool',tool_call_id='call_1',content='晴れ'))
        if parallel: messages.append(dict(role='tool',tool_call_id='call_2',content=[dict(type='text',text='雨')]))
        cases.append((copy.deepcopy(messages),tools))
        messages.append(dict(role='assistant',content='天気を確認しました。',reasoning_content='回答します'))
        cases.append((copy.deepcopy(messages),tools))
        messages.append(dict(role='user',content='ありがとう'))
        cases.append((copy.deepcopy(messages),tools))
    for messages, definitions in cases:
        for thinking in (False, True):
            rendered=template.render(messages=messages,tools=definitions,enable_thinking=thinking,add_generation_prompt=True,bos_token='<bos>')
            ids=tok.encode(rendered,add_special_tokens=False).ids
            records.append(dict(kind='chat',messages=messages,tools=definitions,enable_thinking=thinking,rendered=rendered,ids=ids))
    with open(args.out,'w') as target:
        for record in records: target.write(json.dumps(record,ensure_ascii=False)+'\n')
    print(json.dumps(dict(texts=len(texts),chats=len(cases)*2,out=args.out)))


if __name__=='__main__':
    main()
