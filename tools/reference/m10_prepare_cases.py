"""M10 の KV 比較用に、日本語会話と Go コードの固定入力を作る。評価用の反復文であり自然な対話ベンチではない。"""
import hashlib
import json
from pathlib import Path
from tokenizers import Tokenizer


def main():
    """512/4096/7168 トークンの教師入力を保存する。各入力の末尾 64 位置を比較する。"""
    out=Path('./data/m10');out.mkdir(exist_ok=True)
    tokenizer=Tokenizer.from_file('./data/m7-api/tokenizer/tokenizer.json')
    japanese='\n'.join(f'利用者: 第{i}回の予定を確認します。会議は金曜日の午後三時、場所は図書館です。担当は田中さんです。\n応答: 金曜日の午後三時に図書館で会議ですね。資料を木曜日までに準備し、変更があれば参加者へ連絡します。\n' for i in range(220))
    code='\n'.join(f'// collect{i} は負の値を除き、入力の順序を保持する。\nfunc collect{i}(values []int) []int {{\n result := make([]int, 0, len(values))\n for _, value := range values {{ if value >= 0 {{ result = append(result, value) }} }}\n return result\n}}\n' for i in range(220))
    cases=[];bos=tokenizer.token_to_id('<bos>')
    if bos is None:raise RuntimeError('missing BOS')
    for category,text in [('japanese',japanese),('code',code)]:
        tokens=[bos]+tokenizer.encode(text,add_special_tokens=False).ids
        for length in [512,4096,7168]:
            if len(tokens)<length:raise RuntimeError('insufficient input')
            cases.append(dict(name=f'{category}-{length}',tokens=tokens[:length]))
    encoded=(json.dumps(cases,ensure_ascii=False)+'\n').encode()
    (out/'cases.json').write_bytes(encoded)
    Path('results/20261117-m10/cases-manifest.json').write_text(json.dumps(dict(sha256=hashlib.sha256(encoded).hexdigest(),path=str(out/'cases.json'),cases=[dict(name=c['name'],tokens=len(c['tokens'])) for c in cases]),indent=2)+'\n')


if __name__=='__main__':main()
