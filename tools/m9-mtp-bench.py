"""M8 と M9 の固定/自動候補数を日本語・コード・長文脈の同じ API 要求で比較する。
例: python3 tools/m9-mtp-bench.py --out results/20261117-m9/api
自分で起動した子プロセスだけを終了し、各構成を逐次実行する。
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import time


def prompts():
    """短文 4 種と長文 2 種の固定入力を返す。長さは API の prompt_tokens を記録する。"""
    cases=[('explanation','光合成の仕組みを日本語で詳しく説明してください。'),
           ('code','Goで文字列のスライスから重複を除去する関数を書き、動作も説明してください。'),
           ('story','雨の日に古い図書館を訪れた学生の短い物語を書いてください。'),
           ('arithmetic','120円のりんごを3個、200円の梨を2個買い、1000円払いました。計算の手順とお釣りを説明してください。')]
    notes='\n'.join(f'第{i}回会議: 開発部は検索機能を改善し、営業部は顧客の要望を整理する。期限は月末、担当は田中。品質検査を完了してから公開する。' for i in range(1,106))
    code='\n'.join(f'func value{i}(x int) int {{ if x < 0 {{ return 0 }}; return x + {i} }}' for i in range(1,161))
    cases += [('long_japanese',notes+'\n以上の記録から共通の方針と注意点を日本語でまとめてください。'),
              ('long_code',code+'\n以上のGoコードに共通する動作と保守性の問題を説明し、改善例を提示してください。')]
    return cases


def main():
    """各条件で出力一致を確認し、速度と要求単位の受理分布を保存する。"""
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default='results/20261117-m9/api')
    parser.add_argument('--modes',default='m8,fixed1,fixed2,fixed3,fixed7,auto3,auto7')
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--server',default='./data/m9/server-m9')
    args=parser.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    cases=prompts();(out/'prompts.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n')
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        probe.bind(('127.0.0.1',18121))
    spec=importlib.util.spec_from_file_location('api',Path(__file__).with_name('m3-api-check.py'))
    api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
    baseline={};records=[]
    for mode in args.modes.split(','):
        binary='./data/m9/server-before' if mode=='m8' else args.server
        command=[binary,'-architecture','gemma4','-tokenizer-dir','./data/m7-api/tokenizer','-addr','127.0.0.1:18121','-max-context','8192','-mtp','./models/gemma-4-12b-it-qat/mtp-gemma-4-12B-it.gguf']
        if mode!='m8':
            command+=['-mtp-draft',mode[-1]]
            if mode.startswith('auto'):command+=['-mtp-auto']
        logpath=out/(mode+'.log')
        with logpath.open('w') as log:
            process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
            try:
                deadline=time.monotonic()+90
                while True:
                    if process.poll() is not None:raise RuntimeError('server exited: '+mode)
                    try:
                        status,health=api.get('http://127.0.0.1:18121','/health')
                        if status==200 and health.get('status')=='ok':break
                    except Exception:pass
                    if time.monotonic()>deadline:raise RuntimeError('health timeout: '+mode)
                    time.sleep(0.5)
                for repeat in range(args.repeats):
                    for name,prompt in cases:
                        status,result=api.post('http://127.0.0.1:18121','/v1/chat/completions',dict(messages=[dict(role='user',content=prompt)],temperature=0,max_tokens=256,seed=42))
                        if status!=200:raise RuntimeError(result)
                        message=api.message_of(result)
                        digest=hashlib.sha256(json.dumps(message,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
                        if name not in baseline:baseline[name]=digest
                        record=dict(mode=mode,repeat=repeat,case=name,same_output=digest==baseline[name],sha256=digest,response=result)
                        records.append(record)
                        with (out/'requests.jsonl').open('a') as stream:stream.write(json.dumps(record,ensure_ascii=False)+'\n')
                        print(json.dumps(dict(mode=mode,case=name,repeat=repeat,same_output=record['same_output'],prompt=result['usage']['prompt_tokens'],tps=result['timings']['predicted_per_second'])),flush=True)
                        if not record['same_output']:raise RuntimeError('output differs: '+mode+'/'+name)
            finally:
                process.terminate()
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:process.kill();process.wait()
        metrics=[json.loads(line.split('generation_metrics=',1)[1]) for line in logpath.read_text().splitlines() if 'generation_metrics=' in line]
        if mode!='m8' and len(metrics)!=len(cases)*args.repeats:raise RuntimeError('missing per-request metrics')
        (out/(mode+'-metrics.json')).write_text(json.dumps(metrics,indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps([dict(mode=r['mode'],repeat=r['repeat'],case=r['case'],same_output=r['same_output'],timings=r['response']['timings']) for r in records],indent=2)+'\n')


if __name__=='__main__':
    main()
