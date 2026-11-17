"""Gemma4 の MTP 有無で API 応答と速度を比較し、既存 API 検査も行う。
例: python3 tools/m8-api-check.py --server ./data/m8-speed/server-mtp
専用ポートが空いていることを確認し、このツールが起動した子プロセスだけを終了する。
"""
import argparse
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import time


def main():
    """通常生成と MTP を同じ要求で比較する。GPU を同時使用するサーバーは起動しない。"""
    parser=argparse.ArgumentParser()
    parser.add_argument('--server',required=True)
    parser.add_argument('--kv-cache',default='',help='Gemma4 KV の比較指定（省略時はサーバー既定）')
    parser.add_argument('--out',default='results/20261117-m8-speed')
    args=parser.parse_args()
    output=Path(args.out);output.mkdir(parents=True,exist_ok=True)
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        probe.bind(('127.0.0.1',18121))
    spec=importlib.util.spec_from_file_location('api',Path(__file__).with_name('m3-api-check.py'))
    api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
    base='http://127.0.0.1:18121'
    prompts=[
        '光合成の仕組みを日本語で詳しく説明してください。',
        'Goで文字列のスライスから重複を除去する関数を書き、動作も説明してください。',
        '雨の日に古い図書館を訪れた学生の短い物語を書いてください。',
        '120円のりんごを3個、200円の梨を2個買い、1000円払いました。計算の手順とお釣りを説明してください。',
    ]
    records=[];baseline=[]
    for enabled in (False,True):
        command=[args.server,'-architecture','gemma4','-tokenizer-dir','./data/m7-api/tokenizer','-addr','127.0.0.1:18121','-max-context','8192']
        if args.kv_cache:command+=['-kv-cache',args.kv_cache]
        if enabled:command+=['-mtp','./models/gemma-4-12b-it-qat/mtp-gemma-4-12B-it.gguf']
        with (output/('api-server-mtp.log' if enabled else 'api-server-normal.log')).open('w') as log:
            process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
            try:
                deadline=time.monotonic()+90
                while True:
                    if process.poll() is not None:raise RuntimeError('server exited during startup')
                    try:
                        status,health=api.get(base,'/health')
                        if status==200 and health.get('status')=='ok':break
                    except Exception:pass
                    if time.monotonic()>deadline:raise RuntimeError('server health timeout')
                    time.sleep(0.5)
                for index,prompt in enumerate(prompts):
                    body=dict(messages=[dict(role='user',content=prompt)],temperature=0,max_tokens=256,seed=42)
                    status,result=api.post(base,'/v1/chat/completions',body)
                    if status!=200:raise RuntimeError(result)
                    message=api.message_of(result)
                    if not enabled:baseline.append(message)
                    same=not enabled or message==baseline[index]
                    item={'mtp':enabled,'prompt':prompt,'same_response':same,'response':result}
                    records.append(item)
                    print(json.dumps(item,ensure_ascii=False),flush=True)
                    if not same:raise RuntimeError('MTP response differs')
                if enabled:
                    for script,extra in [('m7-api-check.py',[]),('m6-api-check.py',['--architecture','gemma4'])]:
                        with (output/('mtp-'+script+'.jsonl')).open('w') as checklog:
                            subprocess.run(['python3',str(Path(__file__).with_name(script)),'--base',base]+extra,stdout=checklog,stderr=subprocess.STDOUT,check=True)
            finally:
                process.terminate()
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:process.kill();process.wait()
    (output/'api-comparison.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    main()
