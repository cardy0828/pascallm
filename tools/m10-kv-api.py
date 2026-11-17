"""M10 の KV 精度・MTP 有無を固定の会話課題と速度入力で比較する。子サーバーは逐次実行して終了する。"""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import socket
import subprocess
import time


def load(name, filename):
    """同じ tools ディレクトリの既存 API ヘルパーを読む。"""
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(filename))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def cases():
    """速度用 6 入力と、正解を事前に固定した 8 課題を返す。正答の採点は本文だけで行う。"""
    speed=load('speed','m9-mtp-bench.py')
    result=[dict(name=name,messages=[dict(role='user',content=p)],answer=None,max_tokens=256) for name,p in speed.prompts()]
    tasks=[
      ('ja_dialogue',[dict(role='user',content='来週の会議は木曜日です。'),dict(role='assistant',content='木曜日ですね。'),dict(role='user',content='訂正します。金曜日に変更しました。現在の曜日だけ答えてください。')],'金曜日'),
      ('ja_math','120円のりんご3個と200円の梨2個を買い1000円支払いました。お釣りの金額を数字と円だけで答えてください。','240円'),
      ('ja_order','赤、青、緑の順に並んでいます。中央の色だけ答えてください。','青'),
      ('code_loop','Goの次のコードの出力を数字だけで答えてください。\ns:=0; for i:=1; i<=5; i++ { if i%2==1 { s+=i } }; fmt.Println(s)','9'),
      ('code_slice','Goの次のコードの出力を数字だけで答えてください。\na:=[]int{2,4,6}; b:=a[1:]; b[0]=9; fmt.Println(a[1])','9'),
      ('code_defer','Goで func f() (n int) { defer func(){ n+=2 }(); return 5 } の f() の戻り値を数字だけで答えてください。','7'),
    ]
    notes='\n'.join(f'記録{i}: 定例業務の担当は田中です。点検を終えてから報告書を提出します。期限は月末です。' for i in range(140))
    tasks.append(('long_ja_retrieval',notes[:len(notes)//2]+'\n唯一の合言葉は「青い灯台」です。\n'+notes[len(notes)//2:]+'\n記録中の唯一の合言葉だけ答えてください。','青い灯台'))
    code='\n'.join(f'func item{i}() int {{ return {i+100} }}' for i in range(240))
    tasks.append(('long_code_retrieval',code+'\nfunc specialResult() int { return 73 }\n上の specialResult() の戻り値を数字だけで答えてください。','73'))
    for name,p,answer in tasks:
        messages=p if isinstance(p,list) else [dict(role='user',content=p)]
        result.append(dict(name=name,messages=messages,answer=answer,max_tokens=64))
    return result


def correct(text,answer):
    """空白・句読点・コード囲みを除いた完全一致で採点する。数字の部分一致では誤答を通さない。"""
    return re.sub(r'[\s。．.!！`「」"\u3000]','',text)==answer


def main():
    """4 条件を逐次起動し、応答、採点、速度、MTP の受理分布を保存する。"""
    parser=argparse.ArgumentParser();parser.add_argument('--server',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    inputs=cases();(out/'cases.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2)+'\n')
    api=load('api','m3-api-check.py');records=[]
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);probe.bind(('127.0.0.1',18122))
    for kv in ['fp16','k-fp32']:
      for mtp in [False,True]:
        mode=kv+('-mtp' if mtp else '-normal');logpath=out/(mode+'.log')
        command=[args.server,'-architecture','gemma4','-kv-cache',kv,'-tokenizer-dir','./data/m7-api/tokenizer','-addr','127.0.0.1:18122','-max-context','8192']
        if mtp:command+=['-mtp','./models/gemma-4-12b-it-qat/mtp-gemma-4-12B-it.gguf']
        with logpath.open('w') as log:
          child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
          try:
            deadline=time.monotonic()+90
            while True:
              if child.poll() is not None:raise RuntimeError('server exited '+mode)
              try:
                status,health=api.get('http://127.0.0.1:18122','/health')
                if status==200 and health.get('status')=='ok':break
              except Exception:pass
              if time.monotonic()>deadline:raise RuntimeError('health timeout')
              time.sleep(.5)
            for c in inputs:
              status,response=api.post('http://127.0.0.1:18122','/v1/chat/completions',dict(messages=c['messages'],temperature=0,seed=42,max_tokens=c['max_tokens']))
              if status!=200:raise RuntimeError(response)
              msg=api.message_of(response);row=dict(mode=mode,case=c['name'],answer=c['answer'],correct=None if c['answer'] is None else correct(msg.get('content',''),c['answer']),response=response)
              records.append(row)
              with (out/'requests.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
              print(json.dumps(dict(mode=mode,case=c['name'],correct=row['correct'],tps=response['timings']['predicted_per_second'])),flush=True)
          finally:
            child.terminate()
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        metrics=[json.loads(l.split('generation_metrics=',1)[1]) for l in logpath.read_text().splitlines() if 'generation_metrics=' in l]
        (out/(mode+'-metrics.json')).write_text(json.dumps(metrics,indent=2)+'\n')
    for kv in ['fp16','k-fp32']:
      for c in inputs:
        matching=[r for r in records if r['case']==c['name'] and r['mode'].startswith(kv+'-')]
        if matching[0]['response']['choices'][0]['message']!=matching[1]['response']['choices'][0]['message']:raise RuntimeError('MTP output differs '+kv+'/'+c['name'])
    (out/'summary.json').write_text(json.dumps([dict(mode=r['mode'],case=r['case'],correct=r['correct'],timings=r['response']['timings']) for r in records],ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
