"""M11 の temperature・ツール・JSON を MTP 有無で比較する。外部ツールは実行せず呼出内容だけ検査する。"""
import argparse
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import time


def load(name,filename):
    """既存の HTTP/文法確認ヘルパーを同じディレクトリから読む。"""
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(filename));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def normalized(message):
    """応答ごとに変わる tool id と SSE index を除き、本文・思考・呼出名/引数を比較する。"""
    return dict(content=message.get('content') or '',reasoning=message.get('reasoning_content') or message.get('reasoning') or '',calls=[c['function'] for c in message.get('tool_calls') or []])


def cases(api):
    """要求を測定前に固定する。サンプラー・制約・停止の異なる経路を通す。"""
    basic=dict(messages=[dict(role='user',content='雨の日に図書館を訪れた学生の物語を日本語で書いてください。')],max_tokens=128,seed=42,temperature=.7,top_k=40,top_p=.95,min_p=.05,repetition_penalty=1.1,presence_penalty=.2,frequency_penalty=.1)
    result=[('sampling',basic),('sampling_seed7',{**basic,'seed':7}),('full_vocab',{**basic,'temperature':1.5,'top_k':0,'top_p':1,'min_p':0}),('stop',{**basic,'messages':[dict(role='user',content='1から20までの数字を順に書いてください。')],'stop':['5']})]
    tool=dict(messages=[dict(role='user',content='東京の天気を摂氏で確認してください。')],tools=[api.WEATHER],tool_choice='required',max_tokens=128,temperature=.7,top_k=40,top_p=.95,seed=42)
    result+=[('tool_greedy',{**tool,'temperature':0}),('tool_sampled',tool),('tool_named',{**tool,'tool_choice':{'type':'function','function':{'name':'get_weather'}}})]
    schema=dict(type='object',properties=dict(city=dict(type='string',enum=['東京','大阪']),count=dict(type='integer'),ok=dict(type='boolean')),required=['city','count','ok'],additionalProperties=False)
    structured=dict(messages=[dict(role='user',content='東京で3人が参加したことをJSONで示してください。')],response_format={'type':'json_schema','json_schema':{'name':'event','schema':schema}},max_tokens=128,temperature=.7,top_k=40,top_p=.95,seed=42)
    result+=[('json_greedy',{**structured,'temperature':0}),('json_sampled',structured),('length',{**basic,'max_tokens':3})]
    return result


def main():
    """同一経路・seed の一致、文法、SSE、MTP 使用統計を保存し、失敗時は非ゼロ終了する。"""
    parser=argparse.ArgumentParser();parser.add_argument('--server',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--modes',default='normal,mtp,greedy-only,full-normal,full-mtp');args=parser.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);api=load('m6','m6-api-check.py');http=load('m3','m3-api-check.py')
    inputs=cases(api);(out/'cases.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2)+'\n');baselines={};records=[]
    base='http://127.0.0.1:18123'
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);probe.bind(('127.0.0.1',18123))
    for mode in args.modes.split(','):
        command=[args.server,'-architecture','gemma4','-tokenizer-dir','./data/m7-api/tokenizer','-addr','127.0.0.1:18123','-max-context','8192']
        if mode not in ['normal','full-normal']:command+=['-mtp','./models/gemma-4-12b-it-qat/mtp-gemma-4-12B-it.gguf']
        if mode=='greedy-only':command+=['-mtp-greedy-only']
        if mode.startswith('full-'):command+=['-gpu-candidates=false']
        logpath=out/(mode+'.log')
        with logpath.open('w') as log:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
            try:
                deadline=time.monotonic()+90
                while True:
                    if child.poll() is not None:raise RuntimeError('server exited '+mode)
                    try:
                        status,h=http.get(base,'/health')
                        if status==200 and h.get('status')=='ok':break
                    except Exception:pass
                    if time.monotonic()>deadline:raise RuntimeError('health timeout')
                    time.sleep(.5)
                for name,body in inputs:
                    status,response=api.post(base,body)
                    if status!=200:raise RuntimeError(response)
                    msg=api.message_of(response);value=normalized(msg);key=('full' if mode.startswith('full-') else 'gpu',name)
                    if mode in ['normal','full-normal']:baselines[key]=value
                    same=value==baselines[key]
                    if not same:raise RuntimeError('output differs '+mode+'/'+name)
                    if name.startswith('tool_'):
                        calls=msg.get('tool_calls') or []
                        if not calls or not all(api.valid_call(c,body['tools']) for c in calls):raise RuntimeError('invalid tool call')
                    if name.startswith('json_') and not api.matches(json.loads(msg['content']),body['response_format']['json_schema']['schema']):raise RuntimeError('schema mismatch')
                    row=dict(mode=mode,case=name,same_output=same,response=response);records.append(row)
                    with (out/'requests.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
                    print(json.dumps(dict(mode=mode,case=name,same=same,tps=response['timings']['predicted_per_second'])),flush=True)
                    if mode=='mtp' and name in ['sampling','tool_sampled','json_sampled']:
                        streamed=api.stream(base,body)
                        ok=streamed['status']==200 and normalized(streamed)==value and streamed['finish_reason']==response['choices'][0]['finish_reason']
                        with (out/'sse.jsonl').open('a') as f:f.write(json.dumps(dict(case=name,ok=ok,response=streamed),ensure_ascii=False)+'\n')
                        if not ok:raise RuntimeError('SSE differs '+name)
                if mode=='mtp':
                    for script,extra in [('m7-api-check.py',[]),('m6-api-check.py',['--architecture','gemma4'])]:
                        with (out/(script+'.jsonl')).open('w') as f:subprocess.run(['python3',str(Path(__file__).with_name(script)),'--base',base]+extra,stdout=f,stderr=subprocess.STDOUT,check=True)
            finally:
                child.terminate()
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:child.kill();child.wait()
        metrics=[json.loads(l.split('generation_metrics=',1)[1]) for l in logpath.read_text().splitlines() if 'generation_metrics=' in l]
        (out/(mode+'-metrics.json')).write_text(json.dumps(metrics,indent=2)+'\n')
        if mode in ['mtp','full-mtp'] and not any(m.get('selection_driven') and m['batches']>0 for m in metrics):raise RuntimeError('selection MTP not exercised')
    (out/'summary.json').write_text(json.dumps([dict(mode=r['mode'],case=r['case'],same_output=r['same_output'],timings=r['response']['timings']) for r in records],indent=2)+'\n')


if __name__=='__main__':main()
