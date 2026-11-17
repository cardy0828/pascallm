"""Gemma 4 の API 統合を実機で確認する。標準ライブラリだけを使用する。
例: python3 tools/m7-api-check.py --base http://127.0.0.1:18121
本文・思考・長い生成後のキャッシュ復元・SSE・切断・同時要求・400応答を確認する。
"""
import argparse
import importlib.util
import json
from pathlib import Path
import threading


def main():
    """既存 M3 の HTTP ヘルパーを使い、Gemma の既定値と終了条件を確認する。"""
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',default='http://127.0.0.1:18121')
    parser.add_argument('--prefix-cache',action='store_true',help='M12のホスト保存点による復元を期待する')
    args=parser.parse_args()
    spec=importlib.util.spec_from_file_location('m3',Path(__file__).with_name('m3-api-check.py'))
    api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
    base=args.base
    post=lambda body:api.post(base,'/v1/chat/completions',body)
    question=[dict(role='user',content='日本の首都を一言で答えてください。')]
    body=dict(messages=question,temperature=0,seed=42,max_tokens=64)
    status,health=api.get(base,'/health');api.record('health',status==200 and health.get('status')=='ok')
    status,models=api.get(base,'/v1/models');api.record('models',status==200 and models['data'][0]['id']=='gemma4-12b-qat-pq4',body=models)
    status,response=post(body);message=api.message_of(response)
    api.record('japanese_default_no_thinking',status==200 and '東京' in (message.get('content') or '') and not message.get('reasoning_content'),response=response)
    thinking=dict(messages=[dict(role='user',content='120円のりんごを3個、200円の梨を2個買い、1000円払う。お釣りはいくら？')],temperature=0,max_tokens=768,chat_template_kwargs={'enable_thinking':True})
    status,response=post(thinking);message=api.message_of(response)
    api.record('thinking',status==200 and bool(message.get('reasoning_content')) and '240' in (message.get('content') or '') and not message.get('reasoning_content','').startswith('thought\n'),response=response)
    # 同一要求を再入力する場合、保存位置に戻るか最初から入れ直すため、seed が同じなら同じ出力となる。
    req=dict(messages=[dict(role='user',content='日本語で短い俳句を一つ。')],temperature=0.7,top_k=64,top_p=0.95,seed=7,max_tokens=96,stream_options={'include_usage':True})
    status,whole=post(req);streamed=api.stream(base,req)
    msg=api.message_of(whole)
    api.record('stream_matches_whole',status==200 and streamed['done'] and streamed['content']==(msg.get('content') or '') and streamed['reasoning']==msg.get('reasoning_content','') and all(streamed['usage'][key]==whole['usage'][key] for key in ('prompt_tokens','completion_tokens','total_tokens')),whole=whole,streamed=streamed)
    first=[dict(role='user',content='私の名前は太郎です。覚えてください。')]
    status,response=post(dict(messages=first,temperature=0,max_tokens=64))
    second=first+[api.message_of(response),dict(role='user',content='私の名前は？')]
    status,follow=post(dict(messages=second,temperature=0,max_tokens=64))
    api.record('multi_turn_cache',status==200 and follow['usage']['prompt_tokens_details']['cached_tokens']>0 and '太郎' in api.message_of(follow)['content'],response=follow)
    status,response=post(dict(messages=[dict(role='user',content='1から10までの数字を順番に書いて。')],temperature=0,max_tokens=64,stop=['5']))
    api.record('stop_string',status==200 and '5' not in api.message_of(response)['content'] and response['choices'][0]['finish_reason']=='stop',response=response)
    status,response=post(dict(messages=[dict(role='user',content='日本語で長い物語を書いて。')],temperature=0,max_tokens=5))
    api.record('max_tokens',status==200 and response['choices'][0]['finish_reason']=='length' and response['usage']['completion_tokens']==5,response=response)
    # GPUの保存位置が失効した後、通常は再入力、M12有効時はホストから復元する。本文も生成前と比較する。
    long_prompt=[dict(role='user',content='長い日本語の冒険小説を詳しく書いてください。')]
    schema={'type':'object','properties':{'values':{'type':'array','items':{'const':12345},'minItems':64,'maxItems':64},'more':{'type':'array','items':{'const':12345},'minItems':64,'maxItems':64},'end':{'type':'array','items':{'const':12345},'minItems':64,'maxItems':64}},'required':['values','more','end'],'additionalProperties':False}
    long_req=dict(messages=long_prompt,temperature=0,max_tokens=1150,response_format={'type':'json_schema','json_schema':{'name':'long','schema':schema}})
    before_status,before=post(dict(messages=long_prompt,temperature=0,max_tokens=16))
    status,long=post(long_req)
    status2,again=post(dict(messages=long_prompt,temperature=0,max_tokens=16))
    cached=again['usage']['prompt_tokens_details']['cached_tokens']
    expected_cache=cached>0 if args.prefix_cache else cached==0
    same=before_status==200 and status2==200 and api.message_of(before)==api.message_of(again)
    api.record('expired_snapshot_restores' if args.prefix_cache else 'expired_snapshot_recomputes',status==200 and long['usage']['completion_tokens']>1024 and status2==200 and expected_cache and same,long_usage=long['usage'],long_finish=long['choices'][0]['finish_reason'],next=again)
    cut=api.stream(base,thinking,stop_after_chunks=5)
    status,response=post(body)
    api.record('disconnect_then_next',cut.get('disconnected',False) and status==200,response=response)
    results=[]
    def worker():
        """同一APIへ要求し、順番待ち後の結果を保存する。"""
        results.append(post(body))
    threads=[threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:thread.start()
    for thread in threads:thread.join()
    api.record('concurrent_two',len(results)==2 and all(code==200 for code,_ in results))
    invalids={'empty':{'messages':[]},'image':{'messages':[dict(role='user',content=[dict(type='image_url',image_url={'url':'x'})])]},'n':{**body,'n':2},'context':{'messages':[dict(role='user',content='あいうえお'*20000)]},'orphan_tool':{'messages':[dict(role='tool',tool_call_id='bad',content='x')]}}
    for name,request in invalids.items():
        status,response=post(request);api.record('reject_'+name,status==400,error=response)
    print(json.dumps({'failures':api.FAILURES},ensure_ascii=False))
    raise SystemExit(bool(api.FAILURES))


if __name__=='__main__':main()
