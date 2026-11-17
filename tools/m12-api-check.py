"""M12: 会話切替・共有prefix・M11機能を cache 有無で比較する実機検査。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import time

def load():
    """既存M11の要求・応答正規化ヘルパーを読む。"""
    spec=importlib.util.spec_from_file_location("m11",Path(__file__).with_name("m11-api-check.py"))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    """独立した検証サーバーを起動し、同じseedの結果と転送を含む入力時間を記録する。"""
    parser=argparse.ArgumentParser()
    parser.add_argument("--server",required=True)
    parser.add_argument("--out",required=True)
    parser.add_argument("--queue-only",action="store_true")
    args=parser.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    m11=load();api=m11.load("m6","m6-api-check.py");http=m11.load("m3","m3-api-check.py")
    common="次の記録を読み、最後の質問だけに日本語で答えてください。\n"+"資料：図書館は午前九時から午後六時まで開いています。月曜は休館です。\n"*24
    base_request=dict(max_tokens=32,temperature=.7,top_k=40,top_p=.95,seed=42)
    a={**base_request,"messages":[{"role":"system","content":common},{"role":"user","content":"図書館は何時に開きますか。"}]}
    b={**base_request,"messages":[{"role":"system","content":"別の会話です。短く答えてください。"},{"role":"user","content":"富士山について一文で説明してください。"}]}
    branch={**base_request,"messages":[{"role":"system","content":common},{"role":"user","content":"図書館の休館日はいつですか。"}]}
    requests=[("A1",a),("B1",b),("A2",a),("shared_branch",branch),("B2",b)]
    requests+=m11.cases(api)
    if args.queue_only:requests=[("A",a),("B",b),("branch",branch)]
    (out/"cases.json").write_text(json.dumps(requests,ensure_ascii=False,indent=2)+"\n")
    baselines={};rows=[]
    base="http://127.0.0.1:18123"
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);probe.bind(("127.0.0.1",18123))
    for mode,budget in ([("on",1024)] if args.queue_only else [("off",0),("on",1024)]):
        command=[args.server,"-architecture","gemma4","-tokenizer-dir","./data/m7-api/tokenizer","-max-context","8192","-addr","127.0.0.1:18123","-mtp","./models/gemma-4-12b-it-qat/mtp-gemma-4-12B-it.gguf","-prefix-cache-mib",str(budget)]
        with (out/(mode+".log")).open("w") as log:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=log)
            try:
                deadline=time.monotonic()+90
                while True:
                    if child.poll() is not None:raise RuntimeError("server exited")
                    try:
                        status,h=http.get(base,"/health")
                        if status==200:break
                    except Exception:pass
                    if time.monotonic()>deadline:raise RuntimeError("health timeout")
                    time.sleep(.5)
                for name,body in requests:
                    start=time.monotonic()
                    status,response=api.post(base,body)
                    if status!=200:raise RuntimeError(response)
                    elapsed=(time.monotonic()-start)*1000
                    value=m11.normalized(api.message_of(response))
                    if mode=="off" or args.queue_only:baselines[name]=value
                    assert value==baselines[name],(mode,name,"output differs")
                    if mode=="on" and name in ["A2","B2","shared_branch"]:assert response["timings"]["cache_n"]>0
                    row=dict(mode=mode,case=name,same_output=True,elapsed_ms=elapsed,response=response)
                    rows.append(row)
                    with (out/"requests.jsonl").open("a") as f:f.write(json.dumps(row,ensure_ascii=False)+"\n")
                    print(json.dumps(dict(mode=mode,case=name,timings=response["timings"])),flush=True)
                    if mode=="on" and name in ["sampling","tool_sampled","json_sampled"]:
                        streamed=api.stream(base,body)
                        assert streamed["status"]==200 and m11.normalized(streamed)==value
                if args.queue_only:
                    with ThreadPoolExecutor(max_workers=3) as pool:
                        responses=list(pool.map(lambda pair:api.post(base,pair[1]),requests))
                    concurrent=[]
                    for (name,body),(status,response) in zip(requests,responses):
                        assert status==200 and m11.normalized(api.message_of(response))==baselines[name]
                        concurrent.append(dict(case=name,response=response))
                    assert max(r["response"]["timings"]["queue_ms"] for r in concurrent)>1
                    (out/"concurrent.json").write_text(json.dumps(concurrent,ensure_ascii=False,indent=2)+"\n")
                if mode=="on" and not args.queue_only:
                    for script,extra in [("m7-api-check.py",["--prefix-cache"]),("m6-api-check.py",["--architecture","gemma4"])]:
                        with (out/(script+".jsonl")).open("w") as f:
                            subprocess.run(["python3",str(Path(__file__).with_name(script)),"--base",base]+extra,stdout=f,stderr=subprocess.STDOUT,check=True)
            finally:
                child.terminate()
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:child.kill();child.wait()
    (out/"summary.json").write_text(json.dumps([dict(mode=r["mode"],case=r["case"],elapsed_ms=r["elapsed_ms"],timings=r["response"]["timings"]) for r in rows],indent=2)+"\n")

if __name__=="__main__":main()
