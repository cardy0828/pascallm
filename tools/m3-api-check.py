"""M3 の API サーバー（cmd/pascallm-server）を確かめる一時ツール。Python の標準ライブラリだけを使う。

起動中のサーバーに要求を送り、各確認の結果を 1 行 1 件の JSON で標準出力に書く。
失敗した確認があれば終了コード 1 で終わる。生成された本文も記録する（日本語として読めるかは人が確かめる）。

実行例:
    python3 tools/m3-api-check.py --base http://127.0.0.1:18120 > results/20261116-m3/api-check.jsonl
"""

import argparse
import http.client
import json
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

FAILURES = []


def record(name: str, ok: bool, **details) -> None:
    """確認 1 件の結果を書く。例: record("health", True, status=200)"""
    print(json.dumps({"check": name, "ok": ok, **details}, ensure_ascii=False), flush=True)
    if not ok:
        FAILURES.append(name)


def post(base: str, path: str, body: dict, timeout: float = 600) -> tuple[int, dict]:
    """JSON を POST し、状態コードと JSON の応答を返す。"""
    request = urllib.request.Request(
        base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"{}")


def get(base: str, path: str) -> tuple[int, dict]:
    """GET し、状態コードと JSON の応答を返す。"""
    with urllib.request.urlopen(base + path, timeout=30) as response:
        return response.status, json.loads(response.read())


def stream(base: str, body: dict, stop_after_chunks: int | None = None) -> dict:
    """stream: true で POST し、増分をつなげた結果を返す。stop_after_chunks 個で接続を切ることもできる。"""
    parsed = urllib.parse.urlparse(base)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=600)
    connection.request("POST", "/v1/chat/completions", json.dumps({**body, "stream": True}),
                       {"Content-Type": "application/json"})
    response = connection.getresponse()
    result = {"status": response.status, "content": "", "reasoning": "", "chunks": 0, "finish_reason": None,
              "usage": None, "timings": None, "done": False}
    for raw in response:
        line = raw.decode().strip()
        if not line.startswith("data: "):
            continue
        data = line[len("data: "):]
        if data == "[DONE]":
            result["done"] = True
            break
        chunk = json.loads(data)
        result["chunks"] += 1
        if chunk.get("usage"):
            result["usage"] = chunk["usage"]
        if chunk.get("timings"):
            result["timings"] = chunk["timings"]
        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
            result["content"] += delta.get("content") or ""
            result["reasoning"] += delta.get("reasoning_content") or ""
            if choice.get("finish_reason"):
                result["finish_reason"] = choice["finish_reason"]
        if stop_after_chunks is not None and result["chunks"] >= stop_after_chunks:
            connection.close()
            result["disconnected"] = True
            break
    return result


def chat(base: str, messages: list, **options) -> tuple[int, dict]:
    """まとめて返す形で対話を 1 回送る。"""
    return post(base, "/v1/chat/completions", {"model": "lfm3-1.7b-pq4", "messages": messages, **options})


def message_of(response: dict) -> dict:
    """応答の assistant の発言を返す。"""
    return response["choices"][0]["message"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:18120")
    args = parser.parse_args()
    base = args.base
    no_thinking = {"chat_template_kwargs": {"enable_thinking": False}}

    status, body = get(base, "/health")
    record("health", status == 200 and body.get("status") == "ok", status=status)
    status, body = get(base, "/v1/models")
    record("models", status == 200 and body["data"][0]["id"] == "lfm3-1.7b-pq4", body=body)

    # 日本語の対話（思考なし・あり）。既定の選び方（generation_config.json）で、seed を固定する。
    question = [{"role": "user", "content": "日本の首都はどこですか？一文で答えてください。"}]
    status, body = chat(base, question, seed=42, max_tokens=128, **no_thinking)
    message = message_of(body)
    record("japanese_no_thinking", status == 200 and "東京" in message["content"] and not message.get("reasoning_content"),
           content=message["content"], usage=body["usage"], timings=body["timings"])

    arithmetic = [{"role": "user", "content": "1個120円のりんごを3個と、1個200円のパンを2個買い、1000円札で払いました。おつりはいくらですか？"}]
    # 合否は API の動き（思考と本文が分かれて返り、stop で終わる）で決める。答えが正しいか（おつり 240 円）は
    # モデルの品質なので answer_has_240 に記録するだけにする（seed ごとの正答は arithmetic-seeds.jsonl）。
    status, body = chat(base, arithmetic, seed=42, max_tokens=1024)
    message = message_of(body)
    record("japanese_thinking", status == 200 and bool(message.get("reasoning_content")) and bool(message["content"])
           and body["choices"][0]["finish_reason"] == "stop" and "</think>" not in message["content"],
           answer_has_240="240" in message["content"],
           reasoning=message.get("reasoning_content"), content=message["content"],
           finish_reason=body["choices"][0]["finish_reason"], usage=body["usage"], timings=body["timings"])

    # ストリームとまとめた応答が同じ結果になるか（同じ seed・同じプロンプト）。
    request = {"messages": arithmetic, "seed": 7, "max_tokens": 300, "temperature": 0.7, "top_p": 0.9}
    status, whole = post(base, "/v1/chat/completions", request)
    streamed = stream(base, {**request, "stream_options": {"include_usage": True}})
    whole_message = message_of(whole)
    same = (streamed["content"] == whole_message["content"]
            and streamed["reasoning"] == (whole_message.get("reasoning_content") or "")
            and streamed["finish_reason"] == whole["choices"][0]["finish_reason"]
            and streamed["usage"]["completion_tokens"] == whole["usage"]["completion_tokens"])
    record("stream_matches_whole", status == 200 and streamed["done"] and same,
           stream_chunks=streamed["chunks"], completion_tokens=whole["usage"]["completion_tokens"],
           stream_usage=streamed["usage"])

    # 続きの会話で KV キャッシュを再利用するか（思考なしなら、前の応答がそのままプロンプトの続きになる）。
    first = [{"role": "user", "content": "好きな果物を一つ挙げて、理由を一文で書いてください。"}]
    status, body = chat(base, first, seed=1, max_tokens=100, **no_thinking)
    reply = message_of(body)["content"]
    second = first + [{"role": "assistant", "content": reply}, {"role": "user", "content": "その果物の旬はいつですか？"}]
    status2, body2 = chat(base, second, seed=1, max_tokens=100, **no_thinking)
    cached = body2["usage"]["prompt_tokens_details"]["cached_tokens"]
    record("multi_turn_cache", status == 200 and status2 == 200 and cached > 0,
           first_reply=reply, second_reply=message_of(body2)["content"], usage=body2["usage"], timings=body2["timings"])

    # stop の文字列で止まるか。
    status, body = chat(base, [{"role": "user", "content": "1から10までの数字を、読点で区切って並べてください。"}],
                        seed=3, max_tokens=100, stop=["5"], temperature=0, **no_thinking)
    content = message_of(body)["content"]
    record("stop_string", status == 200 and "5" not in content and body["choices"][0]["finish_reason"] == "stop",
           content=content)

    # max_tokens で止まるか。
    status, body = chat(base, question, max_tokens=5, **no_thinking)
    record("max_tokens", status == 200 and body["choices"][0]["finish_reason"] == "length" and body["usage"]["completion_tokens"] == 5,
           usage=body["usage"])

    # 誤りの要求は 400 になるか。
    # ツールは M6 で扱うようになったため、tools なしの tool_choice と、扱わない役割を確かめる。
    for name, extra in [("tool_choice_without_tools", {"tool_choice": "required"}), ("n2", {"n": 2}), ("bad_temperature", {"temperature": 5}),
                        ("developer_role", {"messages": [{"role": "developer", "content": "x"}]}),
                        ("image", {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]}),
                        ("too_long", {"messages": [{"role": "user", "content": " ".join(str(i) for i in range(5000))}]})]:
        status, body = post(base, "/v1/chat/completions", {"messages": question, **extra})
        record("reject_" + name, status == 400, status=status, error=body.get("error", {}).get("message"))

    # 生成中に接続を切っても、サーバーが次の要求を処理できるか。
    cut = stream(base, {"messages": arithmetic, "max_tokens": 2000}, stop_after_chunks=5)
    status, body = chat(base, question, seed=42, max_tokens=32, **no_thinking)
    record("disconnect_then_next", cut.get("disconnected", False) and status == 200, after=message_of(body)["content"])

    # 2 件を同時に送っても、順に処理されて両方返るか。
    results = []

    def send() -> None:
        results.append(chat(base, question, seed=5, max_tokens=64, **no_thinking))

    threads = [threading.Thread(target=send) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    record("concurrent_two", len(results) == 2 and all(status == 200 for status, _ in results),
           contents=[message_of(body)["content"] for _, body in results])

    # API を通した生成速度。長い文章を頼み、(1) 既定の選び方（logits を CPU に移す）と
    # (2) temperature 0・ペナルティなし（GPU の argmax だけ）を比べる。
    essay = [{"role": "user", "content": "日本の四季について、それぞれの季節の行事や食べ物を交えて、できるだけ詳しく長い文章で説明してください。"}]
    for name, options in [("speed_default_sampling", {"seed": 11}),
                          ("speed_greedy_gpu", {"temperature": 0, "repetition_penalty": 1.0})]:
        started = time.perf_counter()
        status, body = chat(base, essay, max_tokens=512, **no_thinking, **options)
        elapsed = time.perf_counter() - started
        record(name, status == 200, wall_seconds=round(elapsed, 3), usage=body["usage"], timings=body["timings"],
               content_head=message_of(body)["content"][:200])

    print(json.dumps({"failures": FAILURES}, ensure_ascii=False))
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
