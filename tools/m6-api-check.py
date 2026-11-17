"""M6 のツールコールと構造化出力を、起動中の API サーバー（cmd/pascallm-server）で確かめる一時ツール。

Python の標準ライブラリだけを使う。各確認の結果を 1 行 1 件の JSON で標準出力に書き、失敗があれば終了コード 1 で終わる。
  - 形の確認（ok に入れる）: 応答が API の形に合うか、文法で縛った出力が Schema・関数の定義に合うか、誤りの要求が 400 か
  - モデルの判断（ok に入れず model_ok として記録する）: tool_choice: auto で、ツールを呼ぶべき質問に呼んだか など

実行例:
    python3 tools/m6-api-check.py --base http://127.0.0.1:18120 > results/20261117-m6/api-check.jsonl
"""

import argparse
import http.client
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

FAILURES = []

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "指定した都市の現在の天気を返す",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "都市名。例: 東京"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}

TIME = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "指定したタイムゾーンの現在時刻を返す",
        "parameters": {
            "type": "object",
            "properties": {"timezone": {"type": "string"}, "offset_hours": {"type": "integer"}},
            "required": ["timezone", "offset_hours"],
        },
    },
}

TOOLS = [WEATHER, TIME]

PERSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "blood_type": {"type": "string", "enum": ["A", "B", "O", "AB"]},
        "hobbies": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
        "address": {
            "type": "object",
            "properties": {"prefecture": {"type": "string"}, "zip": {"type": ["string", "null"]}},
            "required": ["prefecture", "zip"],
            "additionalProperties": False,
        },
    },
    "required": ["name", "age", "blood_type", "hobbies", "address"],
    "additionalProperties": False,
}

NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}
THINKING = {"chat_template_kwargs": {"enable_thinking": True}}


def record(name: str, ok: bool, **details) -> None:
    """確認 1 件の結果を書く。例: record("tool_auto", True, calls=[...])"""
    print(json.dumps({"check": name, "ok": ok, **details}, ensure_ascii=False), flush=True)
    if not ok:
        FAILURES.append(name)


def post(base: str, body: dict, timeout: float = 600) -> tuple[int, dict]:
    """/v1/chat/completions に JSON を POST し、状態コードと JSON の応答を返す。"""
    request = urllib.request.Request(
        base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"{}")


def stream(base: str, body: dict) -> dict:
    """stream: true で POST し、増分をつなげた結果（本文・思考・ツールの呼び出し）を返す。"""
    parsed = urllib.parse.urlparse(base)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=600)
    connection.request("POST", "/v1/chat/completions", json.dumps({**body, "stream": True}), {"Content-Type": "application/json"})
    response = connection.getresponse()
    result = {"status": response.status, "content": "", "reasoning": "", "tool_calls": [], "finish_reason": None}
    for raw in response:
        line = raw.decode().strip()
        if not line.startswith("data: "):
            continue
        data = line[len("data: "):]
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
            result["content"] += delta.get("content") or ""
            result["reasoning"] += delta.get("reasoning_content") or ""
            result["tool_calls"].extend(delta.get("tool_calls") or [])
            if choice.get("finish_reason"):
                result["finish_reason"] = choice["finish_reason"]
    connection.close()
    return result


def message_of(body: dict) -> dict:
    """応答の message を返す。"""
    return body["choices"][0]["message"]


def calls_of(body: dict) -> list[dict]:
    """応答のツールの呼び出しを返す（なければ空）。"""
    return message_of(body).get("tool_calls") or []


def valid_call(call: dict, tools: list[dict]) -> bool:
    """呼び出しが関数の定義に合うか（名前があり、arguments が JSON で、required があり、型が合うか）を返す。"""
    if call.get("type") != "function" or not call.get("id", "").startswith("call_"):
        return False
    definitions = {tool["function"]["name"]: tool["function"] for tool in tools}
    function = definitions.get(call["function"]["name"])
    if function is None:
        return False
    try:
        arguments = json.loads(call["function"]["arguments"])
    except json.JSONDecodeError:
        return False
    return matches(arguments, function.get("parameters", {"type": "object"}))


def matches(value, schema: dict) -> bool:
    """値が Schema に合うかを確かめる（このスクリプトで使う範囲の簡単な検査）。"""
    if "enum" in schema and value not in schema["enum"]:
        return False
    types = schema.get("type")
    types = [types] if isinstance(types, str) else (types or [])
    checks = {
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
        "null": lambda v: v is None,
        "array": lambda v: isinstance(v, list),
        "object": lambda v: isinstance(v, dict),
    }
    if types and not any(checks[name](value) for name in types):
        return False
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        if any(key not in value for key in schema.get("required", [])):
            return False
        if schema.get("additionalProperties") is False and any(key not in properties for key in value):
            return False
        return all(matches(value[key], properties[key]) for key in value if key in properties)
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", len(value)):
            return False
        return all(matches(item, schema.get("items", {})) for item in value)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=["lfm3", "gemma4"], default="lfm3")
    parser.add_argument("--base", default="http://127.0.0.1:18120")
    args = parser.parse_args()
    base = args.base
    weather_question = [{"role": "user", "content": "東京の今の天気を教えて。"}]

    # 1. tool_choice: auto（既定）で、ツールを呼ぶべき質問。
    status, body = post(base, {"messages": weather_question, "tools": TOOLS, "seed": 1, **NO_THINKING})
    calls = calls_of(body) if status == 200 else []
    record("auto_shape", status == 200 and all(valid_call(call, TOOLS) for call in calls)
           and (body["choices"][0]["finish_reason"] == "tool_calls") == bool(calls),
           model_ok=bool(calls) and calls[0]["function"]["name"] == "get_weather",
           calls=calls, content=message_of(body).get("content") if status == 200 else body)

    # 2. 呼び出しの結果を返して、続きの答えを得る（往復）。
    if calls:
        history = weather_question + [
            {"role": "assistant", "content": message_of(body).get("content"), "tool_calls": calls},
            {"role": "tool", "tool_call_id": calls[0]["id"], "content": json.dumps({"天気": "晴れ", "気温": 21}, ensure_ascii=False)},
        ]
        status, answer = post(base, {"messages": history, "tools": TOOLS, "seed": 1, **NO_THINKING})
        content = message_of(answer).get("content") if status == 200 else answer
        record("round_trip", status == 200 and answer["choices"][0]["finish_reason"] == "stop" and bool(content),
               model_ok=isinstance(content, str) and "晴" in content, content=content,
               cached_tokens=answer.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens") if status == 200 else None)
    else:
        record("round_trip", False, reason="auto で呼び出しがなかったため確かめられない")

    # 3. ストリームでも同じ呼び出しになるか（seed 固定）。
    streamed = stream(base, {"messages": weather_question, "tools": TOOLS, "seed": 1, **NO_THINKING})
    same = [(c["function"]["name"], c["function"]["arguments"]) for c in streamed["tool_calls"]] == \
           [(c["function"]["name"], c["function"]["arguments"]) for c in calls]
    record("stream_matches_whole", streamed["status"] == 200 and same and all("index" in c for c in streamed["tool_calls"])
           and streamed["finish_reason"] == body["choices"][0]["finish_reason"], stream_calls=streamed["tool_calls"])

    # 4. 複数の呼び出し（並列）と parallel_tool_calls: false。
    both = [{"role": "user", "content": "東京と大阪の天気をそれぞれ調べて。"}]
    status, body = post(base, {"messages": both, "tools": TOOLS, "seed": 2, **NO_THINKING})
    calls = calls_of(body) if status == 200 else []
    record("parallel_shape", status == 200 and all(valid_call(call, TOOLS) for call in calls),
           model_ok=len(calls) >= 2, count=len(calls), calls=calls)
    status, body = post(base, {"messages": both, "tools": TOOLS, "seed": 2, "parallel_tool_calls": False, **NO_THINKING})
    record("parallel_false", status == 200 and len(calls_of(body)) <= 1, count=len(calls_of(body)) if status == 200 else body)

    # 5. tool_choice: none なら呼ばない。
    status, body = post(base, {"messages": weather_question, "tools": TOOLS, "tool_choice": "none", "seed": 1, **NO_THINKING})
    record("choice_none", status == 200 and not calls_of(body) and bool(message_of(body).get("content")),
           content=message_of(body).get("content") if status == 200 else body)

    # 6. tool_choice: required なら、関係ない質問でも定義に合う呼び出しをする（思考あり・なし）。
    greeting = [{"role": "user", "content": "こんにちは！"}]
    for label, kwargs in [("required", NO_THINKING), ("required_thinking", THINKING)]:
        status, body = post(base, {"messages": greeting, "tools": TOOLS, "tool_choice": "required", "seed": 3, "max_tokens": 1500, **kwargs})
        calls = calls_of(body) if status == 200 else []
        record("choice_" + label, status == 200 and len(calls) >= 1 and all(valid_call(call, TOOLS) for call in calls)
               and body["choices"][0]["finish_reason"] == "tool_calls",
               calls=calls, reasoning_head=(message_of(body).get("reasoning_content") or "")[:100] if status == 200 else body)

    # 7. 関数を指定したら、その関数を 1 件だけ呼ぶ。
    named = {"type": "function", "function": {"name": "get_time"}}
    status, body = post(base, {"messages": weather_question, "tools": TOOLS, "tool_choice": named, "seed": 4, **NO_THINKING})
    calls = calls_of(body) if status == 200 else []
    record("choice_named", status == 200 and len(calls) == 1 and calls[0]["function"]["name"] == "get_time"
           and valid_call(calls[0], TOOLS), calls=calls)

    # 8. response_format: json_object。
    ask_json = [{"role": "user", "content": "日本の都道府県を3つ、名前と県庁所在地を JSON で答えて。"}]
    status, body = post(base, {"messages": ask_json, "response_format": {"type": "json_object"}, "seed": 5, **NO_THINKING})
    content = message_of(body).get("content") if status == 200 else ""
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        parsed = None
    record("json_object", status == 200 and isinstance(parsed, dict), content=content)

    # 9. response_format: json_schema（思考あり・なし、ストリーム）。
    ask_person = [{"role": "user", "content": "架空の人物を1人作って、プロフィールを答えて。"}]
    response_format = {"type": "json_schema", "json_schema": {"name": "person", "schema": PERSON_SCHEMA, "strict": True}}
    for label, kwargs, use_stream in [("json_schema", NO_THINKING, False), ("json_schema_thinking", THINKING, False),
                                      ("json_schema_stream", NO_THINKING, True)]:
        request = {"messages": ask_person, "response_format": response_format, "seed": 6, "max_tokens": 1500, **kwargs}
        if use_stream:
            result = stream(base, request)
            status, content = result["status"], result["content"]
        else:
            status, body = post(base, request)
            content = message_of(body).get("content") if status == 200 else ""
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        record(label, status == 200 and parsed is not None and matches(parsed, PERSON_SCHEMA), content=content)

    # 10. 文法で縛ったときの生成速度（長めの JSON の配列）と、縛らないときの比較。
    list_schema = {"type": "object", "properties": {"items": {"type": "array", "items": {
        "type": "object", "properties": {"name": {"type": "string"}, "score": {"type": "integer"}}, "required": ["name", "score"]},
        "minItems": 20, "maxItems": 20}}, "required": ["items"]}
    ask_list = [{"role": "user", "content": "日本の有名な食べ物を20個、それぞれ0〜100の人気スコアを付けて JSON で答えて。"}]
    timings = {}
    for label, extra in [("constrained", {"response_format": {"type": "json_schema", "json_schema": {"name": "list", "schema": list_schema}}}),
                         ("free", {})]:
        start = time.time()
        status, body = post(base, {"messages": ask_list, "seed": 7, "max_tokens": 1200, **extra, **NO_THINKING})
        timings[label] = {"status": status, "wall_seconds": round(time.time() - start, 3),
                          "completion_tokens": body.get("usage", {}).get("completion_tokens"),
                          "predicted_per_second": body.get("timings", {}).get("predicted_per_second")}
        if label == "constrained":
            try:
                parsed = json.loads(message_of(body).get("content") or "")
            except json.JSONDecodeError:
                parsed = None
            timings[label]["valid"] = parsed is not None and matches(parsed, list_schema) and len(parsed["items"]) == 20
    record("speed", timings["constrained"]["status"] == 200 and timings["constrained"].get("valid", False), **timings)

    # 11. 誤りの要求は 400 になるか。
    bad_history = weather_question + [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "x", "type": "function", "function": {"name": "get_weather", "arguments": "{not json"}}]}]
    for name, extra in [
        ("required_without_tools", {"tool_choice": "required"}),
        ("named_unknown", {"tools": TOOLS, "tool_choice": {"type": "function", "function": {"name": "nope"}}}),
        ("bad_tools", {"tools": [{"type": "function", "function": {"description": "名前なし"}}]}),
        ("bad_schema", {"response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "decimal"}}}}),
        ("bad_arguments", {"messages": bad_history, "tools": TOOLS}),
        ("unknown_reference", {"tools": TOOLS, "messages": weather_question + [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "r", "type": "function", "function": {"name": "get_time", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "r", "content": [{"type": "tool_reference", "name": "missing"}]}]}),
        ("developer_role", {"messages": [{"role": "developer", "content": "x"}]}),
    ]:
        status, body = post(base, {"messages": weather_question, **extra})
        expected = 200 if name == "developer_role" and args.architecture == "gemma4" else 400
        record(("accept_" if expected == 200 else "reject_") + name, status == expected, status=status, error=body.get("error", {}).get("message"))

    print(json.dumps({"failures": FAILURES}, ensure_ascii=False))
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
