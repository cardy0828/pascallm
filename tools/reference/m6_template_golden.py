"""Go のチャットテンプレートのツールの部分（M6）を確かめる基準を、Hugging Face の実装で作る一時ツール。

出力は 1 行 1 件の JSON で、cmd/tokenizer-check が読む（tokenizer_golden.py の chat と同じ形に tools を足したもの）。
  {"kind": "chat", "messages": [...], "tools": [...], "enable_thinking": bool, "rendered": ..., "ids": [...]}

messages は OpenAI の API の形（tool_calls の arguments は JSON の文字列）のまま書く。
Hugging Face のテンプレートには、vLLM と同じく arguments を json.loads した辞書を渡す（テンプレートが文字列を断るため）。
Go 側は API サーバーと同じ変換（internal/server の RenderChat）を通して比べる。

既存の venv を読み取り専用で使い、パッケージの追加・更新はしない。

実行例:
    HF_HUB_OFFLINE=1 python3 -I tools/reference/m6_template_golden.py \
        --model-dir ./models/lfm3-1.7b-training --out ./data/m6/template-golden.jsonl
"""

import argparse
import copy
import json

from transformers import AutoTokenizer

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "指定した都市の天気を返す。\t単位は摂氏 °C 🌤",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "都市名（例: 東京）"},
                "days": {"type": "integer", "minimum": 1, "default": 1},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}

SEARCH = {
    "type": "function",
    "function": {
        "name": "search_docs",
        "description": "Search documents. Ratio 1.50, tiny 1e-05, big 1E+20, zero -0, float 100.0",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": ["integer", "null"], "default": None},
                "filters": {
                    "type": "object",
                    "properties": {"lang": {"type": "string"}, "strict": {"type": "boolean", "default": False}},
                },
                "tags": {"type": "array", "items": {"type": "string"}},
                "score": {"type": "number", "default": 0.5, "minimum": -0.0, "maximum": 1e20, "multipleOf": 1e-05, "big": 12345678901234567890},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

NO_PARAMS = {"type": "function", "function": {"name": "now", "description": "現在時刻", "parameters": {"type": "object", "properties": {}}}}

DEFERRED = {
    "type": "function",
    "defer_loading": True,
    "function": {"name": "send_mail", "description": "メールを送る", "parameters": {"type": "object", "properties": {"to": {"type": "string"}}}},
}

DEFERRED_INNER = {
    "type": "function",
    "function": {"name": "translate", "defer_loading": True, "parameters": {"type": "object", "properties": {"text": {"type": "string"}}}},
}

SEARCH_TOOLS = {
    "type": "function",
    "function": {"name": "search_tools", "description": "ツールを探す", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}},
}


def call(call_id, name, arguments):
    """OpenAI の形のツールの呼び出し 1 件を作る。arguments は JSON の文字列。"""
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


WEATHER_ARGS = json.dumps({"city": "東京\n(港区)", "days": 3, "unit": "celsius"}, ensure_ascii=False)
SEARCH_ARGS = json.dumps(
    {"query": "a \"quoted\" <tag> & \\ back", "limit": None, "filters": {"lang": "ja", "strict": True}, "tags": ["x", "日本"], "score": 1.50},
    ensure_ascii=False,
)

# (messages, tools) の組。思考の有無の両方で出す。
CASES = [
    ([{"role": "user", "content": "東京の天気は？"}], [WEATHER]),
    ([{"role": "system", "content": "あなたは助手です。"}, {"role": "user", "content": "検索して"}], [WEATHER, SEARCH, NO_PARAMS]),
    ([{"role": "system", "content": ""}, {"role": "user", "content": "空の system と tools"}], [SEARCH]),
    (
        [
            {"role": "user", "content": "東京の天気"},
            {"role": "assistant", "content": None, "tool_calls": [call("call_1", "get_weather", WEATHER_ARGS)]},
            {"role": "tool", "tool_call_id": "call_1", "content": "晴れ、25°C"},
        ],
        [WEATHER],
    ),
    (
        [
            {"role": "user", "content": "東京の天気"},
            {"role": "assistant", "content": "調べます。", "reasoning_content": "天気の関数を呼ぶ。", "tool_calls": [call("call_1", "get_weather", WEATHER_ARGS)]},
            {"role": "tool", "tool_call_id": "call_1", "content": "晴れ"},
            {"role": "assistant", "content": "東京は晴れです。"},
            {"role": "user", "content": "ありがとう。次は検索して"},
        ],
        [WEATHER, SEARCH],
    ),
    (
        [
            {"role": "user", "content": "並列で呼んで"},
            {"role": "assistant", "content": "", "tool_calls": [call("a", "get_weather", WEATHER_ARGS), call("b", "search_docs", SEARCH_ARGS), call("c", "now", "{}")]},
            {"role": "tool", "tool_call_id": "a", "content": "晴れ"},
            {"role": "tool", "tool_call_id": "b", "content": [{"type": "text", "text": "結果1"}, {"type": "text", "text": "結果2"}]},
            {"role": "tool", "tool_call_id": "c", "content": None},
        ],
        [WEATHER, SEARCH, NO_PARAMS],
    ),
    (
        [
            {"role": "user", "content": "同じ id の応答をまとめる"},
            {"role": "assistant", "content": None, "tool_calls": [call("same", "now", "")]},
            {"role": "tool", "tool_call_id": "same", "content": "前半"},
            {"role": "tool", "tool_call_id": "same", "content": ""},
            {"role": "tool", "tool_call_id": "same", "content": [{"type": "text", "text": "後半"}, {"k": 1, "v": [1.0, None]}]},
        ],
        [NO_PARAMS],
    ),
    (
        [
            {"role": "user", "content": "応答の文字列にタグ"},
            {"role": "assistant", "content": None, "tool_calls": [call("x", "now", "{}")]},
            {"role": "tool", "tool_call_id": "x", "content": "<tool_response>\n自前で包んだ\n</tool_response>"},
        ],
        [NO_PARAMS],
    ),
    (
        [
            {"role": "system", "content": "必要ならツールを探す。"},
            {"role": "user", "content": "メールを送って"},
            {"role": "assistant", "content": None, "tool_calls": [call("s1", "search_tools", json.dumps({"q": "mail"}))]},
            {
                "role": "tool",
                "tool_call_id": "s1",
                "content": [
                    {"type": "text", "text": "2 件見つかった。"},
                    {"type": "tool_reference", "name": "send_mail"},
                    {"type": "tool_reference", "name": "translate"},
                    {"type": "text", "text": "以上"},
                ],
            },
        ],
        [SEARCH_TOOLS, DEFERRED, DEFERRED_INNER],
    ),
    (
        [
            {"role": "user", "content": "追加のツール"},
            {"role": "assistant", "content": None, "tool_calls": [call("s2", "search_tools", "{\"q\": \"x\"}")]},
            {"role": "tool", "tool_call_id": "s2", "content": "ignored", "additional_tools": [DEFERRED, DEFERRED_INNER]},
            {"role": "tool", "tool_call_id": "s3", "content": "<additional_tools>\n{}\n</additional_tools>", "additional_tools": True},
            {"role": "tool", "tool_call_id": "s4", "content": [{"result": {"tools": [NO_PARAMS]}}, {"type": "text", "text": "後ろ"}]},
        ],
        [SEARCH_TOOLS, DEFERRED, DEFERRED_INNER, NO_PARAMS],
    ),
    (
        [
            {"role": "user", "content": "最初の質問"},
            {"role": "assistant", "content": "<think>古い思考</think>\n\n古い答え", "tool_calls": [call("o", "get_weather", "{\"city\": \"大阪\"}")]},
            {"role": "tool", "tool_call_id": "o", "content": "雨"},
            {"role": "assistant", "content": "大阪は雨です。"},
            {"role": "user", "content": "次の質問"},
        ],
        [WEATHER],
    ),
    (
        [
            {"role": "user", "content": "最後が呼び出し"},
            {"role": "assistant", "content": "呼びます", "tool_calls": [call("z", "search_docs", "{\"query\": \"q\", \"tags\": [], \"filters\": {}, \"score\": 2e3}")]},
        ],
        [SEARCH],
    ),
    ([{"role": "user", "content": "tools が空"}], []),
]


def to_hf(messages):
    """OpenAI の形の messages を、Hugging Face のテンプレートに渡す形にする（arguments を辞書にする）。"""
    converted = copy.deepcopy(messages)
    for message in converted:
        for tool_call in message.get("tool_calls") or []:
            arguments = tool_call["function"].get("arguments")
            if isinstance(arguments, str):
                tool_call["function"]["arguments"] = json.loads(arguments) if arguments.strip() else {}
    return converted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    count = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for messages, tools in CASES:
            for thinking in (True, False):
                rendered = tokenizer.apply_chat_template(
                    to_hf(messages), tools=tools or None, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
                )
                ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
                record = {"kind": "chat", "messages": messages, "tools": tools, "enable_thinking": thinking, "rendered": rendered, "ids": ids}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    print(json.dumps({"chat": count}))


if __name__ == "__main__":
    main()
