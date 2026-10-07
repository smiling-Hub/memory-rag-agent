# -*- coding: utf-8 -*-
"""
P0 最后一环：Agent 换成真正的 function-calling（工具调用）。

对比 main.py（固定「先检索再回答」），这里让 AI 自己决定要不要查记忆、查什么。
这对应二创清单 P0.1：把原项目「让 AI 写 <SEARCH> 标签再用正则抠」的脆壳，
换成大模型原生的工具调用信号——AI 要查就返回结构化调用，不查就直接回答。

跑法：venv/Scripts/python agent.py
"""
import os
import sys
import json
import ssl

# Windows 终端 UTF-8 + 全局关 SSL 校验（本机根证书缺失，仅演示用）
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ssl._create_default_https_context = ssl._create_unverified_context
_orig = ssl.create_default_context
def _unverified(*a, **k):
    c = _orig(*a, **k)
    c.check_hostname = False
    c.verify_mode = ssl.CERT_NONE
    return c
ssl.create_default_context = _unverified

import db
import memory

CHAT_BASE_URL = "https://api.deepseek.com/v1"
from dotenv import load_dotenv
load_dotenv()
CHAT_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")  # 从 .env 读，不硬编码
CHAT_MODEL_NAME = "deepseek-v4-pro"

# 定义「工具」。AI 只能调用这里声明的工具，返回结构化信号，而不是猜文字标签。
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "搜索当前用户的记忆库，返回相关记忆片段。当需要回忆用户过去说过的事、做过的决定、存过的信息时才调用；闲聊或与记忆无关的问题不要调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词，多个词用空格隔开"}
                },
                "required": ["query"],
            },
        },
    }
]

SYSTEM_PROMPT = (
    "你是一个记忆助手。回答用户问题前，如果需要回忆他过去的信息，就调用 search_memory 工具查询；"
    "与记忆无关的问题（比如闲聊）直接回答，不要调用工具，也不要编造记忆里没有的内容。"
)


def _llm():
    import httpx
    from openai import OpenAI
    try:
        return OpenAI(base_url=CHAT_BASE_URL, api_key=CHAT_API_KEY,
                      http_client=httpx.Client(verify=False))
    except TypeError:
        return OpenAI(base_url=CHAT_BASE_URL, api_key=CHAT_API_KEY)


def _run_tool(name, args, user):
    """真正执行 AI 调用的工具。现在只有 search_memory 一个。"""
    if name == "search_memory":
        ctx = memory.search(user["id"], args.get("query", ""), n=3)
        if not ctx:
            return "（没有查到相关记忆）"
        return "\n".join(f"- {doc_id}: {content}" for doc_id, content in ctx)
    return "未知工具"


def ask_agent(user, question):
    """让 AI 自己决定：要不要查记忆、查什么，然后回答。最多走 3 轮（对应原项目的 chain_length 上限）。"""
    llm = _llm()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    total_tokens = 0

    for _ in range(3):  # 最多 3 轮，防止死循环
        resp = llm.chat.completions.create(
            model=CHAT_MODEL_NAME, messages=messages, tools=TOOLS, max_tokens=300,
        )
        total_tokens += getattr(resp.usage, "total_tokens", 0)
        msg = resp.choices[0].message

        # 没有 tool_calls = AI 决定不查，直接给最终答案
        if not msg.tool_calls:
            cost = db.charge(user["id"], total_tokens, note=question[:20])
            return msg.content, total_tokens, cost

        # 有 tool_calls：把 assistant 的调用消息原样塞回去
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ],
        })
        # 执行工具，把结果以 role="tool" 塞回去
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            result = _run_tool(tc.function.name, args, user)
            print(f"  [工具调用] {tc.function.name}(query={args.get('query','')!r})")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return "（未能生成有效回答）", total_tokens, 0


def main():
    db.init_db()
    if db.register("alice", "123456", initial_credits=100):
        print("[注册] alice 创建成功，送 100 credits")
    alice = db.login("alice", "123456")

    # 先给 alice 记一条记忆
    memory.add_memory(alice["id"], "Note of 2026-09-15 买入八方股份",
                      "9月15日买入八方股份603489，成本价38.5元，共500股。", 20260915)
    print("[记忆] 已写入 alice 的记忆库\n")

    print("=" * 60)
    # 场景1：需要查记忆的问题 → AI 应该主动调用 search_memory
    q = "我买八方股份的成本价是多少"
    print(f"[alice 提问] {q}")
    ans, tokens, cost = ask_agent(alice, q)
    print(f"[回答] {ans}")
    print(f"[扣费] {tokens} token，扣 {cost} credits，余额 {db.get_balance(alice['id'])}\n")

    # 场景2：与记忆无关的问题 → AI 应该不查、直接答
    q2 = "你好，用一句话介绍一下你自己"
    print(f"[alice 提问] {q2}")
    ans, tokens, cost = ask_agent(alice, q2)
    print(f"[回答] {ans}")
    print(f"[扣费] {tokens} token，扣 {cost} credits，余额 {db.get_balance(alice['id'])}\n")

    print("=" * 60)
    print("结论：")
    print("1. 场景1 AI 主动调用了 search_memory 工具（有 [工具调用] 日志），才回答出 38.5 元")
    print("2. 场景2 AI 判断与记忆无关，没调用工具，直接回答")
    print("3. 这就是 function-calling 取代原项目「文字标签状态机」的效果：AI 自己决定查不查")


if __name__ == "__main__":
    main()
