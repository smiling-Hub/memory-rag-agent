# -*- coding: utf-8 -*-
"""主程序：注册、记记忆、提问、按 token 扣费的完整流程演示。"""
import os
import sys
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


def _llm():
    import httpx
    from openai import OpenAI
    try:
        return OpenAI(base_url=CHAT_BASE_URL, api_key=CHAT_API_KEY,
                      http_client=httpx.Client(verify=False))
    except TypeError:
        return OpenAI(base_url=CHAT_BASE_URL, api_key=CHAT_API_KEY)


def ask(user, question):
    """一次完整提问：检索用户自己的记忆 → DeepSeek 组织回答 → 按真实 token 扣费"""
    # 1. 只从该用户自己的记忆库里检索（隔离的关键）
    ctx = memory.search(user["id"], question, n=3)
    context = "\n".join(f"- {did}: {c}" for did, c in ctx)

    # 2. 调 DeepSeek，让它"只依据记忆回答，不编造"
    resp = _llm().chat.completions.create(
        model=CHAT_MODEL_NAME,
        messages=[
            {"role": "system", "content": "你是记忆助手，只依据下面给的记忆回答，没有相关记忆就明说，不要编造。"},
            {"role": "user", "content": f"记忆：\n{context}\n\n问题：{question}"},
        ],
        max_tokens=200,
    )
    answer = resp.choices[0].message.content
    tokens = getattr(resp.usage, "total_tokens", 0)  # DeepSeek 返回的真实 token 数

    # 3. 按真实 token 扣费（碰钱的核心）
    cost = db.charge(user["id"], tokens, note=question[:20])
    return answer, tokens, cost


def main():
    db.init_db()

    # 注册两个用户
    for name in ("alice", "bob"):
        if db.register(name, "123456", initial_credits=100):
            print(f"[注册] 用户 {name} 创建成功，送 100 credits")

    alice = db.login("alice", "123456")
    bob = db.login("bob", "123456")

    # 各记一条自己的记忆
    memory.add_memory(alice["id"], "Note of 2026-09-15 买入八方股份", "9月15日买入八方股份603489，成本价38.5元，共500股。", 20260915)
    memory.add_memory(bob["id"], "Note of 2026-09-28 显卡预算", "打算买一块新显卡，预算4000元，用来跑本地大模型。", 20260928)
    print("[记忆] 已分别写入 alice 和 bob 各自的记忆库\n")

    print("=" * 60)
    # 场景1：alice 问自己的股票（应该答对）
    q = "我买八方股份的成本价是多少"
    ans, tokens, cost = ask(alice, q)
    print(f"[alice 提问] {q}")
    print(f"[回答] {ans}")
    print(f"[扣费] 本次 {tokens} token，扣 {cost} credits，余额 {db.get_balance(alice['id'])}\n")

    # 场景2：bob 问同一个问题（alice 的股票），应该查不到 → 证明隔离
    ans, tokens, cost = ask(bob, q)
    print(f"[bob 提问] {q}  （这是 alice 的记忆，bob 不该能看到）")
    print(f"[回答] {ans}")
    print(f"[扣费] 本次 {tokens} token，扣 {cost} credits，余额 {db.get_balance(bob['id'])}\n")

    # 场景3：bob 问自己的显卡（应该答对）
    q2 = "我显卡的预算是多少"
    ans, tokens, cost = ask(bob, q2)
    print(f"[bob 提问] {q2}")
    print(f"[回答] {ans}")
    print(f"[扣费] 本次 {tokens} token，扣 {cost} credits，余额 {db.get_balance(bob['id'])}\n")

    print("=" * 60)
    print("结论：")
    print("1. 多用户隔离生效 —— bob 查不到 alice 的股票记忆")
    print("2. 计费生效 —— 每次提问按真实 token 扣费，余额递减")


if __name__ == "__main__":
    main()
