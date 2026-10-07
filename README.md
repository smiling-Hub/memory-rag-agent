# 记忆型 AI 助手

带记忆的 RAG 聊天助手，支持多用户隔离和按 token 计费。

## 功能

- 混合检索：语义向量 + BM25 关键词 + 时间过滤
- function-calling Agent：AI 自己决定是否检索记忆
- 多用户隔离：每人独立记忆库
- Credits 计费：按真实 token 消耗扣费

## 运行

```bash
pip install -r requirements.txt
# 在 .env 里配置 DEEPSEEK_API_KEY=xxx
python agent.py
```

首次运行会下载本地 embedding 模型。

## 测试

```bash
pytest tests/ -v
```

## 目录

- `agent.py` — function-calling Agent
- `db.py` — 用户与计费
- `memory.py` — 记忆与多租户隔离
- `demo_retrieval.py` — 检索演示
- `eval_retrieval.py` — 检索评测
- `tests/` — 单元测试
