# 记忆型 AI 助手

带记忆的 RAG 聊天助手，支持多用户隔离和按 token 计费。

## 项目背景

基于开源记忆型 RAG 框架 Loyal-Elephie 二次开发。原框架贡献了「语义 + 关键词 + 时间」混合检索的核心思路；Agent、计费、多租户、安全与工程化均为本人重构与新增。

## 功能

- 混合检索：语义向量 + BM25 关键词 + 时间过滤
- function-calling Agent：AI 自己决定是否检索记忆
- 多用户隔离：每人独立记忆库
- Credits 计费：按真实 token 消耗扣费

## ✨本人改造与新增模块

- Agent 重构：用原生 function-calling 取代原「XML 标签 + 正则解析」状态机
- 多租户隔离：每用户独立 ChromaDB collection，记忆物理隔离
- 安全改造：密码改用 bcrypt 哈希存储
- SQLite 计费：新增用户 / 余额 / 用量流水表，按真实 token 原子扣费
- 检索修复：日期字段由字符串改为数字存储，移除原框架 monkey-patch ChromaDB 的补丁
- 单元测试：pytest 覆盖注册、登录、计费等核心逻辑
- CI 自动化测试：GitHub Actions，push 自动跑测试

## 技术栈

Python / ChromaDB / BM25 / jieba / fastembed / DeepSeek / SQLite / pytest / GitHub Actions

## 运行

```bash
pip install -r requirements.txt
cp .env.example .env   # 填入你的 DEEPSEEK_API_KEY
python agent.py
```

首次运行会下载本地 embedding 模型。

## 环境变量

| 变量 | 说明 |
|---|---|
| DEEPSEEK_API_KEY | DeepSeek API key（必填） |

安全提醒：`.env` 已被 `.gitignore` 排除，切勿把 key 提交到仓库或硬编码进代码。

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

## 许可证

MIT License。本项目基于 [Loyal-Elephie](https://github.com/v2rockets/Loyal-Elephie)（MIT，Copyright (c) 2024 Yipeng Zhang）二次开发，详见 [LICENSE](LICENSE)。
