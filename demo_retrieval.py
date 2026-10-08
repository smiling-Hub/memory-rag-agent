# -*- coding: utf-8 -*-
"""检索核心演示：语义向量 + BM25 + 时间过滤三路混合检索。"""
import os
import sys
import datetime
import ssl

# Windows 终端默认 GBK，强制 stdout 用 UTF-8，否则中文输出会乱码
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 本机根证书缺失，Python 的 requests / httpx（huggingface_hub、openai 都走 httpx）会证书校验失败。
# 这里全局关闭 SSL 校验，只为让本地 embedding 模型能从 HuggingFace 下载下来（仅本地演示用，生产不能这么干）。
ssl._create_default_https_context = ssl._create_unverified_context
_orig_create_default_context = ssl.create_default_context
def _unverified_context(*args, **kwargs):
    ctx = _orig_create_default_context(*args, **kwargs)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx
ssl.create_default_context = _unverified_context

# DeepSeek 聊天接口
CHAT_BASE_URL = "https://api.deepseek.com/v1"
from dotenv import load_dotenv
load_dotenv()
CHAT_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
CHAT_MODEL_NAME = "deepseek-v4-pro"

# 本地 embedding 模型
# 走 hf-mirror.com 镜像，避免直连 HuggingFace 的证书/网络问题
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# 混合检索权重
BM25_WEIGHT = 0.1
TOP_K = 5  # 每个查询返回多少条

# ============ 1. 本地 embedding ============
print("[1] 加载本地 embedding 模型...")
from fastembed import TextEmbedding
embed_model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)

def get_embeddings(texts):
    """把一批文字变成向量"""
    return [e.tolist() for e in embed_model.embed(list(texts))]

# ============ 2. ChromaDB 向量库 ============
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

class LocalEmbedding(EmbeddingFunction):
    """告诉 ChromaDB 用本地模型向量化"""
    def __call__(self, input: Documents) -> Embeddings:
        return get_embeddings(input)

client = chromadb.PersistentClient(path="./demo_chroma")
collection = client.get_or_create_collection(name="memories", embedding_function=LocalEmbedding())

# ============ 3. 示例记忆（交易/量化场景，带日期）============
sample_memories = [
    {"doc_id": "Note of 2026-09-15 买入八方股份", "date": "2026-09-15",
     "content": "9月15日买入八方股份603489，成本价38.5元，共500股，理由是电助力车行业景气度回升。"},
    {"doc_id": "Note of 2026-09-20 止损纪律", "date": "2026-09-20",
     "content": "定下止损纪律：单只股票亏损超过8%必须止损，绝不补仓摊低成本。"},
    {"doc_id": "Note of 2026-09-28 显卡预算", "date": "2026-09-28",
     "content": "打算买一块新显卡，预算4000元，主要用来跑本地大模型推理。"},
    {"doc_id": "Note of 2026-08-10 回测参数", "date": "2026-08-10",
     "content": "回测参数：佣金万2.5，印花税卖出0.05%，初始资金100万，基准沪深300。"},
    {"doc_id": "Note of 2026-07-30 大盘观点", "date": "2026-07-30",
     "content": "7月底判断大盘处于震荡市，决定控制仓位在半仓以下，等待方向明确。"},
]

print("[2] 写入记忆：原文存本地文件 + 向量存 ChromaDB + BM25 建索引")
for m in sample_memories:
    # 原文落盘
    with open(f"./demo_docs/{m['doc_id'].replace(':', ';')}.md", "w", encoding="utf-8") as f:
        f.write(m["content"])
    # 向量入库，metadata 带日期。日期存数字（20260915）而不是字符串，
    # 因为 ChromaDB 的 $gte/$lte 只支持数字比较。
    collection.upsert(
        documents=[m["content"]],
        ids=[m["doc_id"]],
        metadatas=[{"doc_id": m["doc_id"], "doc_time": int(m["date"].replace("-", ""))}],
    )

# ============ 4. BM25 关键词索引 ============
import jieba
from rank_bm25 import BM25Okapi

def tokenize(text):
    """中文用 jieba 分词"""
    return [t for t in jieba.lcut(text.lower()) if t.strip()]

corpus = [m["content"] for m in sample_memories]
bm25 = BM25Okapi([tokenize(c) for c in corpus])
print("    BM25 索引就绪\n")

# ============ 5. 混合检索 ============
def semantic_search(query, n=TOP_K):
    """语义检索：向量距离 → 分数"""
    res = collection.query(query_texts=[query], n_results=n,
                           include=["documents", "metadatas", "distances"])
    docs, metas, dists = res["documents"][0], res["metadatas"][0], res["distances"][0]
    return [(metas[i]["doc_id"], docs[i], 1 / (dists[i] + 0.01)) for i in range(len(docs))]

def blended_search(query, n=TOP_K):
    """语义分 + BM25 分融合"""
    sem = semantic_search(query, n)
    doc_ids = [s[0] for s in sem]
    # BM25 只对语义检索命中的候选打分
    bm25_scores = bm25.get_scores(tokenize(query))
    bm25_map = {m["doc_id"]: bm25_scores[i] for i, m in enumerate(sample_memories)}
    # 归一化 BM25 到类似量纲，再乘权重
    mx = max(bm25_scores) if max(bm25_scores) > 0 else 1
    blended = []
    for doc_id, doc, sem_score in sem:
        bm = bm25_map.get(doc_id, 0) / mx
        blended.append((doc_id, doc, sem_score, bm, sem_score + bm * BM25_WEIGHT))
    blended.sort(key=lambda x: x[4], reverse=True)
    return blended

def time_filtered_search(query, start, end, n=TOP_K):
    """时间范围过滤：start/end 传 "YYYY-MM-DD"，转成数字再比较。"""
    start_n = int(start.replace("-", ""))
    end_n = int(end.replace("-", ""))
    res = collection.query(
        query_texts=[query], n_results=n,
        where={"$and": [{"doc_time": {"$gte": start_n}}, {"doc_time": {"$lte": end_n}}]},
        include=["documents", "metadatas", "distances"],
    )
    docs, metas = res["documents"][0], res["metadatas"][0]
    return [(metas[i]["doc_id"], docs[i]) for i in range(len(docs))]

# ============ 6. 演示 ============
def show(title, rows, show_score=False):
    print(f"--- {title} ---")
    for r in rows:
        if show_score:
            doc_id, doc, sem, bm, total = r
            print(f"   [{doc_id}]\n       内容: {doc[:40]}...\n       语义分={sem:.3f}  BM25={bm:.3f}  总分={total:.3f}")
        else:
            print(f"   [{r[0]}]  {r[1][:40]}...")
    print()

q1 = "我上个月买的显卡花了多少钱"
print("=" * 60)
print(f"查询1: {q1}")
print("=" * 60)
show("① 纯语义检索（向量）", semantic_search(q1))
show("② 语义 + BM25 混合（三路中的两路）", blended_search(q1), show_score=True)
# 手动指定"上个月"≈9月，演示时间过滤
show("③ 加时间过滤：只查 2026-09-01 ~ 2026-09-30", time_filtered_search(q1, "2026-09-01", "2026-09-30"))

q2 = "我买八方股份的成本价是多少"
print("=" * 60)
print(f"查询2: {q2}")
print("=" * 60)
show("① 纯语义检索", semantic_search(q2))
show("② 语义 + BM25 混合", blended_search(q2), show_score=True)

# ============ 7. 把检索结果喂给 DeepSeek 组织语言回答 ============
print("=" * 60)
print("最终一步：把检索到的记忆喂给 DeepSeek，让它组织语言回答")
print("=" * 60)
try:
    import httpx
    from openai import OpenAI
    # openai 客户端走 httpx，同样会撞本机证书问题，给它一个关闭校验的 http_client
    try:
        llm = OpenAI(base_url=CHAT_BASE_URL, api_key=CHAT_API_KEY,
                     http_client=httpx.Client(verify=False))
    except TypeError:
        llm = OpenAI(base_url=CHAT_BASE_URL, api_key=CHAT_API_KEY)

    def answer(query):
        results = blended_search(query, n=3)
        context = "\n".join(f"- {doc}: {content}" for doc, content, *_ in results)
        resp = llm.chat.completions.create(
            model=CHAT_MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个记忆助手，只能依据下面给出的记忆内容回答，不要编造。"},
                {"role": "user", "content": f"记忆内容：\n{context}\n\n问题：{query}\n\n请依据记忆回答。"},
            ],
            max_tokens=200,
        )
        return resp.choices[0].message.content

    for q in [q1, q2]:
        print(f"\nQ: {q}")
        print(f"A: {answer(q)}")
except Exception as e:
    print(f"DeepSeek 调用失败（不影响上面的检索演示）：{type(e).__name__}: {e}")

print("\n演示完成。核心结论：检索已经能把「最相关的记忆」找出来，DeepSeek 负责用这些记忆组织答案。")
