# -*- coding: utf-8 -*-
"""
检索评测：用数字回答「检索质量好不好」（对应岗位的「模型评测」）。

方法：
  1. 准备一个「有标准答案」的测试集：每条查询都标注了「应该命中哪条记忆」
  2. 分别用「纯语义」和「语义 + BM25 混合」跑检索
  3. 算 hit@1 / hit@3 / hit@5 和 MRR，对比两种方法

指标解释：
  - hit@k：正确记忆出现在前 k 名里的查询占比（越高越好）
  - MRR：平均倒数排名，正确记忆排第 1 得 1，排第 2 得 0.5，越靠前分越高

跑法：venv/Scripts/python eval_retrieval.py
"""
import os
import sys
import ssl

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ssl._create_default_https_context = ssl._create_unverified_context
_orig = ssl.create_default_context
def _unverified(*a, **k):
    c = _orig(*a, **k)
    c.check_hostname = False
    c.verify_mode = ssl.CERT_NONE
    return c
ssl.create_default_context = _unverified

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from fastembed import TextEmbedding
import jieba
from rank_bm25 import BM25Okapi

# ============ 1. 语料库（15 条记忆，含干扰项）============
CORPUS = [
    ("Note of 2026-09-15 买入八方股份", 20260915, "9月15日买入八方股份603489，成本价38.5元，共500股，理由是电助力车行业景气度回升。"),
    ("Note of 2026-09-20 止损纪律", 20260920, "定下止损纪律：单只股票亏损超过8%必须止损，绝不补仓摊低成本。"),
    ("Note of 2026-09-28 显卡预算", 20260928, "打算买一块新显卡，预算4000元，主要用来跑本地大模型推理。"),
    ("Note of 2026-08-10 回测参数", 20260810, "回测参数：佣金万2.5，印花税卖出0.05%，初始资金100万，基准沪深300。"),
    ("Note of 2026-07-30 大盘观点", 20260730, "7月底判断大盘处于震荡市，决定控制仓位在半仓以下，等待方向明确。"),
    ("Note of 2026-06-15 加仓规则", 20260615, "加仓规则：只有盈利超过20%才考虑加仓，不追高。"),
    ("Note of 2026-06-20 持仓周期", 20260620, "我的持仓周期偏短线，一般3到5天。"),
    ("Note of 2026-05-10 关注行业", 20260510, "长期关注两个行业：新能源和半导体。"),
    ("Note of 2026-05-15 定投计划", 20260515, "定投计划：每月15号定投沪深300ETF，金额2000元。"),
    ("Note of 2026-04-10 亏损教训", 20260410, "吃过一次亏：追高买入，最后亏损15%割肉离场。"),
    ("Note of 2026-04-20 目标收益", 20260420, "我的目标年化收益率是20%。"),
    ("Note of 2026-03-10 风险偏好", 20260310, "风险偏好稳健，不做杠杆，不碰高风险品种。"),
    ("Note of 2026-03-15 学习计划", 20260315, "学习计划：先学完Python量化，再学深度学习。"),
    ("Note of 2026-02-10 旧显卡", 20260210, "之前买过一块2060显卡，现在跑大模型有点吃力。"),
    ("Note of 2026-01-10 复盘习惯", 20260110, "每天晚上花一小时复盘当天行情和操作。"),
]

# ============ 2. 标注测试集（查询 -> 应该命中的正确记忆）============
TEST_SET = [
    ("我买八方股份的成本价是多少", "Note of 2026-09-15 买入八方股份"),
    ("我的止损纪律是什么", "Note of 2026-09-20 止损纪律"),
    ("我显卡的预算是多少", "Note of 2026-09-28 显卡预算"),
    ("回测的佣金和印花税是多少", "Note of 2026-08-10 回测参数"),
    ("我对大盘走势的判断", "Note of 2026-07-30 大盘观点"),
    ("我的加仓条件是什么", "Note of 2026-06-15 加仓规则"),
    ("我的持仓周期多久", "Note of 2026-06-20 持仓周期"),
    ("我关注哪些行业", "Note of 2026-05-10 关注行业"),
    ("我的定投计划是什么", "Note of 2026-05-15 定投计划"),
    ("我的风险偏好", "Note of 2026-03-10 风险偏好"),
    # ---- 以下为「硬题」：有干扰项，或需靠关键词精确匹配（看混合能否救回语义的失误）----
    ("我 603489 的成本是多少", "Note of 2026-09-15 买入八方股份"),  # 股票代码，语义可能不认识
    ("我的 20% 目标是什么", "Note of 2026-04-20 目标收益"),          # "20%" 出现在多条记忆里
    ("我每月定投多少钱", "Note of 2026-05-15 定投计划"),             # "沪深300" 也在回测参数里
    ("2060 显卡现在怎么样", "Note of 2026-02-10 旧显卡"),            # 显卡型号，和"显卡预算"形成干扰
    ("我上次亏损的教训", "Note of 2026-04-10 亏损教训"),             # 和"止损纪律"语义接近
]

# ============ 3. 建索引（向量 + BM25）============
_embed_model = TextEmbedding(model_name="BAAI/bge-small-zh-v1.5")


class _LocalEmbedding(EmbeddingFunction):
    def __init__(self):
        self._m = _embed_model

    def __call__(self, input: Documents) -> Embeddings:
        return [e.tolist() for e in self._m.embed(list(input))]


_client = chromadb.PersistentClient(path="./eval_chroma")
_col = _client.get_or_create_collection(name="eval_memories", embedding_function=_LocalEmbedding())

for doc_id, date, content in CORPUS:
    _col.upsert(documents=[content], ids=[doc_id],
                metadatas=[{"doc_id": doc_id, "doc_time": date}])


def _tokenize(text):
    return [t for t in jieba.lcut(text.lower()) if t.strip()]


_bm25 = BM25Okapi([_tokenize(c) for _, _, c in CORPUS])
_bm25_map = {doc_id: i for i, (doc_id, _, _) in enumerate(CORPUS)}

# ============ 4. 两种检索 ============
def _semantic_candidates(query, n_candidates=10):
    """语义检索，返回 (候选 doc_id 列表, 距离列表)，供两种排序共用"""
    res = _col.query(query_texts=[query], n_results=n_candidates,
                     include=["metadatas", "distances"])
    ids = [res["metadatas"][0][i]["doc_id"] for i in range(len(res["metadatas"][0]))]
    dists = res["distances"][0]
    return ids, dists


def semantic_rank(query, n_candidates=10):
    """纯语义：按向量距离排序，返回排序后的 doc_id 列表"""
    ids, _ = _semantic_candidates(query, n_candidates)
    return ids


def blended_rank(query, n_candidates=10):
    """语义 + BM25 融合：对全语料算「语义分 + 0.1 * 归一化 BM25」，再排序。

    注意：这是「融合」（两个检索器各自打分后合并），不是「重排」（只重排语义候选）。
    融合的好处：语义漏掉的文档，能被 BM25 的关键词命中捞回来。
    """
    # 语义：语料小，全量取（这样 BM25 有机会捞回语义没排进的文档）
    res = _col.query(query_texts=[query], n_results=len(CORPUS),
                     include=["metadatas", "distances"])
    sem_map = {res["metadatas"][0][i]["doc_id"]: 1 / (res["distances"][0][i] + 0.01)
               for i in range(len(res["metadatas"][0]))}
    # BM25：全语料打分
    q = _tokenize(query)
    bm25_all = _bm25.get_scores(q)
    mx = max(bm25_all) if max(bm25_all) > 0 else 1
    # 融合打分
    scored = [(did, sem_map.get(did, 0) + 0.1 * (bm25_all[i] / mx))
              for did, i in _bm25_map.items()]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scored[:n_candidates]]


# ============ 5. 评测 ============
def evaluate(rank_fn):
    hit1 = hit3 = hit5 = 0
    mrr = 0.0
    rows = []
    for query, correct in TEST_SET:
        ranked = rank_fn(query)
        r = ranked.index(correct) + 1 if correct in ranked else 0
        mrr += 1 / r if r else 0
        if r == 1:
            hit1 += 1
        if 0 < r <= 3:
            hit3 += 1
        if 0 < r <= 5:
            hit5 += 1
        rows.append((query, correct, r))
    n = len(TEST_SET)
    return {"hit@1": hit1 / n, "hit@3": hit3 / n, "hit@5": hit5 / n, "MRR": mrr / n}, rows


if __name__ == "__main__":
    sem_metrics, sem_rows = evaluate(semantic_rank)
    bl_metrics, bl_rows = evaluate(blended_rank)

    print("=" * 64)
    print("检索评测结果（对比：纯语义 vs 语义+BM25 混合）")
    print("=" * 64)
    print(f"{'指标':<8}{'纯语义':>10}{'混合':>10}")
    for k in ["hit@1", "hit@3", "hit@5", "MRR"]:
        print(f"{k:<8}{sem_metrics[k]:>10.2%}{bl_metrics[k]:>10.2%}")

    print("\n逐条排名（0 = 没进前 10）：")
    print(f"{'查询':<26}{'正确排名(语义)':>12}{'正确排名(混合)':>14}")
    for (q, c, r_sem), (_, _, r_bl) in zip(sem_rows, bl_rows):
        print(f"{q:<26}{r_sem:>12}{r_bl:>14}")

    print("\n结论：")
    if bl_metrics["MRR"] >= sem_metrics["MRR"]:
        print(f"  混合检索 MRR 不劣于纯语义（{bl_metrics['MRR']:.2%} vs {sem_metrics['MRR']:.2%}），"
              f"说明 BM25 对语义排序至少无害，对『关键词重叠』的查询有正向修正。")
    else:
        print(f"  这个测试集上纯语义 MRR 反而更高，说明当查询和记忆用词高度重叠时，BM25 的 0.1 权重贡献有限。")
    print("  这正是『模型评测』的价值：不是凭感觉说检索好不好，而是有数字、能对比、能定位问题。")
