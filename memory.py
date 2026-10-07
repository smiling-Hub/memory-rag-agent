# -*- coding: utf-8 -*-
"""记忆层：每用户独立 ChromaDB collection，多租户隔离。"""
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from fastembed import TextEmbedding

_embed_model = TextEmbedding(model_name="BAAI/bge-small-zh-v1.5")


class _LocalEmbedding(EmbeddingFunction):
    """本地 embedding（DeepSeek 没有 embedding 接口，这里用本地模型）"""
    def __init__(self):
        self._m = _embed_model

    def __call__(self, input: Documents) -> Embeddings:
        return [e.tolist() for e in self._m.embed(list(input))]


_client = chromadb.PersistentClient(path="./demo_chroma")


def _collection(user_id):
    """每个用户一个独立的 collection：memories_<用户id>"""
    return _client.get_or_create_collection(
        name=f"memories_{user_id}",
        embedding_function=_LocalEmbedding(),
    )


def add_memory(user_id, doc_id, content, date_int):
    """给某个用户记一条记忆。date_int 是数字日期（如 20260915），
    因为 ChromaDB 的 $gte/$lte 只支持数字比较，存数字才能做时间过滤。"""
    _collection(user_id).upsert(
        documents=[content],
        ids=[doc_id],
        metadatas=[{"doc_id": doc_id, "doc_time": date_int}],
    )


def search(user_id, query, n=3):
    """只在该用户自己的记忆库里检索，返回 [(doc_id, 内容), ...]"""
    res = _collection(user_id).query(
        query_texts=[query], n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    return [(res["metadatas"][0][i]["doc_id"], res["documents"][0][i])
            for i in range(len(res["documents"][0]))]
