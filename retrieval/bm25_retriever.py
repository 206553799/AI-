"""
BM25 关键词检索器 — 基于 jieba 分词，与向量检索互补。
"""
import math
import jieba


class BM25Retriever:
    """轻量级 BM25 检索器"""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = documents
        self.k1 = k1
        self.b = b
        self.tokenized = [list(jieba.cut(d)) for d in documents]
        self.doc_count = len(documents)
        self.avgdl = sum(len(t) for t in self.tokenized) / max(self.doc_count, 1)
        self._df = {}
        for tokens in self.tokenized:
            for term in set(tokens):
                self._df[term] = self._df.get(term, 0) + 1

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        return math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str, k: int = 4) -> list[tuple[int, float]]:
        query_tokens = list(jieba.cut(query))
        scores = []
        for idx, tokens in enumerate(self.tokenized):
            score = 0.0
            doc_len = len(tokens)
            term_freq = {}
            for t in tokens:
                term_freq[t] = term_freq.get(t, 0) + 1
            for term in set(query_tokens):
                tf = term_freq.get(term, 0)
                if tf == 0:
                    continue
                idf = self._idf(term)
                score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
            scores.append((idx, score))
        ranked = sorted(scores, key=lambda x: -x[1])[:k]
        return [(i, s) for i, s in ranked if s > 0]


# ==================== Reranker ====================

class DashScopeReranker:
    """百炼 DashScope 重排序 — 对候选文档精排"""

    def __init__(self, api_key: str, base_url: str, model: str = "gte-rerank-v2"):
        # base_url 示例: https://dashscope.aliyuncs.com/api/v1
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    def rerank(self, query: str, documents: list[str], top_n: int = 4) -> list[tuple[int, float]]:
        """返回 [(原始索引, 相关性分数), ...]"""
        import httpx
        try:
            response = httpx.post(
                f"{self.base_url}/services/rerank/text-rerank/text-rerank",
                json={
                    "model": self.model,
                    "input": {"query": query, "documents": documents},
                    "parameters": {"top_n": top_n},
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            results = data.get("output", {}).get("results", [])
            return [(r["index"], r["relevance_score"]) for r in results]
        except Exception:
            # Rerank 失败时返回原始顺序，上游会跳过
            return []
