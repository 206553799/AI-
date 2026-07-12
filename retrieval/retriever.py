"""
检索器封装 — Agent 调用此模块从向量库中检索相关知识。

检索策略:
- 默认使用 MMR (Max Marginal Relevance) 进行去重，保证结果多样性
- fetch_k 先取更多候选再精选 Top-K
- 对检索结果进行后处理：去空、截断过长内容
"""

from langchain_core.documents import Document
from langchain_core.tools import tool
from .vector_store import VectorStoreManager
from config import config


class Retriever:
    """检索器 — 封装向量库的查询逻辑，同时注册为 LangChain Tool"""

    def __init__(self, vector_manager: VectorStoreManager):
        self.vm = vector_manager
        self.bm25 = None      # BM25 检索器
        self.reranker = None  # 重排序器
        

    # ==================== 通用检索 ====================

    def retrieve(self, query: str, k: int | None = None,
                 use_mmr: bool = True) -> list[Document]:
        """
        检索相关文档。

        参数:
            query: 检索查询
            k: 返回结果数，默认取配置中的 RETRIEVAL_TOP_K
            use_mmr: 是否使用 MMR 去重

        返回:
            相关 Document 列表
        """
        k = k or config.RETRIEVAL_TOP_K
        fetch_k = k * 2  # MMR 先取 2 倍候选

        if use_mmr:
            docs = self.vm.vector_store.max_marginal_relevance_search(
                query, k=k, fetch_k=fetch_k
            )
        else:
            docs = self.vm.similarity_search(query, k=k)

        return self._post_process(docs)

    def retrieve_with_scores(self, query: str, k: int | None = None,
                             score_threshold: float | None = None) -> list[tuple[Document, float]]:
        """
        检索 + 相似度分数。

        参数:
            query: 检索查询
            k: 返回结果数
            score_threshold: 分数阈值，低于此值的被过滤

        返回:
            (Document, score) 列表
        """
        k = k or config.RETRIEVAL_TOP_K
        threshold = score_threshold if score_threshold is not None else config.RETRIEVAL_SCORE_THRESHOLD

        results = self.vm.similarity_search_with_score(query, k=k)
        # 过滤低分
        results = [(doc, score) for doc, score in results if score >= threshold]
        return results

    # ==================== 格式化为上下文 ====================

    def retrieve_as_context(self, query: str, k: int | None = None) -> str:
        """
        检索并将结果拼接为上下文文本，供 LLM 直接使用。

        参数:
            query: 检索查询
            k: 结果数

        返回:
            格式化后的上下文字符串
        """
        results = self.retrieve_with_rerank(query, k=k)
        if not results:
            return "（未检索到相关内容）"

        parts = []
        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get("source", "未知来源") if hasattr(doc, 'metadata') else "未知来源"
            content = doc.page_content.strip() if hasattr(doc, 'page_content') else str(doc).strip()
            parts.append(f"[片段 {i}] 来源: {source} (分数: {score:.3f})\n{content}")

        return "\n\n".join(parts)
    
    # ==================== 混合检索 ====================

    def retrieve_hybrid(self, query: str, k: int | None = None) -> list[tuple[Document, float]]:
        """
        BM25 + 向量 双路召回，RRF 融合排序。

        返回: [(Document, fused_score), ...] 按分数降序
        """
        k = k or config.RETRIEVAL_TOP_K

        # 路1：向量检索 (多拿一些候选)
        vector_docs = self.vm.similarity_search(query, k=k * 3)

        # 路2：BM25 关键词检索
        bm25_results = self.bm25.search(query, k=k * 3) if self.bm25 else []
        bm25_docs = [(self.bm25.docs[i], score) for i, score in bm25_results]

        # RRF 融合
        return self._rrf_fusion(vector_docs, bm25_docs, k=k)

    def _rrf_fusion(self, docs_a: list[Document], docs_b: list[tuple[str, float]],
                    k: int = 4, rrf_k: int = 60) -> list[tuple[Document, float]]:
        """
        Reciprocal Rank Fusion：多路结果融合，不依赖分数绝对值。
        """
        scores: dict[int, float] = {}
        doc_map: dict[int, Document] = {}

        # 路1：按排名贡献分数
        for rank, doc in enumerate(docs_a):
            key = hash(doc.page_content)
            doc_map[key] = doc
            scores[key] = scores.get(key, 0) + 1.0 / (rrf_k + rank + 1)

        # 路2：BM25 结果
        for rank, (text, _) in enumerate(docs_b):
            key = hash(text)
            if key not in doc_map:
                doc_map[key] = Document(page_content=text)
            scores[key] = scores.get(key, 0) + 1.0 / (rrf_k + rank + 1)

        # 按 RRF 分数排序
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:k]
        return [(doc_map[key], score) for key, score in ranked]

    def retrieve_with_rerank(self, query: str, k: int | None = None,
                             candidate_k: int = 12) -> list[tuple[Document, float]]:
        """混合检索 + 重排序：先多拿候选，再用 Reranker 精排取 Top-K"""
        k = k or config.RETRIEVAL_TOP_K
        candidates = self.retrieve_hybrid(query, k=candidate_k)

        if self.reranker and candidates:
            docs = [doc.page_content for doc, _ in candidates]
            ranked = self.reranker.rerank(query, docs, top_n=k)
            if ranked:
                return [(candidates[i][0], score) for i, score in ranked]
        return candidates[:k]

    # ==================== 后处理 ====================

    def _post_process(self, docs: list[Document]) -> list[Document]:
        """去空 & 去重"""
        seen = set()
        cleaned = []
        for doc in docs:
            content = doc.page_content.strip()
            if not content:
                continue
            # 简单去重 (基于内容 hash)
            key = hash(content)
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(doc)
        return cleaned

    # ==================== 统计 ====================

    @property
    def doc_count(self) -> int:
        return self.vm.count()


# ==================== 注册为 LangChain Tool ====================

def create_retrieval_tool(retriever: Retriever) -> tool:
    """将检索器包装为 LangChain Tool，供 Agent 调用"""

    @tool
    def search_knowledge_base(query: str) -> str:
        """
        从业务知识库中检索相关信息。当用户问到以下类型的问题时使用此工具:
        - 业务流程（如退货、退款、换货）
        - 公司政策或规则
        - 操作规范、FAQ
        - 任何需要查阅文档才能回答的问题

        参数:
            query: 检索查询语句，用自然语言描述想了解的内容

        返回: 检索到的相关文档片段
        """
        result = retriever.retrieve_as_context(query)
        return result

    return search_knowledge_base
