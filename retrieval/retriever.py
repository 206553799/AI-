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
        docs = self.retrieve(query, k=k)
        if not docs:
            return "（未检索到相关内容）"

        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "未知来源")
            content = doc.page_content.strip()
            parts.append(f"[片段 {i}] 来源: {source}\n{content}")

        return "\n\n".join(parts)

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
