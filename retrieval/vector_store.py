"""
向量库管理 — 基于 ChromaDB 实现本地向量存储与相似度检索。

流程:
1. 首次运行时，加载文档 → 计算 Embedding → 写入 ChromaDB
2. 后续运行时直接从 ChromaDB 加载已有向量库
3. 提供 add_documents 用于增量添加文档
"""

from pathlib import Path
import httpx
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from config import config


class CompatibleEmbeddings(Embeddings):
    """百炼 MaaS / DashScope 兼容的 Embedding 类。

    绕过 openai SDK 2.x 的请求格式不兼容问题，直接用 httpx 发请求。
    """

    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = httpx.post(
            f"{self.base_url}/embeddings",
            json={"input": texts, "model": self.model},
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data["data"]]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class VectorStoreManager:
    """ChromaDB 向量库管理器"""

    def __init__(self):
        self.persist_dir = config.CHROMA_PERSIST_DIR
        self.collection_name = config.CHROMA_COLLECTION_NAME

        # Embedding 模型 (使用兼容封装，支持百炼 MaaS / DashScope)
        self.embeddings = CompatibleEmbeddings(
            api_key=config.EMBEDDING_API_KEY,
            base_url=config.EMBEDDING_BASE_URL,
            model=config.EMBEDDING_MODEL,
        )

        self._vector_store: Chroma | None = None

    # ==================== 初始化 ====================

    def initialize(self, documents: list[Document] | None = None,
                   force_rebuild: bool = False) -> Chroma:
        """
        初始化向量库。
        - 已有数据且不强制重建 → 从磁盘加载
        - 无数据或强制重建 → 从文档创建

        参数:
            documents: 要索引的文档块列表
            force_rebuild: 是否强制重建向量库

        返回:
            Chroma 向量库实例
        """
        persist_path = Path(self.persist_dir)

        # 已有向量库且不强制重建 → 直接加载
        if persist_path.exists() and not force_rebuild and any(persist_path.iterdir()):
            print(f"[VectorStore] 从 {self.persist_dir} 加载已有向量库")
            self._vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_dir,
            )
            return self._vector_store

        # 否则从文档构建
        if documents is None:
            documents = []

        print(f"[VectorStore] 新建向量库，索引 {len(documents)} 个文档块")
        persist_path.mkdir(parents=True, exist_ok=True)

        self._vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_dir,
        )
        return self._vector_store

    # ==================== 查询 ====================

    @property
    def vector_store(self) -> Chroma:
        if self._vector_store is None:
            raise RuntimeError("向量库未初始化，请先调用 initialize()")
        return self._vector_store

    @property
    def retriever(self):
        """返回 LangChain Retriever 接口"""
        return self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": config.RETRIEVAL_TOP_K,
                "score_threshold": config.RETRIEVAL_SCORE_THRESHOLD,
            },
        )

    # ==================== 管理 ====================

    def add_documents(self, documents: list[Document]) -> None:
        """增量添加文档"""
        self.vector_store.add_documents(documents)
        print(f"[VectorStore] 增量添加 {len(documents)} 个文档块")

    def delete_collection(self) -> None:
        """删除整个集合"""
        self.vector_store.delete_collection()
        self._vector_store = None
        print("[VectorStore] 集合已删除")

    def count(self) -> int:
        """当前索引的文档数"""
        return self.vector_store._collection.count()

    def similarity_search(self, query: str, k: int | None = None) -> list[Document]:
        """语义相似度搜索"""
        k = k or config.RETRIEVAL_TOP_K
        return self.vector_store.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str,
                                     k: int | None = None) -> list[tuple[Document, float]]:
        """语义搜索 + 相似度分数"""
        k = k or config.RETRIEVAL_TOP_K
        return self.vector_store.similarity_search_with_relevance_scores(query, k=k)
