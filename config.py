"""
全局配置 — 优先从 .env 文件读取环境变量。
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ---- LLM ----
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0"))

    # ---- 后端 ----
    BACKEND_BASE_URL: str = os.getenv("BACKEND_BASE_URL", "http://localhost:8080")

    # ---- 向量库 ----
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "business_knowledge")

    # ---- Embedding (可独立配置，未设置则回退到 LLM 配置) ----
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    EMBEDDING_API_KEY: str = os.getenv("EMBEDDING_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    EMBEDDING_BASE_URL: str = os.getenv("EMBEDDING_BASE_URL", "") or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    # ---- 记忆 ----
    MEMORY_WINDOW_SIZE: int = int(os.getenv("MEMORY_WINDOW_SIZE", "10"))

    # ---- 检索 ----
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "4"))
    RETRIEVAL_SCORE_THRESHOLD: float = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.3"))


config = Config()
