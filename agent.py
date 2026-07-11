"""
Agent 核心 — 使用 LangGraph 的 create_react_agent 组装 工具 + 记忆 + 检索器。

架构:
    User Input
        │
        ▼
    ┌─────────────────────┐
    │  检索器 (Retriever)  │  ← 向量库检索相关知识
    └─────────┬───────────┘
              │ retrieved_context
              ▼
    ┌─────────────────────┐
    │  系统提示词            │  ← System Prompt + 检索上下文
    └─────────┬───────────┘
              │
              ▼
    ┌─────────────────────┐
    │  ReAct Agent         │  ← LLM 推理 → 选择工具 → 执行 → 观察 → 循环
    │  (LangGraph)         │
    └─────────┬───────────┘
              │
              ▼
    ┌─────────────────────┐
    │  对话记忆 (Checkpointer)│  ← LangGraph MemorySaver 持久化多轮对话
    └─────────────────────┘
"""

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
from config import config
from tools import ALL_TOOLS
from prompts.system_prompts import SYSTEM_PROMPT
from retrieval.vector_store import VectorStoreManager
from retrieval.document_loader import DocumentLoader
from retrieval.retriever import Retriever, create_retrieval_tool


class InventoryAgent:
    """库存与订单管理智能体"""

    def __init__(self):
        # ---- LLM ----
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            openai_api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
        )

        # ---- 向量库 & 检索器 ----
        self.vector_manager = VectorStoreManager()
        self._init_vector_store()
        self.retriever = Retriever(self.vector_manager)

        # ---- 检索工具 ----
        self.retrieval_tool = create_retrieval_tool(self.retriever)

        # ---- 全部工具 = 业务工具 + 检索工具 ----
        self.tools = ALL_TOOLS + [self.retrieval_tool]

        # ---- 对话记忆 (LangGraph MemorySaver) ----
        self.checkpointer = MemorySaver()

        # ---- Agent ----
        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            checkpointer=self.checkpointer,
        )

        self._thread_id = "default"

    # ==================== 向量库初始化 ====================

    def _init_vector_store(self):
        """加载文档并初始化向量库"""
        loader = DocumentLoader()
        documents = loader.load_directory()
        self.vector_manager.initialize(documents=documents)

    # ==================== 对话接口 ====================

    def chat(self, user_input: str) -> str:
        """
        处理用户输入并返回 Agent 回答。

        参数:
            user_input: 用户输入的自然语言文本

        返回:
            Agent 的回答
        """
        # Step 1: 检索相关知识
        retrieved_context = self.retriever.retrieve_as_context(user_input)

        # Step 2: 构建系统消息（每次动态注入检索结果）
        system_msg = SystemMessage(
            content=f"{SYSTEM_PROMPT}\n\n## 知识库检索结果\n{retrieved_context}"
        )

        # Step 3: 调用 Agent
        response = self.agent.invoke(
            {"messages": [system_msg, HumanMessage(content=user_input)]},
            config={
                "configurable": {"thread_id": self._thread_id},
                "recursion_limit": 25,
            },
        )

        # Step 4: 提取最后一条 AI 消息
        messages = response.get("messages", [])
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.type == "ai":
                return msg.content
        return ""

    # ==================== 管理接口 ====================

    def clear_memory(self):
        """清空对话记忆（重建 checkpointer 切换 thread_id）"""
        import uuid
        self.checkpointer = MemorySaver()
        self._thread_id = str(uuid.uuid4())[:8]
        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            checkpointer=self.checkpointer,
        )
        print("[Agent] 对话记忆已清空")

    def rebuild_index(self):
        """重建向量索引"""
        loader = DocumentLoader()
        documents = loader.load_directory()
        self.vector_manager.initialize(documents=documents, force_rebuild=True)
        print("[Agent] 向量索引已重建")

    def add_knowledge(self, file_path: str):
        """增量添加知识文档"""
        loader = DocumentLoader()
        docs = loader.load_file(file_path)
        self.vector_manager.add_documents(docs)

    @property
    def index_stats(self) -> dict:
        """向量库统计"""
        return {
            "doc_count": self.vector_manager.count(),
            "tool_count": len(self.tools),
            "memory_window": config.MEMORY_WINDOW_SIZE,
        }
