"""
对话记忆管理 — 使用 LangChain 的 ConversationBufferWindowMemory，
保留最近 K 轮对话，实现多轮对话上下文连贯。

设计要点:
- ConversationBufferWindowMemory: 滑动窗口，只保留最近 K 轮
- memory_key="chat_history": 与提示词模板中 {chat_history} 对应
- return_messages=True: 返回 LangChain Message 对象而非纯字符串
- 同时在 input_key/output_key 中指定 human/ai 的角色区分
"""

from langchain.memory import ConversationBufferWindowMemory
from config import config


def create_memory(k: int | None = None) -> ConversationBufferWindowMemory:
    """
    创建带滑动窗口的对话记忆。

    参数:
        k: 保留的最近对话轮数，默认取配置文件中的 MEMORY_WINDOW_SIZE

    返回:
        ConversationBufferWindowMemory 实例
    """
    window_size = k if k is not None else config.MEMORY_WINDOW_SIZE

    memory = ConversationBufferWindowMemory(
        k=window_size,
        memory_key="chat_history",        # 提示词模板中的占位变量名
        return_messages=True,              # 返回 Message 对象列表
        input_key="input",                 # 用户输入 key
        output_key="output",              # AI 输出 key
    )
    return memory
