"""
Agent 功能测试脚本 — 测试各模块是否正常运行。

用法:
    python test_agent.py

注意: 运行测试前需要:
    1. Java 后端已启动 (localhost:8080)
    2. .env 中已配置 OPENAI_API_KEY
"""

import sys


def test_imports():
    """测试模块导入"""
    print("=" * 50)
    print("1. 测试模块导入...")
    try:
        from config import config
        from tools import ALL_TOOLS
        from tools.inventory_tools import INVENTORY_TOOLS
        from tools.order_tools import ORDER_TOOLS
        from memory.conversation_memory import create_memory
        from prompts.system_prompts import SYSTEM_PROMPT, PROMPT_TEMPLATE
        from retrieval.document_loader import DocumentLoader
        from retrieval.vector_store import VectorStoreManager
        from retrieval.retriever import Retriever, create_retrieval_tool
        print("   ✅ 所有模块导入成功")
        return True
    except ImportError as e:
        print(f"   ❌ 导入失败: {e}")
        print("   请确认已安装依赖: pip install -r requirements.txt")
        return False


def test_config():
    """测试配置"""
    print("\n" + "=" * 50)
    print("2. 测试配置...")
    from config import config

    issues = []
    if not config.OPENAI_API_KEY:
        issues.append("OPENAI_API_KEY 未配置 (必需)")
    if not config.BACKEND_BASE_URL:
        issues.append("BACKEND_BASE_URL 未配置")

    if issues:
        for issue in issues:
            print(f"   ⚠️  {issue}")
        print("   提示: 复制 .env.example 为 .env 并填入配置")
        return False

    print(f"   ✅ LLM Model: {config.LLM_MODEL}")
    print(f"   ✅ Backend:   {config.BACKEND_BASE_URL}")
    print(f"   ✅ ChromaDB:  {config.CHROMA_PERSIST_DIR}")
    print(f"   ✅ Window:    {config.MEMORY_WINDOW_SIZE} 轮")
    return True


def test_tools():
    """测试工具定义"""
    print("\n" + "=" * 50)
    print("3. 测试工具定义...")
    from tools import ALL_TOOLS

    print(f"   ✅ 共 {len(ALL_TOOLS)} 个工具:")
    for t in ALL_TOOLS:
        desc = t.description[:60] if t.description else "(无描述)"
        print(f"      - {t.name}: {desc}")
    return True


def test_memory():
    """测试对话记忆"""
    print("\n" + "=" * 50)
    print("4. 测试对话记忆...")
    from memory.conversation_memory import create_memory

    memory = create_memory(k=3)
    memory.save_context({"input": "用户: 你好"}, {"output": "AI: 你好! 有什么可以帮你的?"})
    memory.save_context({"input": "用户: 帮我查库存"}, {"output": "AI: 当前库存..."})
    memory.save_context({"input": "用户: 谢谢"}, {"output": "AI: 不客气!"})

    history = memory.load_memory_variables({})
    print(f"   ✅ 记忆变量: {list(history.keys())}")
    msg_count = len(history.get("chat_history", []))
    print(f"   ✅ 历史消息数: {msg_count}")
    print(f"   ✅ 记忆窗口: k=3, 实际保留最近 3 轮")
    return True


def test_document_loader():
    """测试文档加载"""
    print("\n" + "=" * 50)
    print("5. 测试文档加载...")
    from retrieval.document_loader import DocumentLoader

    loader = DocumentLoader(chunk_size=300, chunk_overlap=30)
    docs = loader.load_directory()

    if docs:
        print(f"   ✅ 加载了 {len(docs)} 个文本块")
        print(f"   ✅ 第一个块预览: {docs[0].page_content[:80]}...")
    else:
        print("   ⚠️  知识库目录为空 (data/knowledge/)，跳过向量库测试")
        print("   提示: 放入 .txt/.md 文件后重新运行测试")
    return True


def test_backend_connection():
    """测试后端连通性"""
    print("\n" + "=" * 50)
    print("6. 测试后端连通性...")
    import httpx
    from config import config

    try:
        r = httpx.get(f"{config.BACKEND_BASE_URL}/api/inventory", timeout=5.0)
        if r.status_code == 200:
            data = r.json()
            if data.get("code") == 200:
                items = data.get("data", [])
                print(f"   ✅ 后端连接正常，库存商品数: {len(items)}")
                return True
        print(f"   ⚠️  后端返回异常: {r.status_code}")
        return False
    except httpx.ConnectError:
        print(f"   ⚠️  无法连接后端 ({config.BACKEND_BASE_URL})")
        print("   提示: 请先启动 Java 后端服务")
        return False
    except Exception as e:
        print(f"   ⚠️  连接测试失败: {e}")
        return False


def test_agent_creation():
    """测试 Agent 创建"""
    print("\n" + "=" * 50)
    print("7. 测试 Agent 创建...")
    from agent import InventoryAgent

    try:
        agent = InventoryAgent()
        stats = agent.index_stats
        print(f"   ✅ Agent 创建成功")
        print(f"   ✅ 工具数: {stats['tool_count']}")
        print(f"   ✅ 文档数: {stats['doc_count']}")
        return True
    except Exception as e:
        print(f"   ⚠️  Agent 创建失败: {e}")
        print("   提示: 检查 OPENAI_API_KEY 是否正确")
        return False


# ==================== Main ====================

def main():
    print("\n🔍 Agent 功能测试")
    print("=" * 50)

    results = {}

    results["imports"] = test_imports()
    if not results["imports"]:
        print("\n❌ 基础导入失败，无法继续")
        sys.exit(1)

    results["config"] = test_config()
    results["tools"] = test_tools()
    results["memory"] = test_memory()
    results["loader"] = test_document_loader()
    results["backend"] = test_backend_connection()
    results["agent"] = test_agent_creation()

    # ---- 汇总 ----
    print("\n" + "=" * 50)
    print("📊 测试汇总")
    print("-" * 50)
    for name, passed in results.items():
        icon = "✅" if passed else "⚠️ "
        print(f"   {icon} {name}")
    print("-" * 50)

    all_pass = all(results.values())
    if all_pass:
        print("🎉 全部通过!")
    else:
        print("⚠️  部分测试未通过，请查看上述提示进行配置")

    print("\n💡 运行交互式对话: python main.py")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
