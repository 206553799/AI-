"""
命令行交互式入口 — 启动 Agent 进行多轮对话。

用法:
    python main.py              # 交互式对话
    python main.py --rebuild    # 重建向量库后启动
    python main.py --stats      # 显示向量库统计信息后启动
"""

import sys
from agent import InventoryAgent

# 修复 Windows 终端 emoji 编码问题
sys.stdout.reconfigure(encoding='utf-8')


def print_banner():
    print("""
╔══════════════════════════════════════════╗
║     🏭 仓库管理与订单助手 (AI Agent)      ║
║                                          ║
║  支持操作:                                ║
║  • 查询/新增/修改/删除 库存商品            ║
║  • 查询/创建/取消/完成/删除 订单           ║
║  • 检索业务知识库 (退货/流程/规则等)       ║
║                                          ║
║  输入 help  查看示例                      ║
║  输入 clear 清空对话记忆                  ║
║  输入 stats 查看状态                      ║
║  输入 exit  退出                          ║
╚══════════════════════════════════════════╝
""")
    print("⏳ Agent 启动中，正在初始化向量库...\n")


def show_examples():
    print("""
📋 对话示例:

  【库存查询】
  你: 帮我看看有哪些库存
  你: 搜一下"键盘"相关的商品
  你: SKU001 的详细信息是什么

  【下单】
  你: 帮张三下单买 2 个 SKU001 (键盘)
  你: 我想创建订单：客户李四，买1个鼠标和3个支架

  【订单管理】
  你: 查一下张三的所有订单
  你: 帮我取消订单 ORD-20260710-0001
  你: 有哪些已确认的订单

  【知识检索】
  你: 退货流程是怎样的
  你: 库存不足时怎么办
""")


def main():
    print_banner()

    try:
        agent = InventoryAgent()
    except Exception as e:
        print(f"❌ Agent 启动失败: {e}")
        print("   请确保:")
        print("   1. .env 文件中已配置 OPENAI_API_KEY")
        print("   2. Java 后端已启动在 http://localhost:8080")
        sys.exit(1)

    print(f"✅ Agent 就绪 | 知识库: {agent.index_stats['doc_count']} 篇文档 | "
          f"工具: {agent.index_stats['tool_count']} 个 | "
          f"记忆窗口: {agent.index_stats['memory_window']} 轮\n")

    # ---- 交互循环 ----
    while True:
        try:
            user_input = input("🧑 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见!")
            break

        if not user_input:
            continue

        # 特殊命令
        if user_input.lower() in ("exit", "quit", "退出"):
            print("👋 再见!")
            break

        if user_input.lower() in ("help", "帮助"):
            show_examples()
            continue

        if user_input.lower() in ("clear", "清空"):
            agent.clear_memory()
            print("✅ 对话记忆已清空")
            continue

        if user_input.lower() == "stats":
            print(f"📊 状态: {agent.index_stats}")
            continue

        if user_input.lower() == "--rebuild":
            agent.rebuild_index()
            continue

        # Agent 处理
        try:
            print("🤖 助手: ", end="", flush=True)
            response = agent.chat(user_input)
            print(response)
            print()
        except Exception as e:
            print(f"\n❌ 出错了: {e}")
            print("   提示: 请检查 Java 后端是否正常运行\n")


if __name__ == "__main__":
    # 命令行参数 — 合并处理，避免重复初始化 Agent
    args = set(sys.argv[1:])
    rebuild_first = "--rebuild" in args
    show_stats = "--stats" in args

    if rebuild_first:
        print("🔨 强制重建向量索引...")
        tmp = InventoryAgent()
        tmp.rebuild_index()
        # 如果只需要重建 + 看统计，复用这个 agent
        if show_stats:
            print(f"📊 向量库统计: {tmp.index_stats}\n")
        print("✅ 重建完成，进入交互模式\n")
        del tmp   # 释放，让 main() 创建正式实例

    elif show_stats:
        tmp = InventoryAgent()
        print(f"📊 向量库统计: {tmp.index_stats}\n")
        del tmp

    main()
