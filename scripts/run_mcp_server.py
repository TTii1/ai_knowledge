"""MCP Server 启动脚本

启动方式：
    python scripts/run_mcp_server.py                         # stdio 模式
    python scripts/run_mcp_server.py --transport sse         # SSE 模式
    python scripts/run_mcp_server.py --transport sse --port 9000
"""

import sys
import os

# 将 src 目录加入 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from knowledge_forge.mcp_server.server import main


if __name__ == "__main__":
    main()
