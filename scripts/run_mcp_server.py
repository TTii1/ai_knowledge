"""MCP Server 启动脚本"""

import asyncio
import sys
import os

# 将 src 目录加入 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from knowledge_forge.mcp_server.server import main as mcp_main


if __name__ == "__main__":
    asyncio.run(mcp_main())
