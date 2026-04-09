"""MCP Server 模块

提供 KnowledgeForge 的 MCP 协议接口，供 Cursor/Claude 等工具调用。

启动方式:
    python -m knowledge_forge.mcp_server                  # stdio 模式
    python -m knowledge_forge.mcp_server --transport sse  # SSE 模式
"""

from knowledge_forge.mcp_server.server import mcp, main

__all__ = ["mcp", "main"]
