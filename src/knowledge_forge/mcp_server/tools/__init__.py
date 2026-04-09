"""MCP 工具定义

工具已迁移到 server.py 中使用 FastMCP @mcp.tool() 装饰器定义。
保留此模块用于兼容和独立工具函数导出。
"""

from knowledge_forge.mcp_server.tools.knowledge_query import knowledge_query
from knowledge_forge.mcp_server.tools.document_search import document_search
from knowledge_forge.mcp_server.tools.knowledge_manage import (
    knowledge_list,
    session_create,
)

__all__ = [
    "knowledge_query",
    "document_search",
    "knowledge_list",
    "session_create",
]
