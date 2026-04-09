"""MCP 资源定义

资源已迁移到 server.py 中使用 FastMCP @mcp.resource() 装饰器定义。
保留此模块用于兼容和独立资源函数导出。
"""

from knowledge_forge.mcp_server.resources.knowledge_base import knowledge_overview

__all__ = ["knowledge_overview"]
