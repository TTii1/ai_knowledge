"""knowledge_query MCP 工具"""

import logging

logger = logging.getLogger(__name__)


async def knowledge_query(query: str, knowledge_base: str = "default", top_k: int = 5) -> str:
    """在知识库中查询相关内容并生成回答"""
    # TODO: 接入完整 RAG 流程
    return f"知识库查询结果: query='{query}', kb='{knowledge_base}'"
