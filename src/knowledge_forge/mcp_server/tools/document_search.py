"""document_search MCP 工具"""

import logging

logger = logging.getLogger(__name__)


async def document_search(query: str, knowledge_base: str = "default", top_k: int = 10) -> list[dict]:
    """搜索知识库中的相关文档片段"""
    # TODO: 接入向量检索
    return []
