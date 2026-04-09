"""knowledge_manage MCP 工具 — 独立函数版本

包含 knowledge_list 和 session_create 工具。
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


async def knowledge_list() -> str:
    """列出所有可用的知识库及其基本信息"""
    from knowledge_forge.api.deps import get_metadata_store

    metadata_store = get_metadata_store()
    if metadata_store is None:
        return "错误：元数据存储未初始化。"

    try:
        if metadata_store._session_factory is None:
            await metadata_store.connect()

        result = await metadata_store.list_knowledge_bases(page=1, page_size=50)
        items = result.get("items", [])

        if not items:
            return "当前没有可用的知识库。请先通过 API 创建知识库并上传文档。"

        output_parts = [f"共有 {result.get('total', 0)} 个知识库：\n"]
        for kb in items:
            status = "✅ 启用" if kb.get("is_active") else "❌ 停用"
            output_parts.append(
                f"- **{kb['name']}** {status}\n"
                f"  描述: {kb.get('description', '无')}\n"
                f"  文档数: {kb.get('document_count', 0)} | "
                f"Chunk 数: {kb.get('chunk_count', 0)}\n"
                f"  Embedding: {kb.get('embedding_model', 'N/A')} "
                f"(dim={kb.get('embedding_dimension', 0)})\n"
                f"  Chunk 配置: size={kb.get('chunk_size', 800)}, "
                f"overlap={kb.get('chunk_overlap', 100)}\n"
            )

        return "\n".join(output_parts)

    except Exception as e:
        logger.error("知识库列表查询失败: %s", str(e), exc_info=True)
        return f"查询失败: {str(e)}"


async def session_create(knowledge_base: str = "default") -> str:
    """创建新的对话会话，返回 session_id

    Args:
        knowledge_base: 知识库名称

    Returns:
        session_id，用于后续对话
    """
    from knowledge_forge.api.deps import get_conversation_memory

    memory = get_conversation_memory()
    if memory is None:
        return "错误：对话记忆管理器未初始化。"

    try:
        session_id = await memory.create_session(knowledge_base=knowledge_base)
        return (
            f"会话创建成功！\n\n"
            f"**Session ID**: `{session_id}`\n\n"
            f"在后续的 knowledge_query 调用中传入此 session_id "
            f"即可保持多轮对话上下文。\n\n"
            f"示例: knowledge_query(query='RAG是什么', session_id='{session_id}')"
        )
    except Exception as e:
        logger.error("会话创建失败: %s", str(e), exc_info=True)
        return f"会话创建失败: {str(e)}"
