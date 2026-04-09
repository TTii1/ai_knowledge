"""知识库 MCP 资源 — 独立函数版本

返回知识库概览信息：文档数、Chunk 数、最近更新时间等。
"""

import json
import logging

logger = logging.getLogger(__name__)


async def knowledge_overview(knowledge_base: str) -> str:
    """返回知识库概览信息

    Args:
        knowledge_base: 知识库名称

    Returns:
        JSON 格式的知识库概览
    """
    from knowledge_forge.api.deps import get_metadata_store

    metadata_store = get_metadata_store()
    if metadata_store is None:
        return json.dumps({"error": "元数据存储未初始化"}, ensure_ascii=False)

    try:
        if metadata_store._session_factory is None:
            await metadata_store.connect()

        kb_info = await metadata_store.get_knowledge_base(knowledge_base)
        if kb_info is None:
            return json.dumps(
                {"error": f"知识库 '{knowledge_base}' 不存在"},
                ensure_ascii=False,
            )

        # 获取文档统计
        doc_stats = await metadata_store.list_documents(
            knowledge_base=knowledge_base, page=1, page_size=1,
        )

        overview = {
            "name": kb_info["name"],
            "description": kb_info.get("description", ""),
            "document_count": kb_info.get("document_count", 0),
            "chunk_count": kb_info.get("chunk_count", 0),
            "embedding_model": kb_info.get("embedding_model", ""),
            "embedding_dimension": kb_info.get("embedding_dimension", 0),
            "chunk_size": kb_info.get("chunk_size", 800),
            "chunk_overlap": kb_info.get("chunk_overlap", 100),
            "is_active": kb_info.get("is_active", True),
            "created_at": kb_info.get("created_at"),
            "updated_at": kb_info.get("updated_at"),
            "total_documents_in_db": doc_stats.get("total", 0),
        }

        return json.dumps(overview, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error("知识库概览查询失败: %s", str(e), exc_info=True)
        return json.dumps({"error": str(e)}, ensure_ascii=False)
