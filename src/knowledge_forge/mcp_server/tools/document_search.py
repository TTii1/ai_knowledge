"""document_search MCP 工具 — 独立函数版本

搜索知识库中的相关文档片段，返回原始内容（不经过 LLM 生成）。
"""

import logging

logger = logging.getLogger(__name__)


async def document_search(
    query: str,
    knowledge_base: str = "default",
    top_k: int = 10,
) -> str:
    """搜索知识库中的相关文档片段

    Args:
        query: 搜索关键词
        knowledge_base: 知识库名称
        top_k: 返回文档片段数

    Returns:
        匹配的文档片段列表
    """
    from knowledge_forge.api.deps import get_retriever

    try:
        retriever = get_retriever()
        documents = await retriever.retrieve(
            query=query,
            top_k=top_k,
            knowledge_base=knowledge_base,
        )

        if not documents:
            return f"未找到与 '{query}' 相关的文档片段。"

        output_parts = [f"找到 {len(documents)} 个相关文档片段：\n"]
        for i, doc in enumerate(documents, 1):
            heading = " > ".join(doc.heading_chain) if doc.heading_chain else ""
            source = doc.metadata.get("source_file", "未知来源")
            score = doc.score or 0

            output_parts.append(f"### [{i}] {source}")
            if heading:
                output_parts.append(f"**路径**: {heading}")
            output_parts.append(f"**相关度**: {score:.4f}")
            output_parts.append(f"\n{doc.content}\n")

        return "\n".join(output_parts)

    except Exception as e:
        logger.error("文档搜索失败: %s", str(e), exc_info=True)
        return f"搜索失败: {str(e)}"
