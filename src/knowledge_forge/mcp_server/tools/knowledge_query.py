"""knowledge_query MCP 工具 — 独立函数版本

在知识库中查询相关内容并生成回答（含完整 RAG 流水线）。
主要逻辑在 server.py 的 @mcp.tool() 装饰器中，此模块提供可复用的独立函数。
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


async def knowledge_query(
    query: str,
    knowledge_base: str = "default",
    top_k: int = 5,
    session_id: Optional[str] = None,
) -> str:
    """在知识库中查询相关内容并生成回答

    Args:
        query: 查询问题
        knowledge_base: 知识库名称
        top_k: 返回结果数
        session_id: 会话 ID（可选，用于多轮对话）

    Returns:
        生成的回答，含来源引用
    """
    from knowledge_forge.api.deps import get_rag_engine, get_conversation_memory

    rag_engine = get_rag_engine()
    if rag_engine is None:
        return "错误：RAG 引擎未初始化，请检查配置。"

    # 获取对话历史
    conversation_history = None
    if session_id:
        memory = get_conversation_memory()
        if memory:
            history_data = await memory.get_session(session_id)
            if history_data:
                conversation_history = history_data.get("messages", [])

    # 执行 RAG 问答
    result = await rag_engine.answer(
        query=query,
        knowledge_base=knowledge_base,
        conversation_history=conversation_history,
        stream=False,
    )

    # 保存对话
    if session_id:
        memory = get_conversation_memory()
        if memory:
            await memory.add_message(session_id, "user", query)
            await memory.add_message(session_id, "assistant", result.answer)

    # 构建返回
    output_parts = [result.answer]
    if result.sources:
        output_parts.append("\n\n---\n**参考来源：**")
        for i, source in enumerate(result.sources, 1):
            source_file = source.get("source_file", "未知文档")
            heading = " > ".join(source.get("heading_chain", []))
            score = source.get("score", 0)
            heading_str = f" ({heading})" if heading else ""
            output_parts.append(f"{i}. {source_file}{heading_str} [相关度: {score:.2f}]")

    if result.latency_ms:
        output_parts.append(f"\n⏱ 耗时: {result.latency_ms:.0f}ms")

    return "\n".join(output_parts)
