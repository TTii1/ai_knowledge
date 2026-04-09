"""MCP Server 主入口 — 基于 FastMCP 高级 API

将知识库查询能力封装为标准 MCP 工具，供 Cursor/Claude 等工具调用。

启动方式：
  1. stdio 模式（供 Cursor/Claude 调用）：
     python -m knowledge_forge.mcp_server
  2. SSE 模式（供 HTTP 客户端调用）：
     python -m knowledge_forge.mcp_server --transport sse --port 9000
"""

import argparse
import asyncio
import json
import logging
from typing import Optional

from mcp.server.fastmcp import FastMCP

from knowledge_forge.config import get_settings

logger = logging.getLogger(__name__)

# ============ FastMCP 实例 ============

mcp = FastMCP(
    "knowledge-forge",
    instructions="""KnowledgeForge 智能知识库助手

你可以通过以下工具与知识库交互：
- knowledge_query: 在知识库中查询并生成回答（含 RAG 流水线）
- document_search: 搜索原始文档片段（不经过 LLM 生成）
- knowledge_list: 列出所有可用的知识库
- session_create: 创建多轮对话会话

资源：
- knowledge://{kb_name}/overview: 知识库概览信息
""",
)


# ============ 延迟初始化的 RAG 组件 ============

_rag_engine = None
_metadata_store = None
_conversation_memory = None
_vector_store = None


async def _get_rag_engine():
    """延迟获取 RAGEngine 实例"""
    global _rag_engine
    if _rag_engine is None:
        from knowledge_forge.api.deps import get_rag_engine
        _rag_engine = get_rag_engine()
    return _rag_engine


async def _get_metadata_store():
    """延迟获取 MetadataStore 实例"""
    global _metadata_store
    if _metadata_store is None:
        from knowledge_forge.api.deps import get_metadata_store
        _metadata_store = get_metadata_store()
    return _metadata_store


async def _get_conversation_memory():
    """延迟获取 ConversationMemory 实例"""
    global _conversation_memory
    if _conversation_memory is None:
        from knowledge_forge.api.deps import get_conversation_memory
        _conversation_memory = get_conversation_memory()
    return _conversation_memory


async def _get_vector_store():
    """延迟获取 VectorStore 实例"""
    global _vector_store
    if _vector_store is None:
        from knowledge_forge.api.deps import get_vector_store
        _vector_store = get_vector_store()
    return _vector_store


# ============ MCP 工具 ============

@mcp.tool()
async def knowledge_query(
    query: str,
    knowledge_base: str = "default",
    top_k: int = 5,
    session_id: Optional[str] = None,
) -> str:
    """在知识库中查询相关内容并生成回答。

    当用户需要查询文档、知识库中的信息时使用此工具。
    支持多轮对话上下文（需提供 session_id）。

    Args:
        query: 查询问题
        knowledge_base: 知识库名称，默认 "default"
        top_k: 返回结果数，默认 5
        session_id: 会话 ID（用于多轮对话上下文），可选

    Returns:
        知识库生成的回答，包含引用来源
    """
    logger.info("MCP knowledge_query: query='%s', kb='%s', top_k=%d, session=%s",
                query[:50], knowledge_base, top_k, session_id)

    rag_engine = await _get_rag_engine()
    if rag_engine is None:
        return "错误：RAG 引擎未初始化，请检查配置（OPENAI_API_KEY 等）。"

    # 获取对话历史
    conversation_history = None
    if session_id:
        memory = await _get_conversation_memory()
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

    # 如果有 session_id，保存对话
    if session_id:
        memory = await _get_conversation_memory()
        if memory:
            await memory.add_message(session_id, "user", query)
            await memory.add_message(session_id, "assistant", result.answer)

    # 构建返回结果
    output_parts = [result.answer]

    # 添加来源引用
    if result.sources:
        output_parts.append("\n\n---\n**参考来源：**")
        for i, source in enumerate(result.sources, 1):
            source_file = source.get("source_file", "未知文档")
            heading = " > ".join(source.get("heading_chain", []))
            score = source.get("score", 0)
            heading_str = f" ({heading})" if heading else ""
            output_parts.append(f"{i}. {source_file}{heading_str} [相关度: {score:.2f}]")

    # 添加元数据
    if result.latency_ms:
        output_parts.append(f"\n⏱ 耗时: {result.latency_ms:.0f}ms")

    return "\n".join(output_parts)


@mcp.tool()
async def document_search(
    query: str,
    knowledge_base: str = "default",
    top_k: int = 10,
) -> str:
    """搜索知识库中的相关文档片段，返回原始内容（不经过 LLM 生成）。

    当需要获取原始文档内容、查看具体段落时使用此工具。

    Args:
        query: 搜索关键词
        knowledge_base: 知识库名称，默认 "default"
        top_k: 返回文档片段数，默认 10

    Returns:
        匹配的文档片段列表，包含来源和相似度分数
    """
    logger.info("MCP document_search: query='%s', kb='%s', top_k=%d",
                query[:50], knowledge_base, top_k)

    # 通过检索器直接检索（不走 RAGEngine 完整流水线）
    try:
        from knowledge_forge.api.deps import get_retriever, get_embedding_service

        retriever = get_retriever()
        embedding_service = get_embedding_service()

        # 生成 query 向量
        query_embeddings = await embedding_service.embed_texts([query])
        query_embedding = query_embeddings[0]

        # 检索
        documents = await retriever.retrieve(
            query=query,
            top_k=top_k,
            knowledge_base=knowledge_base,
        )

        if not documents:
            return f"未找到与 '{query}' 相关的文档片段。"

        # 格式化输出
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


@mcp.tool()
async def knowledge_list() -> str:
    """列出所有可用的知识库及其基本信息。

    Returns:
        知识库列表，包含名称、描述、文档数、Chunk 数等
    """
    logger.info("MCP knowledge_list")

    metadata_store = await _get_metadata_store()
    if metadata_store is None:
        return "错误：元数据存储未初始化。"

    try:
        # 确保连接
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


@mcp.tool()
async def session_create(knowledge_base: str = "default") -> str:
    """创建新的对话会话，返回 session_id。

    用于需要多轮对话上下文的场景。创建后，在后续的
    knowledge_query 调用中传入返回的 session_id 即可
    保持对话上下文连续性。

    Args:
        knowledge_base: 知识库名称，默认 "default"

    Returns:
        session_id，用于后续对话
    """
    logger.info("MCP session_create: kb='%s'", knowledge_base)

    memory = await _get_conversation_memory()
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


# ============ MCP 资源 ============

@mcp.resource("knowledge://{knowledge_base}/overview")
async def knowledge_overview(knowledge_base: str) -> str:
    """返回知识库的概览信息：文档数、Chunk 数、最近更新时间等"""
    logger.info("MCP resource: knowledge_overview kb='%s'", knowledge_base)

    metadata_store = await _get_metadata_store()
    if metadata_store is None:
        return json.dumps({"error": "元数据存储未初始化"})

    try:
        if metadata_store._session_factory is None:
            await metadata_store.connect()

        kb_info = await metadata_store.get_knowledge_base(knowledge_base)
        if kb_info is None:
            return json.dumps({"error": f"知识库 '{knowledge_base}' 不存在"}, ensure_ascii=False)

        # 获取文档统计
        doc_stats = await metadata_store.list_documents(
            knowledge_base=knowledge_base, page=1, page_size=1
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


# ============ MCP Prompt ============

@mcp.prompt()
async def rag_qa_prompt(query: str, knowledge_base: str = "default") -> str:
    """RAG 问答 Prompt 模板

    用于生成结构化的知识库查询提示。
    """
    return f"""请基于知识库 "{knowledge_base}" 回答以下问题：

{query}

要求：
1. 回答必须基于检索到的文档内容，不要编造信息
2. 如果知识库中没有相关信息，请明确说明
3. 引用具体的文档来源
4. 回答要完整、准确、有条理"""


# ============ 启动入口 ============

def main():
    """启动 MCP Server"""
    parser = argparse.ArgumentParser(description="KnowledgeForge MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="传输协议（默认 stdio）",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="SSE/HTTP 模式的监听地址",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9000,
        help="SSE/HTTP 模式的监听端口",
    )
    args = parser.parse_args()

    # 配置日志
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("启动 KnowledgeForge MCP Server (transport=%s)", args.transport)

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    elif args.transport == "streamable-http":
        mcp.run(transport="streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
