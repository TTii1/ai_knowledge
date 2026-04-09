"""MCP Server 主入口

将知识库查询能力封装为标准 MCP 工具，供 Cursor/Claude 等工具调用。
"""

import logging
from typing import Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server

from knowledge_forge.config import get_settings

logger = logging.getLogger(__name__)

# 创建 MCP Server 实例
app = Server("knowledge-forge")


@app.list_tools()
async def list_tools() -> list[dict]:
    """列出所有可用工具"""
    return [
        {
            "name": "knowledge_query",
            "description": "在知识库中查询相关内容并生成回答。当用户需要查询文档、知识库中的信息时使用此工具。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "查询问题",
                    },
                    "knowledge_base": {
                        "type": "string",
                        "description": "知识库名称",
                        "default": "default",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回结果数",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "document_search",
            "description": "搜索知识库中的相关文档片段，返回原始内容（不经过LLM生成）。当需要获取原始文档内容、查看具体段落时使用此工具。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词",
                    },
                    "knowledge_base": {
                        "type": "string",
                        "description": "知识库名称",
                        "default": "default",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回文档片段数",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "knowledge_list",
            "description": "列出所有可用的知识库及其基本信息。",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "session_create",
            "description": "创建新的对话会话，返回 session_id。用于需要多轮对话上下文的场景。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "knowledge_base": {
                        "type": "string",
                        "description": "知识库名称",
                        "default": "default",
                    },
                },
            },
        },
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[dict]:
    """处理工具调用"""
    logger.info("MCP 工具调用: name=%s, args=%s", name, arguments)

    if name == "knowledge_query":
        return await _knowledge_query(
            query=arguments["query"],
            knowledge_base=arguments.get("knowledge_base", "default"),
            top_k=arguments.get("top_k", 5),
        )
    elif name == "document_search":
        return await _document_search(
            query=arguments["query"],
            knowledge_base=arguments.get("knowledge_base", "default"),
            top_k=arguments.get("top_k", 10),
        )
    elif name == "knowledge_list":
        return await _knowledge_list()
    elif name == "session_create":
        return await _session_create(
            knowledge_base=arguments.get("knowledge_base", "default"),
        )
    else:
        return [{"type": "text", "text": f"未知工具: {name}"}]


async def _knowledge_query(query: str, knowledge_base: str, top_k: int) -> list[dict]:
    """知识库查询（含 LLM 生成）"""
    # TODO: 实现完整 RAG 流程
    return [{
        "type": "text",
        "text": f"[待实现] 知识库查询: query='{query}', kb='{knowledge_base}', top_k={top_k}",
    }]


async def _document_search(query: str, knowledge_base: str, top_k: int) -> list[dict]:
    """文档片段搜索（原始检索结果）"""
    # TODO: 实现向量检索
    return [{
        "type": "text",
        "text": f"[待实现] 文档搜索: query='{query}', kb='{knowledge_base}', top_k={top_k}",
    }]


async def _knowledge_list() -> list[dict]:
    """知识库列表"""
    # TODO: 实现知识库列表查询
    return [{
        "type": "text",
        "text": "[待实现] 知识库列表",
    }]


async def _session_create(knowledge_base: str) -> list[dict]:
    """创建对话会话"""
    # TODO: 实现会话创建
    return [{
        "type": "text",
        "text": f"[待实现] 会话创建: kb='{knowledge_base}'",
    }]


async def main():
    """启动 MCP Server"""
    settings = get_settings()

    async with stdio_server() as (read_stream, write_stream):
        logger.info("MCP Server 启动: %s", settings.app_name)
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
