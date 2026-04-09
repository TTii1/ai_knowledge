"""MCP Server 单元测试

测试 MCP Server 的工具定义和核心逻辑。
由于工具函数内部使用延迟导入（from ... import ...），
我们需要 patch 源模块 knowledge_forge.api.deps 上的属性。
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import knowledge_forge.api.deps as deps_mod


# ============ knowledge_query 测试 ============

@pytest.mark.asyncio
async def test_knowledge_query_no_engine():
    """RAGEngine 未初始化时返回错误提示"""
    with patch.object(deps_mod, "get_rag_engine", return_value=None), \
         patch.object(deps_mod, "get_conversation_memory", return_value=None):
        from knowledge_forge.mcp_server.tools.knowledge_query import knowledge_query
        result = await knowledge_query(query="测试问题")
        assert "错误" in result or "RAG" in result


@pytest.mark.asyncio
async def test_knowledge_query_with_engine():
    """RAGEngine 正常时返回答案"""
    mock_result = MagicMock()
    mock_result.answer = "RAG 是检索增强生成技术"
    mock_result.sources = [
        {
            "source_file": "rag_intro.pdf",
            "heading_chain": ["第1章", "1.1 概述"],
            "score": 0.95,
        }
    ]
    mock_result.latency_ms = 150.0

    mock_engine = AsyncMock()
    mock_engine.answer.return_value = mock_result

    with patch.object(deps_mod, "get_rag_engine", return_value=mock_engine), \
         patch.object(deps_mod, "get_conversation_memory", return_value=None):
        from knowledge_forge.mcp_server.tools.knowledge_query import knowledge_query
        result = await knowledge_query(query="RAG是什么")
        assert "RAG 是检索增强生成技术" in result
        assert "rag_intro.pdf" in result


@pytest.mark.asyncio
async def test_knowledge_query_with_session():
    """带 session_id 时获取对话历史"""
    mock_result = MagicMock()
    mock_result.answer = "回答"
    mock_result.sources = []
    mock_result.latency_ms = 100.0

    mock_engine = AsyncMock()
    mock_engine.answer.return_value = mock_result

    mock_memory = AsyncMock()
    mock_memory.get_session.return_value = {"messages": [{"role": "user", "content": "之前的问题"}]}
    mock_memory.add_message = AsyncMock()

    with patch.object(deps_mod, "get_rag_engine", return_value=mock_engine), \
         patch.object(deps_mod, "get_conversation_memory", return_value=mock_memory):
        from knowledge_forge.mcp_server.tools.knowledge_query import knowledge_query
        result = await knowledge_query(query="后续问题", session_id="test-session")
        mock_memory.get_session.assert_called_once_with("test-session")
        assert mock_memory.add_message.call_count == 2


# ============ document_search 测试 ============

@pytest.mark.asyncio
async def test_document_search_no_results():
    """检索无结果时返回提示"""
    mock_retriever = AsyncMock()
    mock_retriever.retrieve.return_value = []

    with patch.object(deps_mod, "get_retriever", return_value=mock_retriever):
        from knowledge_forge.mcp_server.tools.document_search import document_search
        result = await document_search(query="不存在的内容")
        assert "未找到" in result


@pytest.mark.asyncio
async def test_document_search_with_results():
    """检索有结果时返回格式化内容"""
    mock_doc = MagicMock()
    mock_doc.content = "这是检索到的文档内容"
    mock_doc.heading_chain = ["第1章", "1.1 概述"]
    mock_doc.metadata = {"source_file": "test.pdf"}
    mock_doc.score = 0.92

    mock_retriever = AsyncMock()
    mock_retriever.retrieve.return_value = [mock_doc]

    with patch.object(deps_mod, "get_retriever", return_value=mock_retriever):
        from knowledge_forge.mcp_server.tools.document_search import document_search
        result = await document_search(query="测试查询")
        assert "1 个相关文档片段" in result
        assert "test.pdf" in result
        assert "这是检索到的文档内容" in result


# ============ knowledge_list 测试 ============

@pytest.mark.asyncio
async def test_knowledge_list_no_store():
    """MetadataStore 未初始化时返回错误"""
    with patch.object(deps_mod, "get_metadata_store", return_value=None):
        from knowledge_forge.mcp_server.tools.knowledge_manage import knowledge_list
        result = await knowledge_list()
        assert "错误" in result


@pytest.mark.asyncio
async def test_knowledge_list_empty():
    """没有知识库时返回提示"""
    mock_store = AsyncMock()
    mock_store._session_factory = MagicMock()  # 已连接
    mock_store.list_knowledge_bases.return_value = {"items": [], "total": 0}

    with patch.object(deps_mod, "get_metadata_store", return_value=mock_store):
        from knowledge_forge.mcp_server.tools.knowledge_manage import knowledge_list
        result = await knowledge_list()
        assert "没有可用的知识库" in result


@pytest.mark.asyncio
async def test_knowledge_list_with_items():
    """有知识库时返回列表"""
    mock_store = AsyncMock()
    mock_store._session_factory = MagicMock()
    mock_store.list_knowledge_bases.return_value = {
        "items": [
            {
                "name": "test-kb",
                "description": "测试知识库",
                "document_count": 5,
                "chunk_count": 100,
                "is_active": True,
                "embedding_model": "text-embedding-3-small",
                "embedding_dimension": 1536,
                "chunk_size": 800,
                "chunk_overlap": 100,
            }
        ],
        "total": 1,
    }

    with patch.object(deps_mod, "get_metadata_store", return_value=mock_store):
        from knowledge_forge.mcp_server.tools.knowledge_manage import knowledge_list
        result = await knowledge_list()
        assert "test-kb" in result
        assert "5" in result  # document_count


# ============ session_create 测试 ============

@pytest.mark.asyncio
async def test_session_create_no_memory():
    """ConversationMemory 未初始化时返回错误"""
    with patch.object(deps_mod, "get_conversation_memory", return_value=None):
        from knowledge_forge.mcp_server.tools.knowledge_manage import session_create
        result = await session_create()
        assert "错误" in result


@pytest.mark.asyncio
async def test_session_create_success():
    """成功创建会话"""
    mock_memory = AsyncMock()
    mock_memory.create_session.return_value = "session-abc123"

    with patch.object(deps_mod, "get_conversation_memory", return_value=mock_memory):
        from knowledge_forge.mcp_server.tools.knowledge_manage import session_create
        result = await session_create(knowledge_base="test-kb")
        assert "session-abc123" in result
        assert "成功" in result


# ============ knowledge_overview 资源测试 ============

@pytest.mark.asyncio
async def test_knowledge_overview_no_store():
    """MetadataStore 未初始化"""
    with patch.object(deps_mod, "get_metadata_store", return_value=None):
        from knowledge_forge.mcp_server.resources.knowledge_base import knowledge_overview
        result = await knowledge_overview("test-kb")
        data = json.loads(result)
        assert "error" in data


@pytest.mark.asyncio
async def test_knowledge_overview_not_found():
    """知识库不存在"""
    mock_store = AsyncMock()
    mock_store._session_factory = MagicMock()
    mock_store.get_knowledge_base.return_value = None

    with patch.object(deps_mod, "get_metadata_store", return_value=mock_store):
        from knowledge_forge.mcp_server.resources.knowledge_base import knowledge_overview
        result = await knowledge_overview("nonexistent")
        data = json.loads(result)
        assert "error" in data


@pytest.mark.asyncio
async def test_knowledge_overview_success():
    """知识库概览正常返回"""
    mock_store = AsyncMock()
    mock_store._session_factory = MagicMock()
    mock_store.get_knowledge_base.return_value = {
        "name": "test-kb",
        "description": "测试",
        "document_count": 10,
        "chunk_count": 200,
        "embedding_model": "text-embedding-3-small",
        "embedding_dimension": 1536,
        "chunk_size": 800,
        "chunk_overlap": 100,
        "is_active": True,
        "created_at": "2026-04-09T00:00:00",
        "updated_at": "2026-04-09T12:00:00",
    }
    mock_store.list_documents.return_value = {"total": 10}

    with patch.object(deps_mod, "get_metadata_store", return_value=mock_store):
        from knowledge_forge.mcp_server.resources.knowledge_base import knowledge_overview
        result = await knowledge_overview("test-kb")
        data = json.loads(result)
        assert data["name"] == "test-kb"
        assert data["document_count"] == 10


# ============ FastMCP 实例测试 ============

def test_mcp_instance_exists():
    """MCP 实例已正确创建"""
    from knowledge_forge.mcp_server.server import mcp
    assert mcp is not None
    assert mcp.name == "knowledge-forge"


def test_mcp_tools_registered():
    """MCP 工具已注册"""
    from knowledge_forge.mcp_server.server import mcp
    assert hasattr(mcp, "_tool_manager")
