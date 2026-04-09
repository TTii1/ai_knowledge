"""RAG Engine 单元测试"""

import pytest

from knowledge_forge.rag.engine import RAGEngine, RAGResult
from knowledge_forge.rag.query_rewriter import RewriteStrategy
from knowledge_forge.rag.retriever.base import RetrievedDocument
from knowledge_forge.rag.context_builder import ContextBuilder


def test_rag_result_dataclass():
    """测试 RAGResult 数据类"""
    result = RAGResult(query="测试查询")
    assert result.query == "测试查询"
    assert result.resolved_query == ""
    assert result.rewritten_queries == []
    assert result.retrieved_documents == []
    assert result.answer == ""
    assert result.sources == []
    assert result.latency_ms == 0.0


def test_rag_engine_init():
    """测试 RAG Engine 初始化"""
    engine = RAGEngine()
    assert engine.query_rewriter is None
    assert engine.retriever is None
    assert engine.reranker is None
    assert engine.context_builder is None
    assert engine.generator is None
    assert engine.rewrite_strategy == RewriteStrategy.LLM_REWRITE
    assert engine.retrieval_top_k == 20
    assert engine.rerank_top_k == 5


@pytest.mark.asyncio
async def test_rag_engine_no_services():
    """测试无服务时的 RAG 引擎"""
    engine = RAGEngine()
    result = await engine.answer("测试查询")

    assert isinstance(result, RAGResult)
    assert result.query == "测试查询"
    assert result.resolved_query == "测试查询"
    assert result.answer == "未配置 LLM 生成器，无法生成回答。"


@pytest.mark.asyncio
async def test_rag_engine_with_context_builder():
    """测试带 ContextBuilder 的 RAG 引擎（不含 LLM）"""
    context_builder = ContextBuilder(max_tokens=4096, model="gpt-4o-mini")
    engine = RAGEngine(
        context_builder=context_builder,
        generator=None,
    )
    result = await engine.answer("测试查询")

    assert isinstance(result, RAGResult)
    # 无检索结果，上下文为空
    assert result.context == ""


def test_build_sources():
    """测试来源构建"""
    docs = [
        RetrievedDocument(
            id="1",
            content="文档内容1，这是一段测试文本",
            score=0.95,
            source="vector",
            heading_chain=["章节1", "1.1 小节"],
            metadata={"source_file": "test.md", "document_id": "doc-1"},
        ),
        RetrievedDocument(
            id="2",
            content="文档内容2，这也是测试文本",
            score=0.85,
            source="bm25",
            heading_chain=[],
            metadata={"source_file": "test2.md", "document_id": "doc-2"},
        ),
    ]

    sources = RAGEngine._build_sources(docs)

    assert len(sources) == 2
    assert sources[0]["document_id"] == "doc-1"
    assert sources[0]["heading_chain"] == ["章节1", "1.1 小节"]
    assert sources[0]["score"] == 0.95
    assert "source_file" in sources[0]
    assert "content_preview" in sources[0]


def test_build_sources_dedup():
    """测试来源去重"""
    docs = [
        RetrievedDocument(
            id="1",
            content="文档内容1",
            score=0.95,
            source="vector",
            metadata={"source_file": "test.md", "document_id": "doc-1"},
        ),
        # 同一个文件+同id的重复
        RetrievedDocument(
            id="1",
            content="文档内容1",
            score=0.85,
            source="bm25",
            metadata={"source_file": "test.md", "document_id": "doc-1"},
        ),
    ]

    sources = RAGEngine._build_sources(docs)
    assert len(sources) == 1


def test_context_builder():
    """测试 ContextBuilder"""
    builder = ContextBuilder(max_tokens=4096, model="gpt-4o-mini")

    docs = [
        RetrievedDocument(
            id="1",
            content="这是第一个文档片段的内容",
            score=0.95,
            source="vector",
            heading_chain=["第1章", "1.1 概述"],
        ),
        RetrievedDocument(
            id="2",
            content="这是第二个文档片段的内容",
            score=0.85,
            source="bm25",
            heading_chain=[],
        ),
    ]

    context = builder.build(query="测试查询", documents=docs)

    assert "文档片段 1" in context
    assert "文档片段 2" in context
    assert "第1章 > 1.1 概述" in context


def test_context_builder_empty():
    """测试空文档列表的上下文构建"""
    builder = ContextBuilder(max_tokens=4096, model="gpt-4o-mini")
    context = builder.build(query="测试查询", documents=[])
    assert context == ""


def test_context_builder_system_prompt():
    """测试系统 prompt 构建"""
    builder = ContextBuilder()
    prompt = builder.build_system_prompt()
    assert "知识库问答助手" in prompt
    assert "基于提供的文档片段回答" in prompt
