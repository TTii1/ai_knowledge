"""Reranker 单元测试"""

import pytest

from knowledge_forge.rag.reranker import Reranker
from knowledge_forge.rag.retriever.base import RetrievedDocument


def test_reranker_init():
    """测试 Reranker 初始化"""
    reranker = Reranker()
    assert reranker.model_name == "BAAI/bge-reranker-v2-m3"
    assert not reranker._load_attempted
    assert not reranker._load_failed


def test_reranker_not_available_without_model():
    """测试无模型时 Reranker 不可用"""
    reranker = Reranker()
    # 模拟加载失败
    reranker._load_failed = True
    reranker._load_attempted = True
    assert not reranker.is_available


@pytest.mark.asyncio
async def test_reranker_returns_top_k_on_failure():
    """测试 Reranker 失败时降级返回 top_k"""
    reranker = Reranker()
    reranker._load_failed = True
    reranker._load_attempted = True

    docs = [
        RetrievedDocument(id="1", content="doc1", score=0.9, source="vector"),
        RetrievedDocument(id="2", content="doc2", score=0.8, source="vector"),
        RetrievedDocument(id="3", content="doc3", score=0.7, source="vector"),
    ]

    result = await reranker.rerank("test query", docs, top_k=2)
    assert len(result) == 2
    assert result[0].id == "1"


@pytest.mark.asyncio
async def test_reranker_empty_docs():
    """测试空文档列表"""
    reranker = Reranker()
    result = await reranker.rerank("test query", [], top_k=5)
    assert result == []


def test_retrieved_document_dataclass():
    """测试 RetrievedDocument 数据类"""
    doc = RetrievedDocument(
        id="test-1",
        content="测试内容",
        score=0.95,
        source="vector",
        heading_chain=["标题1", "标题2"],
        metadata={"page": 1},
    )

    assert doc.id == "test-1"
    assert doc.content == "测试内容"
    assert doc.score == 0.95
    assert doc.source == "vector"
