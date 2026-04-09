"""Reranker 单元测试"""

import pytest

from knowledge_forge.rag.retriever.base import RetrievedDocument


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
