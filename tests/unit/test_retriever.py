"""检索器单元测试"""

import pytest

from knowledge_forge.rag.retriever.base import RetrievedDocument
from knowledge_forge.rag.retriever.hybrid_retriever import HybridRetriever


def test_rrf_fusion():
    """测试 RRF 融合算法"""
    retriever = HybridRetriever(
        vector_retriever=None,
        bm25_retriever=None,
    )

    vector_results = [
        RetrievedDocument(id="1", content="doc1", score=0.9, source="vector"),
        RetrievedDocument(id="2", content="doc2", score=0.8, source="vector"),
        RetrievedDocument(id="3", content="doc3", score=0.7, source="vector"),
    ]

    bm25_results = [
        RetrievedDocument(id="2", content="doc2", score=3.5, source="bm25"),
        RetrievedDocument(id="4", content="doc4", score=3.0, source="bm25"),
        RetrievedDocument(id="1", content="doc1", score=2.5, source="bm25"),
    ]

    fused = retriever._rrf_fusion(vector_results, bm25_results)

    # 文档1和2同时出现在两路结果中，融合分数应更高
    assert "1" in fused
    assert "2" in fused
    assert "3" in fused
    assert "4" in fused
    # 文档2同时被两路检索到，分数应该比只被一路检索到的文档3高
    assert fused["2"].score > fused["3"].score
