"""检索器单元测试"""

import pytest

from knowledge_forge.rag.retriever.base import BaseRetriever, RetrievedDocument
from knowledge_forge.rag.retriever.hybrid_retriever import HybridRetriever
from knowledge_forge.rag.retriever.bm25_retriever import BM25Retriever


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


def test_rrf_fusion_vector_only():
    """测试只有向量检索结果时的 RRF 融合"""
    retriever = HybridRetriever(vector_retriever=None, bm25_retriever=None)

    vector_results = [
        RetrievedDocument(id="1", content="doc1", score=0.9, source="vector"),
        RetrievedDocument(id="2", content="doc2", score=0.8, source="vector"),
    ]

    fused = retriever._rrf_fusion(vector_results, [])

    assert len(fused) == 2
    assert "1" in fused
    assert "2" in fused
    assert fused["1"].source == "hybrid(vector)"


def test_rrf_fusion_empty():
    """测试空结果时的 RRF 融合"""
    retriever = HybridRetriever(vector_retriever=None, bm25_retriever=None)
    fused = retriever._rrf_fusion([], [])
    assert len(fused) == 0


def test_bm25_retriever_no_index():
    """测试 BM25 未构建索引时返回空结果"""
    retriever = BM25Retriever()
    import asyncio
    result = asyncio.get_event_loop().run_until_complete(
        retriever.retrieve("test query")
    )
    assert result == []


def test_bm25_retriever_index_and_search():
    """测试 BM25 索引构建和检索"""
    retriever = BM25Retriever()

    # 构建索引
    documents = [
        {"id": "1", "content": "RAG 是检索增强生成技术，结合了检索和生成"},
        {"id": "2", "content": "向量数据库用于存储和检索高维向量数据"},
        {"id": "3", "content": "Milvus 是一个开源的向量数据库"},
    ]
    retriever.index(documents)

    # 检索
    import asyncio
    results = asyncio.get_event_loop().run_until_complete(
        retriever.retrieve("向量数据库", top_k=2)
    )

    # 应该返回与"向量数据库"相关的结果
    assert len(results) > 0
    assert all(isinstance(doc, RetrievedDocument) for doc in results)
    # 结果分数应大于0
    assert all(doc.score > 0 for doc in results)
