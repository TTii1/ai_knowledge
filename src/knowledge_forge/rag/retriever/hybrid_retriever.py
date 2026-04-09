"""混合检索器 - 融合向量检索和 BM25 检索结果

融合策略：Reciprocal Rank Fusion (RRF)
- 对每路检索结果按排名计算 RRF 分数
- 合并去重后按融合分数排序
- 支持异步并行多路检索
"""

import asyncio
import logging
from typing import Optional

from knowledge_forge.rag.retriever.base import BaseRetriever, RetrievedDocument
from knowledge_forge.rag.retriever.vector_retriever import VectorRetriever
from knowledge_forge.rag.retriever.bm25_retriever import BM25Retriever

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """混合检索器

    融合策略：Reciprocal Rank Fusion (RRF)
    - 向量检索捕捉语义相似性
    - BM25 检索捕捉关键词精确匹配
    - RRF 融合两路结果，互补优势
    """

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        bm25_retriever: Optional[BM25Retriever] = None,
        rrf_k: int = 60,  # RRF 参数，默认60
        vector_weight: float = 0.7,  # 向量检索权重
        bm25_weight: float = 0.3,   # BM25 检索权重
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

    async def retrieve(
        self,
        query: str,
        top_k: int = 20,
        knowledge_base: str = "default",
    ) -> list[RetrievedDocument]:
        """混合检索：向量 + BM25，使用 RRF 融合"""
        # 1. 并行多路检索
        tasks = [
            self.vector_retriever.retrieve(query, top_k=top_k, knowledge_base=knowledge_base),
        ]
        if self.bm25_retriever:
            tasks.append(
                self.bm25_retriever.retrieve(query, top_k=top_k, knowledge_base=knowledge_base)
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        vector_results = results[0] if not isinstance(results[0], Exception) else []
        bm25_results = results[1] if len(results) > 1 and not isinstance(results[1], Exception) else []

        if isinstance(results[0], Exception):
            logger.warning("向量检索异常: %s", str(results[0]))
        if len(results) > 1 and isinstance(results[1], Exception):
            logger.warning("BM25 检索异常: %s", str(results[1]))

        # 2. RRF 融合
        fused = self._rrf_fusion(vector_results, bm25_results)

        # 3. 取 top_k
        result = sorted(fused.values(), key=lambda x: x.score, reverse=True)[:top_k]

        logger.info(
            "混合检索完成: vector=%d, bm25=%d, fused=%d, final=%d",
            len(vector_results), len(bm25_results), len(fused), len(result),
        )
        return result

    def _rrf_fusion(
        self,
        vector_results: list[RetrievedDocument],
        bm25_results: list[RetrievedDocument],
    ) -> dict[str, RetrievedDocument]:
        """Reciprocal Rank Fusion 融合"""
        fused: dict[str, RetrievedDocument] = {}

        # 向量检索结果
        for rank, doc in enumerate(vector_results):
            rrf_score = self.vector_weight / (self.rrf_k + rank + 1)
            if doc.id in fused:
                fused[doc.id].score += rrf_score
            else:
                doc.score = rrf_score
                doc.source = "hybrid(vector)"
                fused[doc.id] = doc

        # BM25 检索结果
        for rank, doc in enumerate(bm25_results):
            rrf_score = self.bm25_weight / (self.rrf_k + rank + 1)
            if doc.id in fused:
                fused[doc.id].score += rrf_score
                fused[doc.id].source = "hybrid(vector+bm25)"
            else:
                doc.score = rrf_score
                doc.source = "hybrid(bm25)"
                fused[doc.id] = doc

        return fused
