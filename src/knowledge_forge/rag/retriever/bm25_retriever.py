"""BM25 关键词检索器"""

import logging
from typing import Optional

from knowledge_forge.rag.retriever.base import BaseRetriever, RetrievedDocument

logger = logging.getLogger(__name__)


class BM25Retriever(BaseRetriever):
    """基于 BM25 的关键词检索器

    使用 rank-bm25 库实现，支持中文分词（jieba）
    """

    def __init__(self):
        self._corpus: list[str] = []
        self._bm25 = None
        self._doc_ids: list[str] = []

    def index(self, documents: list[dict]) -> None:
        """构建 BM25 索引

        Args:
            documents: 文档列表，每个包含 id 和 content
        """
        from rank_bm25 import BM25Okapi
        import jieba

        self._corpus = [doc["content"] for doc in documents]
        self._doc_ids = [doc["id"] for doc in documents]

        # 中文分词
        tokenized_corpus = [list(jieba.cut(doc)) for doc in self._corpus]
        self._bm25 = BM25Okapi(tokenized_corpus)

        logger.info("BM25 索引构建完成: %d documents", len(documents))

    async def retrieve(
        self,
        query: str,
        top_k: int = 20,
        knowledge_base: str = "default",
    ) -> list[RetrievedDocument]:
        """BM25 关键词检索"""
        if not self._bm25:
            logger.warning("BM25 索引未构建，返回空结果")
            return []

        import jieba

        tokenized_query = list(jieba.cut(query))
        scores = self._bm25.get_scores(tokenized_query)

        # 取 top_k
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        documents = [
            RetrievedDocument(
                id=self._doc_ids[i],
                content=self._corpus[i],
                score=float(scores[i]),
                source="bm25",
            )
            for i in top_indices
            if scores[i] > 0  # 过滤零分结果
        ]

        logger.info("BM25 检索完成: query='%s...', results=%d", query[:30], len(documents))
        return documents
