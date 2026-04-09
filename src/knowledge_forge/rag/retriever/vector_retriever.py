"""向量检索器"""

import logging
from typing import Optional

from knowledge_forge.rag.retriever.base import BaseRetriever, RetrievedDocument
from knowledge_forge.embedding.base import BaseEmbedding
from knowledge_forge.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


class VectorRetriever(BaseRetriever):
    """基于向量相似度的检索器"""

    def __init__(
        self,
        embedding: BaseEmbedding,
        vector_store: VectorStore,
    ):
        self.embedding = embedding
        self.vector_store = vector_store

    async def retrieve(
        self,
        query: str,
        top_k: int = 20,
        knowledge_base: str = "default",
    ) -> list[RetrievedDocument]:
        """向量相似度检索"""
        # 1. 向量化查询
        query_embedding = await self.embedding.embed_query(query)

        # 2. Milvus 搜索
        results = await self.vector_store.search(
            query_embedding=query_embedding,
            knowledge_base=knowledge_base,
            top_k=top_k,
        )

        # 3. 转换为 RetrievedDocument
        documents = [
            RetrievedDocument(
                id=r["id"],
                content=r["content"],
                score=r["score"],
                source="vector",
                heading_chain=r.get("heading_chain", []),
                metadata=r.get("metadata", {}),
            )
            for r in results
        ]

        logger.info("向量检索完成: query='%s...', results=%d", query[:30], len(documents))
        return documents
