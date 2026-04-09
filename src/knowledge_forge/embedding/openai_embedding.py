"""OpenAI Embedding 实现"""

import logging
from typing import Optional

import openai

from knowledge_forge.embedding.base import BaseEmbedding

logger = logging.getLogger(__name__)


class OpenAIEmbedding(BaseEmbedding):
    """基于 OpenAI API 的 Embedding 实现"""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        dimension: int = 1536,
        batch_size: int = 100,
    ):
        self.model = model
        self._dimension = dimension
        self.batch_size = batch_size

        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量向量化文本"""
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        # 分批处理
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            logger.debug("Embedding batch: %d/%d", i + len(batch), len(texts))

            response = await self._client.embeddings.create(
                model=self.model,
                input=batch,
            )

            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        logger.info("Embedding 完成: %d texts → %d vectors", len(texts), len(all_embeddings))
        return all_embeddings

    async def embed_query(self, query: str) -> list[float]:
        """向量化查询"""
        response = await self._client.embeddings.create(
            model=self.model,
            input=[query],
        )
        return response.data[0].embedding

    @property
    def dimension(self) -> int:
        return self._dimension
