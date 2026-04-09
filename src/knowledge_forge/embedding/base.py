"""Embedding 基类"""

from abc import ABC, abstractmethod


class BaseEmbedding(ABC):
    """Embedding 模型抽象基类"""

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量向量化文本

        Args:
            texts: 文本列表

        Returns:
            向量列表
        """
        ...

    @abstractmethod
    async def embed_query(self, query: str) -> list[float]:
        """向量化查询文本

        Args:
            query: 查询文本

        Returns:
            查询向量
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """向量维度"""
        ...
