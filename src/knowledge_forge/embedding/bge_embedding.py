"""BGE 本地 Embedding 实现（预留）"""

import logging
from knowledge_forge.embedding.base import BaseEmbedding

logger = logging.getLogger(__name__)


class BGEEmbedding(BaseEmbedding):
    """基于 BGE-M3 模型的本地 Embedding 实现

    注意：需要安装 FlagEmbedding 和 torch
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", dimension: int = 1024):
        self.model_name = model_name
        self._dimension = dimension
        self._model = None

    def _load_model(self):
        """延迟加载模型"""
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel
            self._model = BGEM3FlagModel(self.model_name, use_fp16=True)
            logger.info("BGE-M3 模型加载完成")

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量向量化文本"""
        self._load_model()
        embeddings = self._model.encode(texts)
        return embeddings.tolist()

    async def embed_query(self, query: str) -> list[float]:
        """向量化查询"""
        self._load_model()
        embedding = self._model.encode([query])
        return embedding[0].tolist()

    @property
    def dimension(self) -> int:
        return self._dimension
