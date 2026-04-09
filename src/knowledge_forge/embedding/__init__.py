"""向量化模块"""

from knowledge_forge.embedding.base import BaseEmbedding
from knowledge_forge.embedding.openai_embedding import OpenAIEmbedding

__all__ = ["BaseEmbedding", "OpenAIEmbedding"]
