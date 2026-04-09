"""检索器基类"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class RetrievedDocument:
    """检索到的文档"""
    id: str
    content: str
    score: float
    source: str = ""  # 来源：vector / bm25 / graph
    heading_chain: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class BaseRetriever(ABC):
    """检索器基类"""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 20,
        knowledge_base: str = "default",
    ) -> list[RetrievedDocument]:
        """检索相关文档

        Args:
            query: 查询文本
            top_k: 返回结果数
            knowledge_base: 知识库名称

        Returns:
            检索结果列表
        """
        ...
