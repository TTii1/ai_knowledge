"""文档切分器数据模型与基类"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class ChunkMetadata:
    """Chunk 元数据"""
    source_file: str = ""
    document_id: str = ""
    page_number: int | None = None
    chunk_index: int = 0
    total_chunks: int = 0
    file_type: str = ""
    knowledge_base: str = "default"


@dataclass
class Chunk:
    """文档切分块"""
    id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""
    context_before: str = ""
    context_after: str = ""
    heading_chain: list[str] = field(default_factory=list)
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)
    token_count: int = 0


class BaseChunker(ABC):
    """文档切分器基类"""

    @abstractmethod
    async def split(self, content: str, metadata: dict | None = None) -> list[Chunk]:
        """将文本内容切分为 chunks

        Args:
            content: 待切分的文本内容
            metadata: 附加元数据

        Returns:
            切分后的 Chunk 列表
        """
        ...

    def _estimate_token_count(self, text: str) -> int:
        """粗略估算 token 数量（中文约 1.5 字/token，英文约 4 字符/token）"""
        chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)
