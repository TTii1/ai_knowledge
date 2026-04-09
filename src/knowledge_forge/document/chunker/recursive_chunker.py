"""递归字符切分器 - 按固定字符数切分（简单但可靠的后备方案）"""

import logging

from knowledge_forge.document.chunker.base import BaseChunker, Chunk, ChunkMetadata

logger = logging.getLogger(__name__)


class RecursiveChunker(BaseChunker):
    """递归字符切分器

    按照分隔符优先级递归切分：
    1. 双换行（段落）
    2. 单换行
    3. 句号/问号/感叹号
    4. 逗号
    5. 空格
    6. 直接按字符数切割
    """

    separators = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", "，", ",", " ", ""]

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def split(self, content: str, metadata: dict | None = None) -> list[Chunk]:
        """递归切分文本"""
        if not content.strip():
            return []

        chunks_text = self._recursive_split(content, self.separators)

        result = []
        for i, chunk_content in enumerate(chunks_text):
            token_count = self._estimate_token_count(chunk_content)
            chunk = Chunk(
                content=chunk_content,
                metadata=ChunkMetadata(
                    chunk_index=i,
                    total_chunks=len(chunks_text),
                    **(metadata or {}),
                ),
                token_count=token_count,
            )
            result.append(chunk)

        return result

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """递归切分"""
        if not text:
            return []

        # 如果文本已经够小，直接返回
        if self._estimate_token_count(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        # 如果没有更多分隔符，强制按字符数切割
        if not separators:
            return self._hard_split(text)

        separator = separators[0]
        remaining_separators = separators[1:]

        # 用当前分隔符切分
        parts = text.split(separator) if separator else list(text)
        parts = [p for p in parts if p.strip()]

        if not parts:
            return self._recursive_split(text, remaining_separators)

        # 合并小片段
        result = []
        buffer = ""
        for part in parts:
            candidate = f"{buffer}{separator}{part}" if buffer else part
            if self._estimate_token_count(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                if buffer:
                    result.append(buffer.strip())
                # 如果单个 part 还是太大，继续递归
                if self._estimate_token_count(part) > self.chunk_size:
                    sub_chunks = self._recursive_split(part, remaining_separators)
                    result.extend(sub_chunks)
                    buffer = ""
                else:
                    buffer = part

        if buffer:
            result.append(buffer.strip())

        return [r for r in result if r]

    def _hard_split(self, text: str) -> list[str]:
        """强制按字符数切割"""
        approx_chars = self.chunk_size * 2  # 粗略估算
        overlap_chars = self.chunk_overlap * 2

        result = []
        start = 0
        while start < len(text):
            end = start + approx_chars
            chunk = text[start:end].strip()
            if chunk:
                result.append(chunk)
            start = end - overlap_chars

        return result
