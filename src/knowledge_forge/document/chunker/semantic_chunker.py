"""递归语义切分器 - Recursive Semantic Chunking

核心策略：
1. 首先按文档结构（标题/章节）做一级切分
2. 对超长章节按段落做二级切分
3. 对超长段落按句子做三级切分
4. 每个chunk保留前后 overlap 内容和标题层级上下文
"""

import logging
import re
from dataclasses import dataclass

from knowledge_forge.document.chunker.base import BaseChunker, Chunk, ChunkMetadata
from knowledge_forge.document.parsers import ParsedDocument

logger = logging.getLogger(__name__)

# 分句正则：匹配中英文句末标点
SENTENCE_PATTERN = re.compile(r"(?<=[。！？.!?\n])\s*")


@dataclass
class SemanticChunkerConfig:
    """语义切分器配置"""
    chunk_size: int = 800       # 目标 chunk token 数
    chunk_overlap: int = 100    # overlap token 数
    min_chunk_size: int = 100   # 最小 chunk token 数（低于此值合并）


class SemanticChunker(BaseChunker):
    """递归语义切分器"""

    def __init__(self, config: SemanticChunkerConfig | None = None):
        self.config = config or SemanticChunkerConfig()

    async def split(self, content: str, metadata: dict | None = None) -> list[Chunk]:
        """将文本内容切分为 chunks"""
        if not content.strip():
            return []

        # 第一步：按段落切分
        paragraphs = self._split_by_paragraph(content)

        # 第二步：合并小段落 + 拆分大段落
        chunks = self._merge_and_split(paragraphs)

        # 第三步：添加 overlap
        chunks = self._add_overlap(chunks)

        # 构建 Chunk 对象
        result = []
        for i, chunk_content in enumerate(chunks):
            token_count = self._estimate_token_count(chunk_content)
            chunk = Chunk(
                content=chunk_content,
                metadata=ChunkMetadata(
                    chunk_index=i,
                    total_chunks=len(chunks),
                    **(metadata or {}),
                ),
                token_count=token_count,
            )
            result.append(chunk)

        logger.info("切分完成: 输入 %d 字符 → %d chunks", len(content), len(result))
        return result

    async def split_document(self, doc: ParsedDocument) -> list[Chunk]:
        """切分已解析的文档，保留标题层级上下文"""
        all_chunks: list[Chunk] = []
        heading_stack: list[str] = []

        for section in doc.sections:
            # 更新标题链
            if section.level > 0:
                # 保持标题链与层级对应
                while len(heading_stack) >= section.level:
                    heading_stack.pop()
                heading_stack.append(section.title)

            # 切分该 section
            chunks = await self.split(
                section.content,
                metadata={
                    "source_file": doc.source_file,
                    "file_type": doc.file_type,
                    "page_number": section.page_number,
                },
            )

            # 为每个 chunk 设置标题链
            for chunk in chunks:
                chunk.heading_chain = list(heading_stack)
                chunk.metadata.document_id = doc.source_file

            all_chunks.extend(chunks)

        # 更新 total_chunks
        for chunk in all_chunks:
            chunk.metadata.total_chunks = len(all_chunks)

        logger.info(
            "文档切分完成: %s → %d chunks (sections=%d)",
            doc.title, len(all_chunks), len(doc.sections),
        )
        return all_chunks

    def _split_by_paragraph(self, content: str) -> list[str]:
        """按段落切分"""
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [content.strip()]
        return paragraphs

    def _merge_and_split(self, paragraphs: list[str]) -> list[str]:
        """合并小段落，拆分大段落"""
        result: list[str] = []
        buffer: list[str] = []
        buffer_tokens = 0

        for para in paragraphs:
            para_tokens = self._estimate_token_count(para)

            # 大段落需要拆分
            if para_tokens > self.config.chunk_size:
                # 先把 buffer 中的内容输出
                if buffer:
                    result.append("\n\n".join(buffer))
                    buffer = []
                    buffer_tokens = 0

                # 拆分大段落
                sub_chunks = self._split_large_paragraph(para)
                result.extend(sub_chunks)
            else:
                # 尝试合并到 buffer
                if buffer_tokens + para_tokens > self.config.chunk_size:
                    if buffer:
                        result.append("\n\n".join(buffer))
                    buffer = [para]
                    buffer_tokens = para_tokens
                else:
                    buffer.append(para)
                    buffer_tokens += para_tokens

        # 处理剩余 buffer
        if buffer:
            merged = "\n\n".join(buffer)
            merged_tokens = self._estimate_token_count(merged)
            # 如果太小且结果非空，合并到最后一个 chunk
            if merged_tokens < self.config.min_chunk_size and result:
                result[-1] = result[-1] + "\n\n" + merged
            else:
                result.append(merged)

        return result

    def _split_large_paragraph(self, paragraph: str) -> list[str]:
        """拆分大段落为句子级别的 chunks"""
        sentences = SENTENCE_PATTERN.split(paragraph)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return [paragraph]

        result: list[str] = []
        buffer: list[str] = []
        buffer_tokens = 0

        for sentence in sentences:
            sent_tokens = self._estimate_token_count(sentence)

            if buffer_tokens + sent_tokens > self.config.chunk_size:
                if buffer:
                    result.append(" ".join(buffer))
                buffer = [sentence]
                buffer_tokens = sent_tokens
            else:
                buffer.append(sentence)
                buffer_tokens += sent_tokens

        if buffer:
            remaining = " ".join(buffer)
            if self._estimate_token_count(remaining) < self.config.min_chunk_size and result:
                result[-1] = result[-1] + " " + remaining
            else:
                result.append(remaining)

        return result

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        """为 chunks 添加前后 overlap"""
        if len(chunks) <= 1 or self.config.chunk_overlap <= 0:
            return chunks

        result = []
        for i, chunk in enumerate(chunks):
            # 前置 overlap：取前一个 chunk 的末尾
            overlap_before = ""
            if i > 0:
                overlap_before = self._get_overlap_text(chunks[i - 1], from_end=True)

            # 后置 overlap：取后一个 chunk 的开头
            overlap_after = ""
            if i < len(chunks) - 1:
                overlap_after = self._get_overlap_text(chunks[i + 1], from_end=False)

            # 构建完整 chunk（overlap 作为上下文不放入主内容）
            enhanced = chunk
            if overlap_before:
                enhanced = f"[上文]...{overlap_before}\n---\n{enhanced}"
            if overlap_after:
                enhanced = f"{enhanced}\n---\n[下文]...{overlap_after}"

            result.append(enhanced)

        return result

    def _get_overlap_text(self, text: str, from_end: bool = True) -> str:
        """获取 overlap 文本"""
        overlap_tokens = self.config.chunk_overlap

        if from_end:
            # 从末尾取
            chars = len(text)
            # 粗略估算：1 token ≈ 2 字符（中文为主）
            approx_chars = overlap_tokens * 2
            start = max(0, chars - approx_chars)
            return text[start:]
        else:
            # 从开头取
            approx_chars = overlap_tokens * 2
            return text[:approx_chars]
