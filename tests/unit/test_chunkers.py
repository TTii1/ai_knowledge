"""文档切分器单元测试"""

import pytest

from knowledge_forge.document.chunker.semantic_chunker import SemanticChunker, SemanticChunkerConfig
from knowledge_forge.document.chunker.recursive_chunker import RecursiveChunker
from knowledge_forge.document.chunker.base import Chunk, ChunkMetadata
from knowledge_forge.document.parsers import ParsedDocument, DocumentSection


@pytest.fixture
def semantic_chunker():
    return SemanticChunker(config=SemanticChunkerConfig(
        chunk_size=800,
        chunk_overlap=100,
        min_chunk_size=100,
    ))


@pytest.fixture
def small_chunker():
    """小 chunk 测试用切分器"""
    return SemanticChunker(config=SemanticChunkerConfig(
        chunk_size=100,
        chunk_overlap=20,
        min_chunk_size=20,
    ))


class TestSemanticChunker:
    @pytest.mark.asyncio
    async def test_split_short_text(self, semantic_chunker):
        """测试短文本（不需要切分）"""
        text = "这是一段短文本，不需要切分。"
        chunks = await semantic_chunker.split(text)

        assert len(chunks) == 1
        assert chunks[0].content == text
        assert chunks[0].token_count > 0

    @pytest.mark.asyncio
    async def test_split_empty_text(self, semantic_chunker):
        """测试空文本"""
        chunks = await semantic_chunker.split("")
        assert len(chunks) == 0

    @pytest.mark.asyncio
    async def test_split_with_overlap(self, small_chunker):
        """测试 overlap 存入 context_before/context_after 而不污染 content"""
        # 创建足够长的文本以产生多个 chunks
        paragraphs = []
        for i in range(10):
            paragraphs.append(f"这是第{i+1}段文本，内容足够长以便测试切分和重叠机制。" * 5)
        text = "\n\n".join(paragraphs)

        chunks = await small_chunker.split(text)

        assert len(chunks) > 1

        # 验证 overlap 存入 context 字段而非 content
        for i, chunk in enumerate(chunks):
            # content 不应包含 [上文] 或 [下文] 标记
            assert "[上文]" not in chunk.content
            assert "[下文]" not in chunk.content

            # 非首 chunk 应有 context_before
            if i > 0:
                assert chunk.context_before != "", f"Chunk {i} 应有 context_before"

            # 非末 chunk 应有 context_after
            if i < len(chunks) - 1:
                assert chunk.context_after != "", f"Chunk {i} 应有 context_after"

    @pytest.mark.asyncio
    async def test_split_document_with_headings(self, semantic_chunker):
        """测试文档切分保留标题层级"""
        doc = ParsedDocument(
            title="测试文档",
            content="",
            sections=[
                DocumentSection(title="第一章", content="第一章的内容。" * 20, level=1),
                DocumentSection(title="1.1 小节", content="小节内容。" * 20, level=2),
                DocumentSection(title="第二章", content="第二章内容。" * 20, level=1),
            ],
            source_file="test.md",
            file_type="markdown",
        )

        chunks = await semantic_chunker.split_document(doc)

        assert len(chunks) > 0
        # 每个 chunk 都应该有标题链
        for chunk in chunks:
            assert isinstance(chunk.heading_chain, list)

    @pytest.mark.asyncio
    async def test_chunk_metadata(self, semantic_chunker):
        """测试 Chunk 元数据"""
        text = "测试文本。" * 50
        chunks = await semantic_chunker.split(text, metadata={"source_file": "test.txt"})

        for chunk in chunks:
            assert chunk.id != ""
            assert chunk.token_count > 0
            assert chunk.metadata.chunk_index >= 0
            assert chunk.metadata.total_chunks == len(chunks)
            assert chunk.metadata.source_file == "test.txt"


class TestRecursiveChunker:
    @pytest.mark.asyncio
    async def test_split_text(self):
        """测试递归字符切分"""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
        text = "这是一段测试文本。" * 30

        chunks = await chunker.split(text)

        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.content != ""
            assert chunk.token_count > 0
