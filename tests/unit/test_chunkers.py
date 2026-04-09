"""文档切分器单元测试"""

import pytest

from knowledge_forge.document.chunker.semantic_chunker import SemanticChunker
from knowledge_forge.document.chunker.recursive_chunker import RecursiveChunker
from knowledge_forge.document.parsers.txt_parser import TXTParser


@pytest.mark.asyncio
async def test_semantic_chunker_basic():
    """测试语义切分器基本功能"""
    chunker = SemanticChunker()
    content = "这是第一段内容。" * 100 + "\n\n" + "这是第二段内容。" * 100

    chunks = await chunker.split(content)

    assert len(chunks) > 0
    # 每个 chunk 应该有内容
    for chunk in chunks:
        assert len(chunk.content) > 0
        assert chunk.token_count > 0


@pytest.mark.asyncio
async def test_semantic_chunker_with_document(sample_txt_file):
    """测试语义切分器处理完整文档"""
    parser = TXTParser()
    doc = await parser.parse(sample_txt_file)

    chunker = SemanticChunker()
    chunks = await chunker.split_document(doc)

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.id
        assert chunk.content
        assert chunk.metadata.total_chunks == len(chunks)


@pytest.mark.asyncio
async def test_recursive_chunker():
    """测试递归字符切分器"""
    chunker = RecursiveChunker(chunk_size=200, chunk_overlap=20)
    content = "这是一个很长的文本。" * 200

    chunks = await chunker.split(content)

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.content
        assert chunk.token_count > 0


@pytest.mark.asyncio
async def test_empty_content():
    """测试空内容处理"""
    chunker = SemanticChunker()
    chunks = await chunker.split("")
    assert len(chunks) == 0

    chunks = await chunker.split("   ")
    assert len(chunks) == 0
