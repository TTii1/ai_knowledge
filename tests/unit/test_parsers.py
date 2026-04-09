"""文档解析器单元测试"""

import pytest
from pathlib import Path

from knowledge_forge.document.parsers.txt_parser import TXTParser
from knowledge_forge.document.parsers.markdown_parser import MarkdownParser


@pytest.mark.asyncio
async def test_txt_parser(sample_txt_file):
    """测试 TXT 解析器"""
    parser = TXTParser()
    assert parser.can_parse(sample_txt_file)

    result = await parser.parse(sample_txt_file)

    assert result.title == "test"
    assert result.file_type == "txt"
    assert len(result.content) > 0
    assert len(result.sections) > 0
    assert "RAG" in result.content


@pytest.mark.asyncio
async def test_markdown_parser(sample_markdown_file):
    """测试 Markdown 解析器"""
    parser = MarkdownParser()
    assert parser.can_parse(sample_markdown_file)

    result = await parser.parse(sample_markdown_file)

    assert result.title == "test"
    assert result.file_type == "markdown"
    assert len(result.sections) > 0
    # 检查标题层级是否被识别
    heading_sections = [s for s in result.sections if s.level > 0]
    assert len(heading_sections) > 0


def test_parser_extension_check():
    """测试文件扩展名检查"""
    txt_parser = TXTParser()
    assert txt_parser.can_parse(Path("test.txt"))
    assert not txt_parser.can_parse(Path("test.pdf"))

    md_parser = MarkdownParser()
    assert md_parser.can_parse(Path("test.md"))
    assert md_parser.can_parse(Path("test.markdown"))
    assert not md_parser.can_parse(Path("test.pdf"))
