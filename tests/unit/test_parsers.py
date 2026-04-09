"""文档解析器单元测试"""

import pytest
from pathlib import Path

from knowledge_forge.document.parsers.txt_parser import TXTParser
from knowledge_forge.document.parsers.markdown_parser import MarkdownParser
from knowledge_forge.document.parsers.word_parser import WordParser, _parse_heading_level
from knowledge_forge.document.parsers.pdf_parser import PDFParser


@pytest.fixture
def txt_parser():
    return TXTParser()


@pytest.fixture
def md_parser():
    return MarkdownParser()


class TestTXTParser:
    @pytest.mark.asyncio
    async def test_parse_txt(self, txt_parser, tmp_path):
        """测试 TXT 解析"""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("第一段\n\n第二段\n\n第三段", encoding="utf-8")

        result = await txt_parser.parse(txt_file)

        assert result.title == "test"
        assert result.file_type == "txt"
        assert len(result.sections) == 3
        assert "第一段" in result.sections[0].content
        assert result.total_pages == 3

    @pytest.mark.asyncio
    async def test_parse_empty_txt(self, txt_parser, tmp_path):
        """测试空文件"""
        txt_file = tmp_path / "empty.txt"
        txt_file.write_text("", encoding="utf-8")

        result = await txt_parser.parse(txt_file)
        assert result.file_type == "txt"
        assert len(result.sections) == 0

    @pytest.mark.asyncio
    async def test_supported_extensions(self, txt_parser):
        assert ".txt" in txt_parser.supported_extensions
        assert ".text" in txt_parser.supported_extensions


class TestMarkdownParser:
    @pytest.mark.asyncio
    async def test_parse_markdown_with_headings(self, md_parser, tmp_path):
        """测试 Markdown 标题层级解析"""
        md_file = tmp_path / "test.md"
        md_file.write_text(
            "# 标题一\n\n"
            "标题一的内容\n\n"
            "## 标题二\n\n"
            "标题二的内容\n\n"
            "### 标题三\n\n"
            "标题三的内容\n",
            encoding="utf-8",
        )

        result = await md_parser.parse(md_file)

        assert result.title == "test"
        assert result.file_type == "markdown"
        # 应该有 3 个 section（每个标题对应一个）
        assert len(result.sections) == 3
        # 验证标题层级
        assert result.sections[0].level == 1
        assert result.sections[1].level == 2
        assert result.sections[2].level == 3

    @pytest.mark.asyncio
    async def test_parse_markdown_code_block(self, md_parser, tmp_path):
        """测试代码块内的 # 不被误判为标题"""
        md_file = tmp_path / "code.md"
        md_file.write_text(
            "# 真正的标题\n\n"
            "正文内容\n\n"
            "```python\n"
            "# 这是注释不是标题\n"
            "x = 1\n"
            "## 也不是标题\n"
            "```\n\n"
            "正文继续\n",
            encoding="utf-8",
        )

        result = await md_parser.parse(md_file)

        # 只有1个真正的标题（# 真正的标题）
        heading_sections = [s for s in result.sections if s.level > 0]
        assert len(heading_sections) == 1
        assert heading_sections[0].title == "真正的标题"
        # 代码块内容应完整保留
        full_content = result.content
        assert "# 这是注释不是标题" in full_content

    @pytest.mark.asyncio
    async def test_parse_markdown_inline_code(self, md_parser, tmp_path):
        """测试行内代码不影响解析"""
        md_file = tmp_path / "inline.md"
        md_file.write_text(
            "# 标题\n\n"
            "行内代码 `# not heading` 不影响解析\n",
            encoding="utf-8",
        )

        result = await md_parser.parse(md_file)
        assert len(result.sections) >= 1


class TestWordHeadingLevel:
    """测试 Word 解析器的标题级别解析"""

    def test_english_heading(self):
        assert _parse_heading_level("Heading 1") == 1
        assert _parse_heading_level("Heading 2") == 2
        assert _parse_heading_level("Heading 3") == 3
        assert _parse_heading_level("heading 1") == 1  # 大小写不敏感

    def test_chinese_heading(self):
        assert _parse_heading_level("标题 1") == 1
        assert _parse_heading_level("标题 2") == 2
        assert _parse_heading_level("标题1") == 1
        assert _parse_heading_level("标题2") == 2

    def test_non_heading(self):
        assert _parse_heading_level("Normal") is None
        assert _parse_heading_level("List Paragraph") is None
        assert _parse_heading_level("") is None
        assert _parse_heading_level(None) is None

    def test_edge_cases(self):
        assert _parse_heading_level("Heading 4") == 4
        assert _parse_heading_level("Heading 6") == 6
