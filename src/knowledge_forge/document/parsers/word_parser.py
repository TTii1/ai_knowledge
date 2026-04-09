"""Word 文档解析器 - 基于 python-docx"""

import logging
from pathlib import Path

from knowledge_forge.document.parsers import BaseParser, ParsedDocument, DocumentSection, Table

logger = logging.getLogger(__name__)


class WordParser(BaseParser):
    """Word (.docx) 文档解析器"""

    supported_extensions = [".docx"]

    async def parse(self, file_path: Path) -> ParsedDocument:
        """解析 Word 文档"""
        from docx import Document

        logger.info("开始解析 Word: %s", file_path)

        doc = Document(str(file_path))
        sections: list[DocumentSection] = []
        tables: list[Table] = []
        full_content_parts: list[str] = []

        current_heading = ""
        current_level = 0
        current_content_parts: list[str] = []

        for para in doc.paragraphs:
            style_name = para.style.name if para.style else ""

            # 检测标题
            if style_name.startswith("Heading"):
                # 保存前一个 section
                if current_content_parts:
                    content = "\n".join(current_content_parts)
                    sections.append(DocumentSection(
                        title=current_heading,
                        content=content,
                        level=current_level,
                    ))
                    full_content_parts.append(content)
                    current_content_parts = []

                current_level = int(style_name.replace("Heading ", "").replace("Heading", "1") or "1")
                current_heading = para.text
                current_content_parts.append(para.text)
            else:
                current_content_parts.append(para.text)

        # 保存最后一个 section
        if current_content_parts:
            content = "\n".join(current_content_parts)
            sections.append(DocumentSection(
                title=current_heading,
                content=content,
                level=current_level,
            ))
            full_content_parts.append(content)

        # 提取表格
        for table in doc.tables:
            rows_data = []
            for row in table.rows:
                row_data = [cell.text for cell in row.cells]
                rows_data.append(row_data)
            if rows_data:
                tables.append(Table(
                    headers=rows_data[0],
                    rows=rows_data[1:],
                ))

        full_content = "\n\n".join(full_content_parts)
        title = Path(file_path).stem

        return ParsedDocument(
            title=title,
            content=full_content,
            sections=sections,
            metadata={"has_tables": len(tables) > 0},
            tables=tables,
            source_file=str(file_path),
            total_pages=len(sections),
            file_size=file_path.stat().st_size,
            file_type="docx",
        )
