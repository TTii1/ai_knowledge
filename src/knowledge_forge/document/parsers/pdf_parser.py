"""PDF 文档解析器 - 基于 PyMuPDF"""

import logging
from pathlib import Path

from knowledge_forge.document.parsers import BaseParser, ParsedDocument, DocumentSection, Table

logger = logging.getLogger(__name__)


class PDFParser(BaseParser):
    """PDF 文档解析器，支持文本提取和表格识别"""

    supported_extensions = [".pdf"]

    async def parse(self, file_path: Path) -> ParsedDocument:
        """解析 PDF 文档

        Args:
            file_path: PDF 文件路径

        Returns:
            ParsedDocument: 解析后的文档
        """
        import fitz  # pymupdf

        logger.info("开始解析 PDF: %s", file_path)

        doc = fitz.open(str(file_path))
        sections: list[DocumentSection] = []
        tables: list[Table] = []
        full_content_parts: list[str] = []

        try:
            title = Path(file_path).stem

            for page_num in range(len(doc)):
                page = doc[page_num]

                # 提取文本
                text = page.get_text("text")
                if text.strip():
                    section = DocumentSection(
                        title=f"Page {page_num + 1}",
                        content=text.strip(),
                        level=0,
                        page_number=page_num + 1,
                    )
                    sections.append(section)
                    full_content_parts.append(text)

                # 提取表格（PyMuPDF 的表格提取功能）
                tab = page.find_tables()
                if tab.tables:
                    for table in tab.tables:
                        table_data = table.extract()
                        if table_data and len(table_data) > 1:
                            headers = table_data[0]
                            rows = table_data[1:]
                            tables.append(Table(headers=headers, rows=rows))

            full_content = "\n\n".join(full_content_parts)

            return ParsedDocument(
                title=title,
                content=full_content,
                sections=sections,
                metadata={
                    "page_count": len(doc),
                    "has_tables": len(tables) > 0,
                },
                tables=tables,
                source_file=str(file_path),
                total_pages=len(doc),
                file_size=file_path.stat().st_size,
                file_type="pdf",
            )
        finally:
            doc.close()
