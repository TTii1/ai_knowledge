"""TXT 文档解析器"""

import logging
from pathlib import Path

from knowledge_forge.document.parsers import BaseParser, ParsedDocument, DocumentSection

logger = logging.getLogger(__name__)


class TXTParser(BaseParser):
    """纯文本文档解析器，按空行分段"""

    supported_extensions = [".txt", ".text"]

    async def parse(self, file_path: Path) -> ParsedDocument:
        """解析 TXT 文档"""
        logger.info("开始解析 TXT: %s", file_path)

        content = file_path.read_text(encoding="utf-8")

        # 按空行分段
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        sections = [
            DocumentSection(
                title=f"段落 {i+1}",
                content=para,
                level=0,
            )
            for i, para in enumerate(paragraphs)
        ]

        return ParsedDocument(
            title=Path(file_path).stem,
            content=content,
            sections=sections,
            metadata={"paragraph_count": len(paragraphs)},
            source_file=str(file_path),
            total_pages=len(sections),
            file_size=file_path.stat().st_size,
            file_type="txt",
        )
