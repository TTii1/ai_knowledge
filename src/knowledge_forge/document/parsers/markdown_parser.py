"""Markdown 文档解析器 - 基于 mistune"""

import logging
from pathlib import Path

from knowledge_forge.document.parsers import BaseParser, ParsedDocument, DocumentSection

logger = logging.getLogger(__name__)


class MarkdownParser(BaseParser):
    """Markdown 文档解析器，保留标题层级"""

    supported_extensions = [".md", ".markdown"]

    async def parse(self, file_path: Path) -> ParsedDocument:
        """解析 Markdown 文档"""
        logger.info("开始解析 Markdown: %s", file_path)

        content = file_path.read_text(encoding="utf-8")
        lines = content.split("\n")

        sections: list[DocumentSection] = []
        full_content_parts: list[str] = []

        current_heading = Path(file_path).stem
        current_level = 0
        current_content_parts: list[str] = []

        for line in lines:
            stripped = line.strip()

            # 检测 Markdown 标题
            if stripped.startswith("#"):
                # 保存前一个 section
                if current_content_parts:
                    sec_content = "\n".join(current_content_parts).strip()
                    if sec_content:
                        sections.append(DocumentSection(
                            title=current_heading,
                            content=sec_content,
                            level=current_level,
                        ))
                        full_content_parts.append(sec_content)

                # 解析标题级别
                level = 0
                for char in stripped:
                    if char == "#":
                        level += 1
                    else:
                        break
                current_heading = stripped.lstrip("#").strip()
                current_level = level
                current_content_parts = [line]
            else:
                current_content_parts.append(line)

        # 保存最后一个 section
        if current_content_parts:
            sec_content = "\n".join(current_content_parts).strip()
            if sec_content:
                sections.append(DocumentSection(
                    title=current_heading,
                    content=sec_content,
                    level=current_level,
                ))
                full_content_parts.append(sec_content)

        full_content = "\n\n".join(full_content_parts)

        return ParsedDocument(
            title=Path(file_path).stem,
            content=full_content,
            sections=sections,
            metadata={},
            source_file=str(file_path),
            total_pages=len(sections),
            file_size=file_path.stat().st_size,
            file_type="markdown",
        )
