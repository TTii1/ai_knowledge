"""Markdown 文档解析器 - 基于正则 + 状态机，正确处理代码块"""

import logging
import re
from pathlib import Path

from knowledge_forge.document.parsers import BaseParser, ParsedDocument, DocumentSection

logger = logging.getLogger(__name__)

# 标题正则：行首 1-6 个 # 后跟空格
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$")


class MarkdownParser(BaseParser):
    """Markdown 文档解析器，保留标题层级，正确跳过代码块中的 # """

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

        # 状态机：追踪代码块
        in_code_block = False

        for line in lines:
            stripped = line.strip()

            # 追踪代码块状态（``` 开关）
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                current_content_parts.append(line)
                continue

            # 在代码块内部，不识别标题
            if in_code_block:
                current_content_parts.append(line)
                continue

            # 检测 Markdown 标题（仅非代码块内）
            heading_match = HEADING_PATTERN.match(stripped)
            if heading_match:
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

                # 解析标题级别和文本
                hashes = heading_match.group(1)
                heading_text = heading_match.group(2).strip()
                current_level = len(hashes)
                current_heading = heading_text
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
