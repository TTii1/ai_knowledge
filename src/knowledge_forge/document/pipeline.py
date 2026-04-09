"""文档处理流水线 - 串联解析 → 切分 → 向量化 → 存储"""

import logging
from pathlib import Path
from typing import Optional

from knowledge_forge.document.parsers.base import BaseParser
from knowledge_forge.document.parsers.pdf_parser import PDFParser
from knowledge_forge.document.parsers.word_parser import WordParser
from knowledge_forge.document.parsers.markdown_parser import MarkdownParser
from knowledge_forge.document.parsers.txt_parser import TXTParser
from knowledge_forge.document.chunker.semantic_chunker import SemanticChunker
from knowledge_forge.document.chunker.base import Chunk

logger = logging.getLogger(__name__)


class DocumentPipeline:
    """文档处理流水线

    完整流程：文件上传 → 格式识别 → 解析 → 切分 → (向量化 → 存储)
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        min_chunk_size: int = 100,
    ):
        # 注册解析器
        self._parsers: dict[str, BaseParser] = {}
        self._register_parsers()

        # 初始化切分器
        self._chunker = SemanticChunker(
            config=SemanticChunker.SemanticChunkerConfig(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                min_chunk_size=min_chunk_size,
            )
        )

    def _register_parsers(self) -> None:
        """注册所有文档解析器"""
        parsers = [PDFParser(), WordParser(), MarkdownParser(), TXTParser()]
        for parser in parsers:
            for ext in parser.supported_extensions:
                self._parsers[ext] = parser

    def get_parser(self, file_path: Path) -> Optional[BaseParser]:
        """根据文件扩展名获取对应的解析器"""
        ext = file_path.suffix.lower()
        return self._parsers.get(ext)

    async def process(self, file_path: Path) -> list[Chunk]:
        """处理单个文档：解析 → 切分

        Args:
            file_path: 文档路径

        Returns:
            切分后的 Chunk 列表

        Raises:
            ValueError: 不支持的文件格式
        """
        parser = self.get_parser(file_path)
        if not parser:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")

        logger.info("开始处理文档: %s (parser=%s)", file_path, type(parser).__name__)

        # 1. 解析文档
        parsed_doc = await parser.parse(file_path)
        logger.info(
            "文档解析完成: title=%s, sections=%d, tables=%d",
            parsed_doc.title, len(parsed_doc.sections), len(parsed_doc.tables),
        )

        # 2. 切分文档
        chunks = await self._chunker.split_document(parsed_doc)
        logger.info("文档切分完成: chunks=%d", len(chunks))

        return chunks
