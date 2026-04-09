"""文档处理流水线 - 串联解析 → 切分 → 向量化 → 存储

支持两种使用方式：
1. 仅解析+切分（默认）：process() → list[Chunk]
2. 完整流水线：process_and_store() → dict（含存储结果）
"""

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

    def get_supported_extensions(self) -> list[str]:
        """获取支持的文件扩展名列表"""
        return list(self._parsers.keys())

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
            raise ValueError(f"不支持的文件格式: {file_path.suffix}，支持: {self.get_supported_extensions()}")

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

    async def process_and_store(
        self,
        file_path: Path,
        knowledge_base: str = "default",
        embedding_service=None,
        vector_store=None,
        metadata_store=None,
        doc_id: str | None = None,
    ) -> dict:
        """完整流水线：解析 → 切分 → 向量化 → 存储

        Args:
            file_path: 文档路径
            knowledge_base: 知识库名称
            embedding_service: Embedding 服务实例
            vector_store: 向量存储实例
            metadata_store: 元数据存储实例
            doc_id: 文档 ID（用于状态更新）

        Returns:
            处理结果字典
        """
        # 1. 解析 + 切分
        chunks = await self.process(file_path)

        # 2. 向量化
        if embedding_service and vector_store:
            # 更新状态：正在向量化
            if metadata_store and doc_id:
                await metadata_store.update_document_status(doc_id, status="processing")

            texts = [chunk.content for chunk in chunks]
            embeddings = await embedding_service.embed_texts(texts)
            logger.info("向量化完成: %d chunks → %d vectors", len(texts), len(embeddings))

            # 3. 存入向量数据库
            await vector_store.insert_chunks(chunks, embeddings, knowledge_base)
            logger.info("向量存储完成: %d chunks → knowledge_base=%s", len(chunks), knowledge_base)
        else:
            logger.warning("Embedding 或 VectorStore 未提供，跳过向量化存储步骤")

        # 4. 更新元数据
        total_tokens = sum(c.token_count for c in chunks)
        if metadata_store and doc_id:
            await metadata_store.update_document_status(
                doc_id,
                status="completed",
                chunk_count=len(chunks),
                total_tokens=total_tokens,
            )
            await metadata_store.update_knowledge_base_stats(
                knowledge_base, doc_delta=0, chunk_delta=len(chunks)
            )

        return {
            "file_path": str(file_path),
            "chunks": len(chunks),
            "total_tokens": total_tokens,
            "status": "completed",
        }
