"""PostgreSQL 元数据存储 - SQLAlchemy ORM + 完整 CRUD

存储内容：
- 知识库（Knowledge Base）配置
- 文档元数据与处理状态
- 文档 Chunk 统计信息
"""

import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from sqlalchemy import (
    Column, String, Integer, Text, DateTime, Boolean,
    JSON, select, update, delete, func,
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

logger = logging.getLogger(__name__)


# ============ ORM 基类 ============

class Base(DeclarativeBase):
    pass


# ============ ORM 模型 ============

class KnowledgeBaseModel(Base):
    """知识库表"""
    __tablename__ = "knowledge_bases"

    name = Column(String(256), primary_key=True, comment="知识库名称")
    description = Column(Text, default="", comment="知识库描述")
    embedding_model = Column(String(128), default="text-embedding-3-small", comment="Embedding 模型")
    embedding_dimension = Column(Integer, default=1536, comment="向量维度")
    chunk_size = Column(Integer, default=800, comment="Chunk 大小")
    chunk_overlap = Column(Integer, default=100, comment="Overlap 大小")
    document_count = Column(Integer, default=0, comment="文档数量")
    chunk_count = Column(Integer, default=0, comment="Chunk 总数")
    is_active = Column(Boolean, default=True, comment="是否启用")
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")


class DocumentModel(Base):
    """文档表"""
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True, comment="文档 ID (UUID)")
    filename = Column(String(512), nullable=False, comment="原始文件名")
    file_type = Column(String(32), nullable=False, comment="文件类型 (pdf/docx/md/txt)")
    file_size = Column(Integer, default=0, comment="文件大小 (bytes)")
    file_path = Column(String(1024), default="", comment="存储路径")
    knowledge_base = Column(String(256), default="default", index=True, comment="所属知识库")
    title = Column(String(512), default="", comment="文档标题")
    status = Column(String(32), default="pending", comment="处理状态: pending/processing/completed/failed")
    chunk_count = Column(Integer, default=0, comment="Chunk 数量")
    total_tokens = Column(Integer, default=0, comment="总 Token 数")
    error_message = Column(Text, default="", comment="错误信息")
    metadata_json = Column(JSON, default=dict, comment="额外元数据")
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")


# ============ 元数据存储 ============

class MetadataStore:
    """PostgreSQL 元数据存储

    存储：文档信息、知识库配置、处理状态追踪
    """

    def __init__(self, database_url: str):
        self.database_url = database_url
        self._engine = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None

    async def connect(self) -> None:
        """连接数据库并创建表结构"""
        # SQLite 不支持 pool_size/max_overflow 参数
        engine_kwargs = {"echo": False}
        if not self.database_url.startswith("sqlite"):
            engine_kwargs["pool_size"] = 5
            engine_kwargs["max_overflow"] = 10

        self._engine = create_async_engine(
            self.database_url,
            **engine_kwargs,
        )
        self._session_factory = async_sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False,
        )

        # 创建所有表
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        logger.info("元数据存储连接成功: %s", self.database_url.split("@")[-1] if "@" in self.database_url else self.database_url)

    async def disconnect(self) -> None:
        """断开连接"""
        if self._engine:
            await self._engine.dispose()
            logger.info("元数据存储连接已关闭")

    def get_session(self) -> AsyncSession:
        """获取数据库 session"""
        if not self._session_factory:
            raise RuntimeError("MetadataStore 未连接，请先调用 connect()")
        return self._session_factory()

    # ============ 知识库操作 ============

    async def create_knowledge_base(
        self,
        name: str,
        description: str = "",
        embedding_model: str = "text-embedding-3-small",
        embedding_dimension: int = 1536,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ) -> dict:
        """创建知识库"""
        async with self.get_session() as session:
            kb = KnowledgeBaseModel(
                name=name,
                description=description,
                embedding_model=embedding_model,
                embedding_dimension=embedding_dimension,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            session.add(kb)
            await session.commit()
            logger.info("创建知识库: %s", name)
            return self._kb_to_dict(kb)

    async def get_knowledge_base(self, name: str) -> Optional[dict]:
        """获取知识库信息"""
        async with self.get_session() as session:
            result = await session.execute(
                select(KnowledgeBaseModel).where(KnowledgeBaseModel.name == name)
            )
            kb = result.scalar_one_or_none()
            return self._kb_to_dict(kb) if kb else None

    async def list_knowledge_bases(
        self, page: int = 1, page_size: int = 20, active_only: bool = True,
    ) -> dict:
        """列出知识库"""
        async with self.get_session() as session:
            query = select(KnowledgeBaseModel)
            if active_only:
                query = query.where(KnowledgeBaseModel.is_active == True)  # noqa: E712
            query = query.order_by(KnowledgeBaseModel.created_at.desc())

            # 计算总数
            count_query = select(func.count()).select_from(query.subquery())
            total = (await session.execute(count_query)).scalar() or 0

            # 分页
            items = (await session.execute(
                query.offset((page - 1) * page_size).limit(page_size)
            )).scalars().all()

            return {
                "items": [self._kb_to_dict(kb) for kb in items],
                "total": total,
                "page": page,
                "page_size": page_size,
            }

    async def update_knowledge_base_stats(self, name: str, doc_delta: int = 0, chunk_delta: int = 0) -> None:
        """更新知识库统计信息"""
        async with self.get_session() as session:
            await session.execute(
                update(KnowledgeBaseModel)
                .where(KnowledgeBaseModel.name == name)
                .values(
                    document_count=KnowledgeBaseModel.document_count + doc_delta,
                    chunk_count=KnowledgeBaseModel.chunk_count + chunk_delta,
                    updated_at=datetime.now(timezone.utc),
                )
            )
            await session.commit()

    async def delete_knowledge_base(self, name: str) -> bool:
        """删除知识库（软删除）"""
        async with self.get_session() as session:
            result = await session.execute(
                update(KnowledgeBaseModel)
                .where(KnowledgeBaseModel.name == name)
                .values(is_active=False, updated_at=datetime.now(timezone.utc))
            )
            await session.commit()
            return result.rowcount > 0

    # ============ 文档操作 ============

    async def create_document(
        self,
        filename: str,
        file_type: str,
        file_size: int,
        file_path: str,
        knowledge_base: str = "default",
        title: str = "",
        metadata: dict | None = None,
    ) -> dict:
        """创建文档记录"""
        doc_id = str(uuid4())
        async with self.get_session() as session:
            doc = DocumentModel(
                id=doc_id,
                filename=filename,
                file_type=file_type,
                file_size=file_size,
                file_path=file_path,
                knowledge_base=knowledge_base,
                title=title or filename,
                status="pending",
                metadata_json=metadata or {},
            )
            session.add(doc)
            await session.commit()
            logger.info("创建文档记录: id=%s, filename=%s", doc_id, filename)
            return self._doc_to_dict(doc)

    async def get_document(self, doc_id: str) -> Optional[dict]:
        """获取文档详情"""
        async with self.get_session() as session:
            result = await session.execute(
                select(DocumentModel).where(DocumentModel.id == doc_id)
            )
            doc = result.scalar_one_or_none()
            return self._doc_to_dict(doc) if doc else None

    async def list_documents(
        self,
        knowledge_base: str | None = None,
        status: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict:
        """列出文档"""
        async with self.get_session() as session:
            query = select(DocumentModel)

            if knowledge_base:
                query = query.where(DocumentModel.knowledge_base == knowledge_base)
            if status:
                query = query.where(DocumentModel.status == status)

            query = query.order_by(DocumentModel.created_at.desc())

            # 计算总数
            count_query = select(func.count()).select_from(query.subquery())
            total = (await session.execute(count_query)).scalar() or 0

            # 分页
            items = (await session.execute(
                query.offset((page - 1) * page_size).limit(page_size)
            )).scalars().all()

            return {
                "items": [self._doc_to_dict(doc) for doc in items],
                "total": total,
                "page": page,
                "page_size": page_size,
            }

    async def update_document_status(
        self,
        doc_id: str,
        status: str,
        chunk_count: int | None = None,
        total_tokens: int | None = None,
        error_message: str | None = None,
        title: str | None = None,
    ) -> None:
        """更新文档处理状态"""
        async with self.get_session() as session:
            values = {"status": status, "updated_at": datetime.now(timezone.utc)}
            if chunk_count is not None:
                values["chunk_count"] = chunk_count
            if total_tokens is not None:
                values["total_tokens"] = total_tokens
            if error_message is not None:
                values["error_message"] = error_message
            if title is not None:
                values["title"] = title

            await session.execute(
                update(DocumentModel).where(DocumentModel.id == doc_id).values(**values)
            )
            await session.commit()

    async def delete_document(self, doc_id: str) -> Optional[dict]:
        """删除文档记录，返回被删除的文档信息"""
        async with self.get_session() as session:
            result = await session.execute(
                select(DocumentModel).where(DocumentModel.id == doc_id)
            )
            doc = result.scalar_one_or_none()
            if doc:
                doc_dict = self._doc_to_dict(doc)
                await session.execute(
                    delete(DocumentModel).where(DocumentModel.id == doc_id)
                )
                await session.commit()
                # 更新知识库统计
                await self.update_knowledge_base_stats(
                    doc.knowledge_base, doc_delta=-1, chunk_delta=-doc.chunk_count
                )
                logger.info("删除文档: id=%s", doc_id)
                return doc_dict
            return None

    # ============ 辅助方法 ============

    @staticmethod
    def _kb_to_dict(kb: KnowledgeBaseModel) -> dict:
        return {
            "name": kb.name,
            "description": kb.description,
            "embedding_model": kb.embedding_model,
            "embedding_dimension": kb.embedding_dimension,
            "chunk_size": kb.chunk_size,
            "chunk_overlap": kb.chunk_overlap,
            "document_count": kb.document_count,
            "chunk_count": kb.chunk_count,
            "is_active": kb.is_active,
            "created_at": kb.created_at.isoformat() if kb.created_at else None,
            "updated_at": kb.updated_at.isoformat() if kb.updated_at else None,
        }

    @staticmethod
    def _doc_to_dict(doc: DocumentModel) -> dict:
        return {
            "id": doc.id,
            "filename": doc.filename,
            "file_type": doc.file_type,
            "file_size": doc.file_size,
            "file_path": doc.file_path,
            "knowledge_base": doc.knowledge_base,
            "title": doc.title,
            "status": doc.status,
            "chunk_count": doc.chunk_count,
            "total_tokens": doc.total_tokens,
            "error_message": doc.error_message,
            "metadata": doc.metadata_json or {},
            "created_at": doc.created_at.isoformat() if doc.created_at else None,
            "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
        }
