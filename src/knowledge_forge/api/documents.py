"""文档管理 API - 完整的文档上传、查询、删除接口"""

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile

from knowledge_forge.api.deps import (
    get_app_settings,
    get_file_store,
    get_metadata_store,
    get_pipeline,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# 支持的文件类型
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".md", ".markdown", ".txt", ".text"}


@router.post("/upload", summary="上传文档")
async def upload_documents(
    files: list[UploadFile] = File(..., description="支持 PDF/Word/Markdown/TXT"),
    knowledge_base: Optional[str] = "default",
):
    """上传文档，自动解析、切分、向量化

    流程：
    1. 验证文件格式
    2. 保存文件到本地
    3. 在数据库中创建文档记录
    4. 触发 Celery 异步任务处理（解析→切分→向量化→存储）
    """
    settings = get_app_settings()
    file_store = get_file_store()
    metadata_store = get_metadata_store()
    pipeline = get_pipeline()

    results = []
    errors = []

    for file in files:
        # 验证文件格式
        ext = Path(file.filename or "").suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            errors.append({
                "filename": file.filename,
                "error": f"不支持的文件格式: {ext}，支持: {list(ALLOWED_EXTENSIONS)}",
            })
            continue

        # 验证文件大小
        max_size = settings.max_upload_size_mb * 1024 * 1024
        content = await file.read()
        if len(content) > max_size:
            errors.append({
                "filename": file.filename,
                "error": f"文件过大: {len(content) / 1024 / 1024:.1f}MB，最大 {settings.max_upload_size_mb}MB",
            })
            continue

        try:
            # 1. 保存文件到本地
            file_path = await file_store.save(
                filename=file.filename or "unknown",
                content=content,
                knowledge_base=knowledge_base,
            )

            # 2. 创建文档元数据记录
            file_type = ext.lstrip(".")
            doc_record = await metadata_store.create_document(
                filename=file.filename or "unknown",
                file_type=file_type,
                file_size=len(content),
                file_path=file_path,
                knowledge_base=knowledge_base,
            )
            doc_id = doc_record["id"]

            # 3. 触发异步处理任务
            try:
                from knowledge_forge.tasks.document_tasks import process_document
                task = process_document.delay(
                    file_path=file_path,
                    knowledge_base=knowledge_base,
                    doc_id=doc_id,
                )
                logger.info("异步任务已触发: task_id=%s, doc_id=%s", task.id, doc_id)
            except Exception as e:
                # Celery 不可用时降级为同步处理
                logger.warning("Celery 不可用，降级为同步处理: %s", str(e))
                import asyncio
                from knowledge_forge.api.deps import get_embedding_service, get_vector_store

                asyncio.create_task(_process_document_async(
                    file_path=file_path,
                    knowledge_base=knowledge_base,
                    doc_id=doc_id,
                ))

            results.append({
                "doc_id": doc_id,
                "filename": file.filename,
                "status": "processing",
                "message": "文件已接收，正在后台处理",
            })

        except Exception as e:
            logger.error("文件上传处理失败: %s, error=%s", file.filename, str(e))
            errors.append({
                "filename": file.filename,
                "error": str(e),
            })

    response = {"files": results}
    if errors:
        response["errors"] = errors
    return response


@router.get("", summary="获取文档列表")
async def list_documents(
    knowledge_base: Optional[str] = None,
    status: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
):
    """获取文档列表，支持按知识库和状态筛选"""
    metadata_store = get_metadata_store()
    return await metadata_store.list_documents(
        knowledge_base=knowledge_base,
        status=status,
        page=page,
        page_size=page_size,
    )


@router.get("/{doc_id}", summary="获取文档详情")
async def get_document(doc_id: str):
    """获取文档详情，包含处理状态和 chunk 统计"""
    metadata_store = get_metadata_store()
    doc = await metadata_store.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"文档不存在: {doc_id}")
    return doc


@router.delete("/{doc_id}", summary="删除文档")
async def delete_document(doc_id: str):
    """删除文档及其所有 chunks（包括文件、向量、元数据）"""
    metadata_store = get_metadata_store()
    file_store = get_file_store()

    # 获取文档信息
    doc = await metadata_store.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"文档不存在: {doc_id}")

    # 删除文件
    if doc.get("file_path"):
        await file_store.delete(doc["file_path"])

    # 删除向量（从 Milvus 中删除该文档的所有 chunks）
    try:
        from knowledge_forge.api.deps import get_vector_store
        vector_store = get_vector_store()
        await vector_store.delete_by_document(doc_id, doc.get("knowledge_base", "default"))
    except Exception as e:
        logger.warning("删除向量失败（Milvus 可能未连接）: %s", str(e))

    # 删除元数据记录
    await metadata_store.delete_document(doc_id)

    return {"doc_id": doc_id, "status": "deleted", "message": "文档及其所有数据已删除"}


@router.post("/{doc_id}/reindex", summary="重新索引文档")
async def reindex_document(doc_id: str):
    """重新索引文档（重新切分 + 向量化）"""
    metadata_store = get_metadata_store()
    pipeline = get_pipeline()

    # 获取文档信息
    doc = await metadata_store.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"文档不存在: {doc_id}")

    # 检查文件是否存在
    file_path = doc.get("file_path", "")
    if not file_path or not Path(file_path).exists():
        raise HTTPException(status_code=400, detail="原始文件不存在，无法重新索引")

    # 更新状态为处理中
    await metadata_store.update_document_status(doc_id, status="processing")

    # 触发重新处理
    try:
        from knowledge_forge.tasks.document_tasks import process_document
        task = process_document.delay(
            file_path=file_path,
            knowledge_base=doc.get("knowledge_base", "default"),
            doc_id=doc_id,
        )
    except Exception as e:
        logger.warning("Celery 不可用，降级为同步处理: %s", str(e))
        import asyncio
        asyncio.create_task(_process_document_async(
            file_path=file_path,
            knowledge_base=doc.get("knowledge_base", "default"),
            doc_id=doc_id,
        ))

    return {"doc_id": doc_id, "status": "reindexing", "message": "正在重新索引"}


# ============ 辅助函数 ============

async def _process_document_async(file_path: str, knowledge_base: str, doc_id: str):
    """同步降级的异步文档处理"""
    try:
        from knowledge_forge.api.deps import (
            get_pipeline, get_embedding_service, get_vector_store, get_metadata_store,
        )
        pipeline = get_pipeline()
        embedding_service = get_embedding_service()
        vector_store = get_vector_store()
        metadata_store = get_metadata_store()

        result = await pipeline.process_and_store(
            file_path=Path(file_path),
            knowledge_base=knowledge_base,
            embedding_service=embedding_service,
            vector_store=vector_store,
            metadata_store=metadata_store,
            doc_id=doc_id,
        )
        logger.info("文档异步处理完成: doc_id=%s, result=%s", doc_id, result)
    except Exception as e:
        logger.error("文档异步处理失败: doc_id=%s, error=%s", doc_id, str(e))
        try:
            metadata_store = get_metadata_store()
            await metadata_store.update_document_status(
                doc_id, status="failed", error_message=str(e)
            )
        except Exception:
            pass
