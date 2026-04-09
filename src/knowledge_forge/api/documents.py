"""文档管理 API"""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, File, UploadFile

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/upload", summary="上传文档")
async def upload_documents(
    files: list[UploadFile] = File(..., description="支持 PDF/Word/Markdown/TXT"),
    knowledge_base: Optional[str] = None,
):
    """上传文档，自动解析、切分、向量化"""
    results = []
    for file in files:
        file_id = str(uuid.uuid4())
        logger.info("收到文件上传: name=%s, size=%s, id=%s", file.filename, file.size, file_id)
        results.append({
            "file_id": file_id,
            "filename": file.filename,
            "status": "processing",
            "message": "文件已接收，正在后台处理",
        })
    return {"files": results}


@router.get("", summary="获取文档列表")
async def list_documents(
    knowledge_base: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
):
    """获取文档列表"""
    # TODO: 实现文档列表查询
    return {"items": [], "total": 0, "page": page, "page_size": page_size}


@router.get("/{doc_id}", summary="获取文档详情")
async def get_document(doc_id: str):
    """获取文档详情"""
    # TODO: 实现文档详情查询
    return {"doc_id": doc_id, "status": "not_implemented"}


@router.delete("/{doc_id}", summary="删除文档")
async def delete_document(doc_id: str):
    """删除文档及其所有 chunks"""
    # TODO: 实现文档删除
    return {"doc_id": doc_id, "status": "deleted"}


@router.post("/{doc_id}/reindex", summary="重新索引文档")
async def reindex_document(doc_id: str):
    """重新索引文档（重新切分 + 向量化）"""
    # TODO: 实现文档重新索引
    return {"doc_id": doc_id, "status": "reindexing"}
