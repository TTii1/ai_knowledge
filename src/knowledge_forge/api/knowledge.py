"""知识库管理 API - 完整的 CRUD 接口"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from knowledge_forge.api.deps import get_metadata_store, get_app_settings

logger = logging.getLogger(__name__)
router = APIRouter()


# ============ 请求模型 ============

class CreateKnowledgeBaseRequest(BaseModel):
    """创建知识库请求"""
    name: str = Field(..., min_length=1, max_length=256, description="知识库名称")
    description: str = Field(default="", max_length=2048, description="知识库描述")
    chunk_size: int = Field(default=800, ge=100, le=4000, description="Chunk 大小")
    chunk_overlap: int = Field(default=100, ge=0, le=500, description="Overlap 大小")
    embedding_model: str = Field(default="text-embedding-3-small", description="Embedding 模型")
    embedding_dimension: int = Field(default=1536, description="向量维度")


# ============ API 接口 ============

@router.post("", summary="创建知识库")
async def create_knowledge_base(request: CreateKnowledgeBaseRequest):
    """创建新的知识库"""
    settings = get_app_settings()
    metadata_store = get_metadata_store()

    # 检查是否已存在
    existing = await metadata_store.get_knowledge_base(request.name)
    if existing:
        raise HTTPException(status_code=409, detail=f"知识库已存在: {request.name}")

    kb = await metadata_store.create_knowledge_base(
        name=request.name,
        description=request.description,
        embedding_model=request.embedding_model,
        embedding_dimension=request.embedding_dimension,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
    )
    return kb


@router.get("", summary="获取知识库列表")
async def list_knowledge_bases(
    page: int = 1,
    page_size: int = 20,
    active_only: bool = True,
):
    """获取所有知识库"""
    metadata_store = get_metadata_store()
    return await metadata_store.list_knowledge_bases(
        page=page, page_size=page_size, active_only=active_only,
    )


@router.get("/{kb_name}", summary="获取知识库详情")
async def get_knowledge_base(kb_name: str):
    """获取知识库详情"""
    metadata_store = get_metadata_store()
    kb = await metadata_store.get_knowledge_base(kb_name)
    if not kb:
        raise HTTPException(status_code=404, detail=f"知识库不存在: {kb_name}")
    return kb


@router.delete("/{kb_name}", summary="删除知识库")
async def delete_knowledge_base(kb_name: str):
    """删除知识库（软删除），同时删除关联的向量数据"""
    metadata_store = get_metadata_store()

    kb = await metadata_store.get_knowledge_base(kb_name)
    if not kb:
        raise HTTPException(status_code=404, detail=f"知识库不存在: {kb_name}")

    # 删除向量数据
    try:
        from knowledge_forge.api.deps import get_vector_store
        vector_store = get_vector_store()
        await vector_store.delete_by_knowledge_base(kb_name)
    except Exception as e:
        logger.warning("删除向量数据失败（Milvus 可能未连接）: %s", str(e))

    # 软删除知识库
    success = await metadata_store.delete_knowledge_base(kb_name)
    if not success:
        raise HTTPException(status_code=500, detail="删除知识库失败")

    return {"name": kb_name, "status": "deleted", "message": "知识库已删除（软删除）"}
