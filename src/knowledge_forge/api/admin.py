"""管理后台 API"""

import logging

from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/stats", summary="系统统计")
async def get_system_stats():
    """获取系统统计信息"""
    # TODO: 实现系统统计
    return {
        "total_documents": 0,
        "total_chunks": 0,
        "total_knowledge_bases": 0,
        "total_sessions": 0,
    }


@router.get("/evaluation/datasets", summary="评估数据集列表")
async def list_evaluation_datasets():
    """获取评估数据集列表"""
    # TODO: 实现评估数据集列表
    return {"items": []}


@router.post("/evaluation/run", summary="运行评估")
async def run_evaluation():
    """运行 RAG 效果评估"""
    # TODO: 实现评估运行
    return {"status": "not_implemented"}


@router.get("/evaluation/reports/{report_id}", summary="获取评估报告")
async def get_evaluation_report(report_id: str):
    """获取评估报告详情"""
    # TODO: 实现评估报告查询
    return {"report_id": report_id, "status": "not_implemented"}
