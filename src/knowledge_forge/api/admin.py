"""管理后台 API — 系统统计 + 评估接口"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from knowledge_forge.api.deps import (
    get_metadata_store,
    get_conversation_memory,
    get_rag_engine,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ============ 请求/响应模型 ============

class EvalConfigRequest(BaseModel):
    """评估配置请求"""
    name: str = "default_eval"
    knowledge_base: str = "default"
    retrieval_top_k: int = 20
    rerank_top_k: int = 5
    enable_rerank: bool = True
    enable_query_rewrite: bool = True
    experiment_id: str = "E4"


class EvalRunResponse(BaseModel):
    """评估运行响应"""
    status: str
    report_name: str
    retrieval_metrics: dict = {}
    generation_metrics: dict = {}
    performance_metrics: dict = {}
    summary: str = ""


# ============ 系统统计 ============

@router.get("/stats", summary="系统统计")
async def get_system_stats():
    """获取系统统计信息"""
    stats = {
        "total_documents": 0,
        "total_chunks": 0,
        "total_knowledge_bases": 0,
        "total_sessions": 0,
        "rag_engine_available": False,
    }

    try:
        metadata_store = get_metadata_store()
        if metadata_store and metadata_store._session_factory:
            # 知识库统计
            kb_result = await metadata_store.list_knowledge_bases(page=1, page_size=1000, active_only=False)
            kbs = kb_result.get("items", [])
            stats["total_knowledge_bases"] = kb_result.get("total", 0)
            stats["total_documents"] = sum(kb.get("document_count", 0) for kb in kbs)
            stats["total_chunks"] = sum(kb.get("chunk_count", 0) for kb in kbs)
    except Exception as e:
        logger.warning("获取知识库统计失败: %s", str(e))

    try:
        memory = get_conversation_memory()
        if memory:
            stats["total_sessions"] = await memory.count_sessions()
    except Exception as e:
        logger.warning("获取会话统计失败: %s", str(e))

    stats["rag_engine_available"] = get_rag_engine() is not None

    return stats


# ============ 评估 API ============

@router.get("/evaluation/datasets", summary="评估数据集列表")
async def list_evaluation_datasets():
    """获取评估数据集列表"""
    from knowledge_forge.evaluation.dataset import EvalDataset
    from knowledge_forge.config import get_settings
    from pathlib import Path

    settings = get_settings()
    eval_dir = Path("./data/evaluation")
    datasets = []

    if eval_dir.exists():
        for f in eval_dir.glob("*.json"):
            try:
                ds = EvalDataset.load(f)
                datasets.append({
                    "filename": f.name,
                    "name": ds.name,
                    "description": ds.description,
                    "question_count": len(ds.questions),
                })
            except Exception as e:
                logger.warning("加载数据集失败: %s, error: %s", f.name, str(e))

    return {"items": datasets, "total": len(datasets)}


@router.post("/evaluation/run", summary="运行评估", response_model=EvalRunResponse)
async def run_evaluation(config: EvalConfigRequest):
    """运行 RAG 效果评估"""
    from knowledge_forge.evaluation.engine import EvalEngine, EvalConfig
    from knowledge_forge.evaluation.dataset import EvalDataset

    rag_engine = get_rag_engine()
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG 引擎未初始化，请检查配置")

    # 加载评估数据集
    eval_dir = Path("./data/evaluation")
    dataset = None

    # 尝试加载默认数据集
    for f in eval_dir.glob("*.json"):
        try:
            ds = EvalDataset.load(f)
            if ds.questions:
                dataset = ds
                break
        except Exception:
            continue

    if dataset is None:
        # 如果没有数据集，创建一个示例数据集
        dataset = _create_sample_dataset()

    eval_config = EvalConfig(
        name=config.name,
        knowledge_base=config.knowledge_base,
        retrieval_top_k=config.retrieval_top_k,
        rerank_top_k=config.rerank_top_k,
        enable_rerank=config.enable_rerank,
        enable_query_rewrite=config.enable_query_rewrite,
        experiment_id=config.experiment_id,
    )

    engine = EvalEngine(rag_engine=rag_engine)
    report = await engine.run_evaluation(dataset, eval_config)

    return EvalRunResponse(
        status="completed",
        report_name=report.name,
        retrieval_metrics=report.retrieval_metrics.__dict__ if report.retrieval_metrics else {},
        generation_metrics=report.generation_metrics.__dict__ if report.generation_metrics else {},
        performance_metrics=report.performance_metrics.__dict__ if report.performance_metrics else {},
        summary=report.summary(),
    )


@router.get("/evaluation/reports", summary="评估报告列表")
async def list_evaluation_reports():
    """获取评估报告列表"""
    eval_dir = Path("./data/evaluation")
    reports = []

    if eval_dir.exists():
        for f in sorted(eval_dir.glob("*.json"), reverse=True):
            try:
                import json
                data = json.loads(f.read_text(encoding="utf-8"))
                # 检查是否是报告（有 retrieval_metrics 字段）
                if "retrieval_metrics" in data:
                    reports.append({
                        "filename": f.name,
                        "name": data.get("name", ""),
                        "created_at": data.get("created_at", ""),
                    })
            except Exception:
                continue

    return {"items": reports, "total": len(reports)}


@router.get("/evaluation/reports/{report_filename}", summary="获取评估报告")
async def get_evaluation_report(report_filename: str):
    """获取评估报告详情"""
    import json

    report_path = Path("./data/evaluation") / report_filename
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="报告不存在")

    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取报告失败: {str(e)}")


@router.post("/evaluation/ab", summary="A/B 对比实验")
async def run_ab_experiment(
    experiment_ids: list[str] = ["E0", "E4"],
    dataset_name: Optional[str] = None,
):
    """运行 A/B 对比实验"""
    from knowledge_forge.evaluation.engine import EvalEngine
    from knowledge_forge.evaluation.dataset import EvalDataset

    rag_engine = get_rag_engine()
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG 引擎未初始化")

    # 加载数据集
    dataset = _load_dataset(dataset_name)
    if dataset is None:
        dataset = _create_sample_dataset()

    engine = EvalEngine(rag_engine=rag_engine)
    reports = await engine.run_ab_experiment(dataset, experiment_ids)

    # 返回对比结果
    comparison = {}
    for exp_id, report in reports.items():
        comparison[exp_id] = {
            "name": report.name,
            "retrieval_metrics": report.retrieval_metrics.__dict__ if report.retrieval_metrics else {},
            "generation_metrics": report.generation_metrics.__dict__ if report.generation_metrics else {},
            "performance_metrics": report.performance_metrics.__dict__ if report.performance_metrics else {},
            "summary": report.summary(),
        }

    return {"comparison": comparison}


# ============ 对话日志 ============

@router.get("/conversation-logs", summary="对话日志列表")
async def list_conversation_logs(page: int = 1, page_size: int = 20):
    """获取对话日志"""
    memory = get_conversation_memory()
    if memory is None:
        return {"items": [], "total": 0}

    try:
        sessions = await memory.list_sessions(page=page, page_size=page_size)
        return sessions
    except Exception as e:
        logger.error("获取对话日志失败: %s", str(e))
        return {"items": [], "total": 0}


# ============ 辅助函数 ============

def _create_sample_dataset():
    """创建示例评估数据集"""
    from knowledge_forge.evaluation.dataset import EvalDataset, EvalQuestion

    return EvalDataset(
        name="sample_dataset",
        description="示例评估数据集",
        questions=[
            EvalQuestion(
                id="q_001",
                question="RAG 技术的核心流程是什么？",
                ground_truth="RAG 的核心流程包括：1. 检索相关文档片段 2. 构建增强上下文 3. LLM 生成回答",
                relevant_chunks=["chunk_rag_001", "chunk_rag_002"],
                difficulty="easy",
                question_type="factual",
            ),
            EvalQuestion(
                id="q_002",
                question="多路召回相比单路向量检索有什么优势？",
                ground_truth="多路召回结合向量检索和关键词检索，能同时覆盖语义相似和精确匹配，提高召回率。",
                relevant_chunks=["chunk_retrieval_001"],
                difficulty="medium",
                question_type="reasoning",
            ),
            EvalQuestion(
                id="q_003",
                question="为什么需要 Rerank 重排序？它的作用是什么？",
                ground_truth="Rerank 对候选文档重新排序，通过交叉编码器计算 query-doc 对的相关性分数，提升排序精度。",
                relevant_chunks=["chunk_rerank_001", "chunk_rerank_002"],
                difficulty="medium",
                question_type="factual",
            ),
        ],
    )


def _load_dataset(name: Optional[str] = None):
    """加载数据集"""
    from knowledge_forge.evaluation.dataset import EvalDataset

    eval_dir = Path("./data/evaluation")
    if not eval_dir.exists():
        return None

    if name:
        path = eval_dir / f"{name}.json"
        if path.exists():
            return EvalDataset.load(path)
        return None

    # 加载第一个可用数据集
    for f in eval_dir.glob("*_dataset.json"):
        try:
            return EvalDataset.load(f)
        except Exception:
            continue

    return None
