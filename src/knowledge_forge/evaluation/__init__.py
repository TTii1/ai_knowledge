"""评估模块

提供 RAG 系统的量化评估能力：
- 评估数据集管理（创建、加载、保存）
- 评估指标计算（检索质量、生成质量、系统性能）
- 评估引擎（运行评估，串联 RAG → 指标 → 报告）
- 评估报告生成
"""

from knowledge_forge.evaluation.dataset import EvalDataset, EvalQuestion
from knowledge_forge.evaluation.metrics import (
    RetrievalMetrics,
    GenerationMetrics,
    PerformanceMetrics,
    compute_recall_at_k,
    compute_precision_at_k,
    compute_mrr,
    compute_hit_rate,
)
from knowledge_forge.evaluation.report import EvalReport
from knowledge_forge.evaluation.engine import EvalEngine

__all__ = [
    "EvalDataset", "EvalQuestion",
    "RetrievalMetrics", "GenerationMetrics", "PerformanceMetrics",
    "compute_recall_at_k", "compute_precision_at_k", "compute_mrr", "compute_hit_rate",
    "EvalReport", "EvalEngine",
]
