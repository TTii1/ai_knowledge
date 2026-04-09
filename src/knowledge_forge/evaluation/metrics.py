"""评估指标计算

指标体系：
- 检索质量：Recall@K, Precision@K, MRR, Hit Rate
- 生成质量：Relevance, Faithfulness, Completeness
- 系统性能：端到端延迟, 检索延迟, 生成延迟
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    """检索质量指标"""
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    recall_at_20: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank
    hit_rate: float = 0.0


@dataclass
class GenerationMetrics:
    """生成质量指标"""
    relevance: float = 0.0      # 答案与问题的相关程度
    faithfulness: float = 0.0   # 答案是否基于检索内容
    completeness: float = 0.0   # 答案是否完整覆盖问题


@dataclass
class PerformanceMetrics:
    """系统性能指标"""
    e2e_latency_ms: float = 0.0     # 端到端延迟
    retrieval_latency_ms: float = 0.0  # 检索延迟
    generation_latency_ms: float = 0.0  # 生成延迟


def compute_recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """计算 Recall@K"""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = len(top_k & set(relevant_ids))
    return hits / len(relevant_ids)


def compute_precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """计算 Precision@K"""
    if k == 0:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = len(top_k & set(relevant_ids))
    return hits / k


def compute_mrr(
    retrieved_ids: list[str],
    relevant_ids: list[str],
) -> float:
    """计算 MRR (Mean Reciprocal Rank)"""
    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def compute_hit_rate(
    retrieved_ids: list[str],
    relevant_ids: list[str],
) -> float:
    """计算 Hit Rate"""
    top_k_set = set(retrieved_ids)
    return 1.0 if top_k_set & set(relevant_ids) else 0.0
