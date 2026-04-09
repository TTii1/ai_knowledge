"""评估指标计算

指标体系：
- 检索质量：Recall@K, Precision@K, MRR, Hit Rate
- 生成质量：Relevance, Faithfulness, Completeness
- 系统性能：端到端延迟, 检索延迟, 生成延迟
"""

import logging
from dataclasses import dataclass

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


def compute_retrieval_metrics(
    results: list[dict],
) -> RetrievalMetrics:
    """批量计算检索指标

    Args:
        results: 每个元素包含 retrieved_ids 和 relevant_ids

    Returns:
        平均检索指标
    """
    if not results:
        return RetrievalMetrics()

    n = len(results)
    total = RetrievalMetrics()

    for r in results:
        retrieved = r.get("retrieved_ids", [])
        relevant = r.get("relevant_ids", [])

        total.recall_at_5 += compute_recall_at_k(retrieved, relevant, 5)
        total.recall_at_10 += compute_recall_at_k(retrieved, relevant, 10)
        total.recall_at_20 += compute_recall_at_k(retrieved, relevant, 20)
        total.precision_at_5 += compute_precision_at_k(retrieved, relevant, 5)
        total.precision_at_10 += compute_precision_at_k(retrieved, relevant, 10)
        total.mrr += compute_mrr(retrieved, relevant)
        total.hit_rate += compute_hit_rate(retrieved, relevant)

    return RetrievalMetrics(
        recall_at_5=total.recall_at_5 / n,
        recall_at_10=total.recall_at_10 / n,
        recall_at_20=total.recall_at_20 / n,
        precision_at_5=total.precision_at_5 / n,
        precision_at_10=total.precision_at_10 / n,
        mrr=total.mrr / n,
        hit_rate=total.hit_rate / n,
    )


def compute_performance_metrics(
    results: list[dict],
) -> PerformanceMetrics:
    """批量计算性能指标

    Args:
        results: 每个元素包含 e2e_latency_ms, retrieval_latency_ms, generation_latency_ms

    Returns:
        平均性能指标
    """
    if not results:
        return PerformanceMetrics()

    n = len(results)
    total = PerformanceMetrics()

    for r in results:
        total.e2e_latency_ms += r.get("e2e_latency_ms", 0)
        total.retrieval_latency_ms += r.get("retrieval_latency_ms", 0)
        total.generation_latency_ms += r.get("generation_latency_ms", 0)

    return PerformanceMetrics(
        e2e_latency_ms=total.e2e_latency_ms / n,
        retrieval_latency_ms=total.retrieval_latency_ms / n,
        generation_latency_ms=total.generation_latency_ms / n,
    )
