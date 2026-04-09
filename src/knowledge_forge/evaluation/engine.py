"""评估引擎 — 串联 RAG 问答 + 指标计算 + 报告生成

支持：
1. 使用评估数据集运行 RAG 问答
2. 自动计算检索/生成/性能指标
3. 生成评估报告
4. A/B 对比实验
"""

import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from knowledge_forge.evaluation.dataset import EvalDataset, EvalQuestion
from knowledge_forge.evaluation.metrics import (
    RetrievalMetrics,
    GenerationMetrics,
    PerformanceMetrics,
    compute_retrieval_metrics,
    compute_performance_metrics,
)
from knowledge_forge.evaluation.report import EvalReport

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """评估配置"""
    name: str = "default_eval"
    knowledge_base: str = "default"
    retrieval_top_k: int = 20
    rerank_top_k: int = 5
    enable_rerank: bool = True
    enable_query_rewrite: bool = True
    # 实验编号 (E0-E5)
    experiment_id: str = "E4"


@dataclass
class EvalResult:
    """单个问题的评估结果"""
    question_id: str
    question: str
    ground_truth: str
    predicted_answer: str = ""
    retrieved_ids: list[str] = field(default_factory=list)
    relevant_chunks: list[str] = field(default_factory=list)
    e2e_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    error: str = ""


class EvalEngine:
    """评估引擎

    使用评估数据集运行 RAG 问答，计算指标，生成报告。
    """

    # 实验配置映射
    EXPERIMENT_CONFIGS = {
        "E0": {"enable_rerank": False, "enable_query_rewrite": False, "use_hybrid": False},
        "E1": {"enable_rerank": False, "enable_query_rewrite": True, "use_hybrid": False},
        "E2": {"enable_rerank": False, "enable_query_rewrite": False, "use_hybrid": True},
        "E3": {"enable_rerank": True, "enable_query_rewrite": False, "use_hybrid": True},
        "E4": {"enable_rerank": True, "enable_query_rewrite": True, "use_hybrid": True},
        "E5": {"enable_rerank": True, "enable_query_rewrite": True, "use_hybrid": True, "use_hyde": True},
    }

    def __init__(
        self,
        rag_engine=None,
        output_dir: str = "./data/evaluation",
    ):
        self.rag_engine = rag_engine
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def run_evaluation(
        self,
        dataset: EvalDataset,
        config: Optional[EvalConfig] = None,
    ) -> EvalReport:
        """运行评估

        Args:
            dataset: 评估数据集
            config: 评估配置

        Returns:
            评估报告
        """
        if config is None:
            config = EvalConfig()

        logger.info(
            "开始评估: dataset=%s, config=%s, questions=%d",
            dataset.name, config.name, len(dataset.questions),
        )

        # 运行每个问题的评估
        eval_results: list[EvalResult] = []

        for q in dataset.questions:
            result = await self._evaluate_question(q, config)
            eval_results.append(result)

        # 计算检索指标
        retrieval_data = [
            {
                "retrieved_ids": r.retrieved_ids,
                "relevant_ids": r.relevant_chunks,
            }
            for r in eval_results
        ]
        retrieval_metrics = compute_retrieval_metrics(retrieval_data)

        # 计算性能指标
        perf_data = [
            {
                "e2e_latency_ms": r.e2e_latency_ms,
                "retrieval_latency_ms": r.retrieval_latency_ms,
                "generation_latency_ms": r.generation_latency_ms,
            }
            for r in eval_results
        ]
        performance_metrics = compute_performance_metrics(perf_data)

        # 生成质量指标（简化版，基于规则评估）
        generation_metrics = self._compute_generation_metrics(eval_results)

        # 生成报告
        report = EvalReport(
            name=f"{config.name}_{dataset.name}",
            config=asdict(config),
            retrieval_metrics=retrieval_metrics,
            generation_metrics=generation_metrics,
            performance_metrics=performance_metrics,
        )

        # 保存报告
        report_path = self.output_dir / f"{report.name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        report.save(report_path)

        logger.info(
            "评估完成: %d questions, Recall@5=%.2f%%, MRR=%.4f, latency=%.0fms",
            len(eval_results),
            retrieval_metrics.recall_at_5 * 100,
            retrieval_metrics.mrr,
            performance_metrics.e2e_latency_ms,
        )

        return report

    async def _evaluate_question(
        self,
        question: EvalQuestion,
        config: EvalConfig,
    ) -> EvalResult:
        """评估单个问题"""
        result = EvalResult(
            question_id=question.id,
            question=question.question,
            ground_truth=question.ground_truth,
            relevant_chunks=question.relevant_chunks,
        )

        if self.rag_engine is None:
            result.error = "RAG 引擎未初始化"
            return result

        try:
            start_time = time.time()

            # 运行 RAG 问答
            rag_result = await self.rag_engine.answer(
                query=question.question,
                knowledge_base=config.knowledge_base,
                stream=False,
            )

            result.e2e_latency_ms = (time.time() - start_time) * 1000
            result.predicted_answer = rag_result.answer
            result.retrieved_ids = [doc.id for doc in rag_result.retrieved_documents]

            # 估算子阶段延迟（如果有 metadata）
            if rag_result.metadata:
                result.retrieval_latency_ms = rag_result.metadata.get("retrieval_latency_ms", 0)
                result.generation_latency_ms = rag_result.metadata.get("generation_latency_ms", 0)
            else:
                # 简化估算：检索约占 30%，生成约占 70%
                result.retrieval_latency_ms = result.e2e_latency_ms * 0.3
                result.generation_latency_ms = result.e2e_latency_ms * 0.7

        except Exception as e:
            logger.error("评估问题 %s 失败: %s", question.id, str(e))
            result.error = str(e)

        return result

    def _compute_generation_metrics(
        self,
        results: list[EvalResult],
    ) -> GenerationMetrics:
        """计算生成质量指标（简化版）

        完整实现需要 LLM-as-Judge，这里用规则做基础评估：
        - relevance: 回答非空且非错误
        - faithfulness: 基于检索到文档（有 retrieved_ids）
        - completeness: 回答长度与 ground_truth 的比例
        """
        if not results:
            return GenerationMetrics()

        n = len(results)
        relevance_sum = 0.0
        faithfulness_sum = 0.0
        completeness_sum = 0.0

        for r in results:
            # Relevance: 回答非空且非错误
            if r.predicted_answer and not r.error:
                relevance_sum += 1.0

            # Faithfulness: 有检索文档支撑
            if r.retrieved_ids:
                faithfulness_sum += 1.0

            # Completeness: 回答长度与 ground_truth 的比值（上限1.0）
            if r.ground_truth and r.predicted_answer:
                ratio = min(len(r.predicted_answer) / max(len(r.ground_truth), 1), 1.0)
                completeness_sum += ratio

        return GenerationMetrics(
            relevance=relevance_sum / n,
            faithfulness=faithfulness_sum / n,
            completeness=completeness_sum / n,
        )

    async def run_ab_experiment(
        self,
        dataset: EvalDataset,
        experiment_ids: list[str] | None = None,
    ) -> dict[str, EvalReport]:
        """运行 A/B 对比实验

        Args:
            dataset: 评估数据集
            experiment_ids: 实验编号列表，如 ["E0", "E4"]

        Returns:
            实验编号到报告的映射
        """
        if experiment_ids is None:
            experiment_ids = ["E0", "E4"]

        reports = {}
        for exp_id in experiment_ids:
            config = EvalConfig(
                name=f"experiment_{exp_id}",
                experiment_id=exp_id,
            )

            # 应用实验配置
            exp_config = self.EXPERIMENT_CONFIGS.get(exp_id, {})
            if "enable_rerank" in exp_config:
                config.enable_rerank = exp_config["enable_rerank"]
            if "enable_query_rewrite" in exp_config:
                config.enable_query_rewrite = exp_config["enable_query_rewrite"]

            report = await self.run_evaluation(dataset, config)
            reports[exp_id] = report

        return reports
