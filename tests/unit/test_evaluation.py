"""评估模块单元测试"""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from knowledge_forge.evaluation.metrics import (
    RetrievalMetrics,
    GenerationMetrics,
    PerformanceMetrics,
    compute_recall_at_k,
    compute_precision_at_k,
    compute_mrr,
    compute_hit_rate,
    compute_retrieval_metrics,
    compute_performance_metrics,
)
from knowledge_forge.evaluation.dataset import EvalDataset, EvalQuestion
from knowledge_forge.evaluation.report import EvalReport


# ============ 指标计算测试 ============

class TestRetrievalMetrics:
    """检索指标计算测试"""

    def test_recall_at_k_perfect(self):
        """完美召回"""
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = ["a", "b", "c"]
        assert compute_recall_at_k(retrieved, relevant, 5) == 1.0

    def test_recall_at_k_partial(self):
        """部分召回"""
        retrieved = ["x", "y", "a", "b"]
        relevant = ["a", "b", "c"]
        assert compute_recall_at_k(retrieved, relevant, 5) == pytest.approx(2/3)

    def test_recall_at_k_no_match(self):
        """无匹配"""
        retrieved = ["x", "y", "z"]
        relevant = ["a", "b", "c"]
        assert compute_recall_at_k(retrieved, relevant, 5) == 0.0

    def test_recall_at_k_empty_relevant(self):
        """空相关集合"""
        assert compute_recall_at_k(["a"], [], 5) == 0.0

    def test_precision_at_k_perfect(self):
        """完美精度"""
        retrieved = ["a", "b", "c"]
        relevant = ["a", "b", "c", "d"]
        assert compute_precision_at_k(retrieved, relevant, 3) == 1.0

    def test_precision_at_k_partial(self):
        """部分精度"""
        retrieved = ["a", "x", "b"]
        relevant = ["a", "b", "c"]
        assert compute_precision_at_k(retrieved, relevant, 3) == pytest.approx(2/3)

    def test_precision_at_k_zero_k(self):
        """k=0"""
        assert compute_precision_at_k(["a"], ["a"], 0) == 0.0

    def test_mrr_first_position(self):
        """MRR 第一位匹配"""
        assert compute_mrr(["a", "b", "c"], ["a", "d"]) == 1.0

    def test_mrr_second_position(self):
        """MRR 第二位匹配"""
        assert compute_mrr(["x", "a", "b"], ["a", "d"]) == 0.5

    def test_mrr_no_match(self):
        """MRR 无匹配"""
        assert compute_mrr(["x", "y", "z"], ["a", "b"]) == 0.0

    def test_hit_rate_match(self):
        """Hit Rate 命中"""
        assert compute_hit_rate(["a", "b", "c"], ["a"]) == 1.0

    def test_hit_rate_no_match(self):
        """Hit Rate 未命中"""
        assert compute_hit_rate(["x", "y"], ["a"]) == 0.0


class TestBatchMetrics:
    """批量指标计算测试"""

    def test_compute_retrieval_metrics_empty(self):
        """空结果"""
        result = compute_retrieval_metrics([])
        assert result.recall_at_5 == 0.0

    def test_compute_retrieval_metrics_single(self):
        """单条结果"""
        data = [{"retrieved_ids": ["a", "b"], "relevant_ids": ["a"]}]
        result = compute_retrieval_metrics(data)
        assert result.recall_at_5 == 1.0
        assert result.hit_rate == 1.0

    def test_compute_retrieval_metrics_multiple(self):
        """多条结果取平均"""
        data = [
            {"retrieved_ids": ["a"], "relevant_ids": ["a"]},   # recall@5=1, mrr=1
            {"retrieved_ids": ["x"], "relevant_ids": ["a"]},   # recall@5=0, mrr=0
        ]
        result = compute_retrieval_metrics(data)
        assert result.recall_at_5 == 0.5
        assert result.mrr == 0.5

    def test_compute_performance_metrics(self):
        """性能指标计算"""
        data = [
            {"e2e_latency_ms": 100, "retrieval_latency_ms": 30, "generation_latency_ms": 70},
            {"e2e_latency_ms": 200, "retrieval_latency_ms": 60, "generation_latency_ms": 140},
        ]
        result = compute_performance_metrics(data)
        assert result.e2e_latency_ms == 150.0
        assert result.retrieval_latency_ms == 45.0
        assert result.generation_latency_ms == 105.0

    def test_compute_performance_metrics_empty(self):
        """空性能数据"""
        result = compute_performance_metrics([])
        assert result.e2e_latency_ms == 0.0


# ============ 数据集测试 ============

class TestEvalDataset:
    """评估数据集测试"""

    def test_create_dataset(self):
        """创建数据集"""
        ds = EvalDataset(name="test", description="测试")
        assert ds.name == "test"
        assert len(ds.questions) == 0

    def test_add_questions(self):
        """添加问题"""
        ds = EvalDataset(name="test")
        q = EvalQuestion(
            id="q1",
            question="测试问题",
            ground_truth="测试答案",
            difficulty="easy",
            question_type="factual",
        )
        ds.questions.append(q)
        assert len(ds.questions) == 1
        assert ds.questions[0].id == "q1"

    def test_save_and_load(self, tmp_path):
        """保存和加载数据集"""
        ds = EvalDataset(
            name="test_save",
            description="测试保存",
            questions=[
                EvalQuestion(
                    id="q1",
                    question="问题1",
                    ground_truth="答案1",
                    relevant_chunks=["chunk1"],
                ),
            ],
        )
        path = tmp_path / "test_dataset.json"
        ds.save(path)

        # 验证文件存在
        assert path.exists()

        # 加载
        loaded = EvalDataset.load(path)
        assert loaded.name == "test_save"
        assert len(loaded.questions) == 1
        assert loaded.questions[0].question == "问题1"


# ============ 报告测试 ============

class TestEvalReport:
    """评估报告测试"""

    def test_create_report(self):
        """创建报告"""
        report = EvalReport(name="test_report")
        assert report.name == "test_report"
        assert report.created_at  # 自动生成

    def test_report_with_metrics(self):
        """带指标的报告"""
        report = EvalReport(
            name="metrics_report",
            retrieval_metrics=RetrievalMetrics(recall_at_5=0.85, mrr=0.72),
            performance_metrics=PerformanceMetrics(e2e_latency_ms=350),
        )
        assert report.retrieval_metrics.recall_at_5 == 0.85
        assert report.performance_metrics.e2e_latency_ms == 350

    def test_report_summary(self):
        """报告摘要"""
        report = EvalReport(
            name="summary_test",
            retrieval_metrics=RetrievalMetrics(recall_at_5=0.85, mrr=0.72, hit_rate=0.9),
            performance_metrics=PerformanceMetrics(e2e_latency_ms=350),
        )
        summary = report.summary()
        assert "summary_test" in summary
        assert "85.00%" in summary
        assert "350ms" in summary

    def test_report_save_and_load(self, tmp_path):
        """报告保存和从字典恢复"""
        report = EvalReport(
            name="save_test",
            retrieval_metrics=RetrievalMetrics(recall_at_5=0.8),
        )
        path = tmp_path / "report.json"
        report.save(path)
        assert path.exists()

        # 从字典恢复
        data = json.loads(path.read_text(encoding="utf-8"))
        loaded = EvalReport.from_dict(data)
        assert loaded.name == "save_test"
        assert loaded.retrieval_metrics.recall_at_5 == 0.8

    def test_report_to_dict(self):
        """报告转字典"""
        report = EvalReport(
            name="dict_test",
            generation_metrics=GenerationMetrics(relevance=0.9, faithfulness=0.85),
        )
        d = report.to_dict()
        assert d["name"] == "dict_test"
        assert d["generation_metrics"]["relevance"] == 0.9
