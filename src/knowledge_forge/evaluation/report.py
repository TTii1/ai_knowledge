"""评估报告生成"""

import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from knowledge_forge.evaluation.metrics import (
    RetrievalMetrics,
    GenerationMetrics,
    PerformanceMetrics,
)

logger = logging.getLogger(__name__)


@dataclass
class EvalReport:
    """评估报告"""
    name: str
    created_at: str = ""
    config: dict = field(default_factory=dict)
    retrieval_metrics: Optional[RetrievalMetrics] = None
    generation_metrics: Optional[GenerationMetrics] = None
    performance_metrics: Optional[PerformanceMetrics] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def save(self, path: Path) -> None:
        """保存报告"""
        data = asdict(self)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("评估报告已保存: %s", path)

    def summary(self) -> str:
        """生成摘要文本"""
        lines = [
            f"# 评估报告: {self.name}",
            f"时间: {self.created_at}",
            "",
        ]

        if self.retrieval_metrics:
            m = self.retrieval_metrics
            lines.extend([
                "## 检索质量",
                f"- Recall@5:  {m.recall_at_5:.2%}",
                f"- Recall@10: {m.recall_at_10:.2%}",
                f"- Recall@20: {m.recall_at_20:.2%}",
                f"- Precision@5:  {m.precision_at_5:.2%}",
                f"- Precision@10: {m.precision_at_10:.2%}",
                f"- MRR: {m.mrr:.4f}",
                f"- Hit Rate: {m.hit_rate:.2%}",
                "",
            ])

        if self.generation_metrics:
            m = self.generation_metrics
            lines.extend([
                "## 生成质量",
                f"- 答案相关性: {m.relevance:.2%}",
                f"- 忠实度: {m.faithfulness:.2%}",
                f"- 完整性: {m.completeness:.2%}",
                "",
            ])

        if self.performance_metrics:
            m = self.performance_metrics
            lines.extend([
                "## 系统性能",
                f"- 端到端延迟: {m.e2e_latency_ms:.0f}ms",
                f"- 检索延迟: {m.retrieval_latency_ms:.0f}ms",
                f"- 生成延迟: {m.generation_latency_ms:.0f}ms",
            ])

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """转为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EvalReport":
        """从字典创建报告"""
        retrieval_metrics = None
        if data.get("retrieval_metrics"):
            retrieval_metrics = RetrievalMetrics(**data["retrieval_metrics"])

        generation_metrics = None
        if data.get("generation_metrics"):
            generation_metrics = GenerationMetrics(**data["generation_metrics"])

        performance_metrics = None
        if data.get("performance_metrics"):
            performance_metrics = PerformanceMetrics(**data["performance_metrics"])

        return cls(
            name=data["name"],
            created_at=data.get("created_at", ""),
            config=data.get("config", {}),
            retrieval_metrics=retrieval_metrics,
            generation_metrics=generation_metrics,
            performance_metrics=performance_metrics,
        )
