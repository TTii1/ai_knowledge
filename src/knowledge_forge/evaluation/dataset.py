"""评估数据集管理"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class EvalQuestion:
    """评估问题"""
    id: str
    question: str
    ground_truth: str
    relevant_chunks: list[str] = field(default_factory=list)
    difficulty: str = "medium"  # easy / medium / hard
    question_type: str = "factual"  # factual / reasoning / multi_hop


@dataclass
class EvalDataset:
    """评估数据集"""
    name: str
    description: str = ""
    questions: list[EvalQuestion] = field(default_factory=list)

    def save(self, path: Path) -> None:
        """保存数据集到文件"""
        data = {
            "name": self.name,
            "description": self.description,
            "questions": [asdict(q) for q in self.questions],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("评估数据集已保存: %s (%d questions)", path, len(self.questions))

    @classmethod
    def load(cls, path: Path) -> "EvalDataset":
        """从文件加载数据集"""
        data = json.loads(path.read_text(encoding="utf-8"))
        questions = [EvalQuestion(**q) for q in data.get("questions", [])]
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            questions=questions,
        )
