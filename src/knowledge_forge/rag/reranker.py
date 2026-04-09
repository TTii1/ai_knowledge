"""Rerank 重排序模块

基于 BGE-reranker-v2-m3 对候选文档重新排序
"""

import logging
from typing import Optional

from knowledge_forge.rag.retriever.base import RetrievedDocument

logger = logging.getLogger(__name__)


class Reranker:
    """重排序器"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        use_fp16: bool = True,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.device = device
        self._model = None

    def _load_model(self):
        """延迟加载模型"""
        if self._model is None:
            from FlagEmbedding import FlagReranker
            self._model = FlagReranker(
                self.model_name,
                use_fp16=self.use_fp16,
            )
            logger.info("Reranker 模型加载完成: %s", self.model_name)

    async def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
        top_k: int = 5,
    ) -> list[RetrievedDocument]:
        """对候选文档重排序

        Args:
            query: 查询文本
            documents: 候选文档列表
            top_k: 返回结果数

        Returns:
            重排序后的文档列表
        """
        if not documents:
            return []

        self._load_model()

        # 构建 (query, doc) 对
        pairs = [[query, doc.content] for doc in documents]

        # 计算相关性分数
        scores = self._model.compute_score(pairs, normalize=True)

        # 确保返回列表
        if isinstance(scores, float):
            scores = [scores]

        # 按分数排序
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # 取 top_k
        result = []
        for doc, score in scored_docs[:top_k]:
            doc.score = float(score)
            doc.metadata["rerank_score"] = float(score)
            result.append(doc)

        logger.info(
            "Rerank 完成: candidates=%d, top_k=%d, best_score=%.4f",
            len(documents), top_k, result[0].score if result else 0,
        )
        return result
