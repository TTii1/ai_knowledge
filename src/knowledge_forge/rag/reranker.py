"""Rerank 重排序模块

基于 BGE-reranker-v2-m3 对候选文档重新排序
支持：
- 本地模型（FlagEmbedding）延迟加载
- API 调用（Cohere/Jina）预留
- 降级策略：模型不可用时跳过 rerank
"""

import asyncio
import logging
from typing import Optional

from knowledge_forge.rag.retriever.base import RetrievedDocument

logger = logging.getLogger(__name__)


class Reranker:
    """重排序器

    支持两种后端：
    1. 本地模型：FlagReranker (BAAI/bge-reranker-v2-m3)
    2. API：预留 Cohere/Jina 等外部 API

    安全降级：模型未安装或加载失败时跳过 rerank
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        use_fp16: bool = True,
        device: Optional[str] = None,
        backend: str = "local",  # local / cohere / jina
    ):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.device = device
        self.backend = backend
        self._model = None
        self._load_attempted = False
        self._load_failed = False

    @property
    def is_available(self) -> bool:
        """Reranker 模型是否可用"""
        return self._model is not None or (not self._load_attempted and not self._load_failed)

    def _load_model(self) -> bool:
        """延迟加载模型，返回是否成功"""
        if self._model is not None:
            return True

        if self._load_failed:
            return False

        self._load_attempted = True

        try:
            if self.backend == "local":
                from FlagEmbedding import FlagReranker
                self._model = FlagReranker(
                    self.model_name,
                    use_fp16=self.use_fp16,
                )
                logger.info("Reranker 模型加载完成: %s", self.model_name)
                return True
            else:
                logger.warning("不支持的 reranker backend: %s", self.backend)
                self._load_failed = True
                return False
        except ImportError:
            logger.warning(
                "FlagEmbedding 未安装，Reranker 不可用。安装: pip install FlagEmbedding"
            )
            self._load_failed = True
            return False
        except Exception as e:
            logger.warning("Reranker 模型加载失败: %s", str(e))
            self._load_failed = True
            return False

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

        # 尝试加载模型
        if not self._load_model():
            logger.info("Reranker 不可用，返回原始排序（截取 top_k=%d）", top_k)
            return documents[:top_k]

        try:
            # 在线程池中运行同步的 compute_score
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._rerank_sync,
                query,
                documents,
                top_k,
            )
            return result
        except Exception as e:
            logger.warning("Rerank 失败，返回原始排序: %s", str(e))
            return documents[:top_k]

    def _rerank_sync(
        self,
        query: str,
        documents: list[RetrievedDocument],
        top_k: int,
    ) -> list[RetrievedDocument]:
        """同步重排序（在线程池中运行）"""
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
