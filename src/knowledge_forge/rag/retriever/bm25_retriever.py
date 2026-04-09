"""BM25 关键词检索器

使用 rank-bm25 库实现，支持中文分词（jieba）
支持从外部数据源构建索引，或手动索引
"""

import logging
from typing import Optional

from knowledge_forge.rag.retriever.base import BaseRetriever, RetrievedDocument

logger = logging.getLogger(__name__)


class BM25Retriever(BaseRetriever):
    """基于 BM25 的关键词检索器

    特点：
    - 延迟加载 jieba 和 rank_bm25（避免未安装时导入报错）
    - 支持手动索引（index 方法）和从 VectorStore 同步加载
    - 安全降级：依赖库未安装或索引未构建时返回空结果
    """

    def __init__(self):
        self._corpus: list[str] = []
        self._bm25 = None
        self._doc_ids: list[str] = []
        self._doc_metadata: list[dict] = []
        self._heading_chains: list[list[str]] = []
        self._jieba_available: Optional[bool] = None
        self._bm25_available: Optional[bool] = None

    def _check_dependencies(self) -> bool:
        """检查依赖库是否可用"""
        if self._jieba_available is None:
            try:
                import jieba  # noqa: F401
                self._jieba_available = True
            except ImportError:
                self._jieba_available = False
                logger.warning("jieba 未安装，BM25 中文分词不可用。安装: pip install jieba")

        if self._bm25_available is None:
            try:
                from rank_bm25 import BM25Okapi  # noqa: F401
                self._bm25_available = True
            except ImportError:
                self._bm25_available = False
                logger.warning("rank-bm25 未安装，BM25 检索不可用。安装: pip install rank-bm25")

        return self._jieba_available and self._bm25_available

    def index(self, documents: list[dict]) -> None:
        """构建 BM25 索引

        Args:
            documents: 文档列表，每个包含:
                - id: 文档 ID
                - content: 文档内容
                - heading_chain (可选): 标题层级
                - metadata (可选): 元数据
        """
        if not self._check_dependencies():
            logger.warning("BM25 依赖库不可用，跳过索引构建")
            return

        from rank_bm25 import BM25Okapi
        import jieba

        self._corpus = [doc.get("content", "") for doc in documents]
        self._doc_ids = [doc.get("id", "") for doc in documents]
        self._doc_metadata = [doc.get("metadata", {}) for doc in documents]
        self._heading_chains = [doc.get("heading_chain", []) for doc in documents]

        # 中文分词
        tokenized_corpus = [list(jieba.cut(text)) for text in self._corpus]
        self._bm25 = BM25Okapi(tokenized_corpus)

        logger.info("BM25 索引构建完成: %d documents", len(documents))

    async def build_index_from_vector_store(
        self,
        vector_store,
        knowledge_base: str = "default",
    ) -> int:
        """从 VectorStore 加载所有 chunks 构建 BM25 索引

        Args:
            vector_store: VectorStore 实例
            knowledge_base: 知识库名称

        Returns:
            索引的文档数
        """
        try:
            # 使用 query 方法获取全量数据
            # 注意：这需要 VectorStore 支持全量查询
            # 这里我们用一个通用的空 embedding 查询 + 大 top_k 来获取尽可能多的结果
            import json

            if not vector_store._collection:
                vector_store.connect()

            collection = vector_store._collection
            collection.load()

            # 查询所有属于该知识库的 chunks
            results = collection.query(
                expr=f'knowledge_base == "{knowledge_base}"',
                output_fields=["id", "content", "heading_chain", "metadata", "document_id", "chunk_index"],
                limit=16384,  # Milvus 最大限制
            )

            if not results:
                logger.info("BM25: 知识库 %s 无数据", knowledge_base)
                return 0

            # 构建 BM25 索引
            documents = []
            for r in results:
                metadata = json.loads(r.get("metadata", "{}")) if isinstance(r.get("metadata"), str) else r.get("metadata", {})
                heading_chain = json.loads(r.get("heading_chain", "[]")) if isinstance(r.get("heading_chain"), str) else r.get("heading_chain", [])
                doc_metadata = {
                    "document_id": r.get("document_id", ""),
                    "chunk_index": r.get("chunk_index", 0),
                    **metadata,
                }
                documents.append({
                    "id": r.get("id", ""),
                    "content": r.get("content", ""),
                    "heading_chain": heading_chain,
                    "metadata": doc_metadata,
                })

            self.index(documents)
            logger.info("BM25 索引从 VectorStore 构建: knowledge_base=%s, docs=%d", knowledge_base, len(documents))
            return len(documents)

        except Exception as e:
            logger.warning("从 VectorStore 构建 BM25 索引失败: %s", str(e))
            return 0

    async def retrieve(
        self,
        query: str,
        top_k: int = 20,
        knowledge_base: str = "default",
    ) -> list[RetrievedDocument]:
        """BM25 关键词检索"""
        if not self._bm25:
            logger.debug("BM25 索引未构建，返回空结果")
            return []

        if not self._check_dependencies():
            return []

        import jieba

        tokenized_query = list(jieba.cut(query))
        scores = self._bm25.get_scores(tokenized_query)

        # 取 top_k
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        documents = [
            RetrievedDocument(
                id=self._doc_ids[i],
                content=self._corpus[i],
                score=float(scores[i]),
                source="bm25",
                heading_chain=self._heading_chains[i] if i < len(self._heading_chains) else [],
                metadata=self._doc_metadata[i] if i < len(self._doc_metadata) else {},
            )
            for i in top_indices
            if scores[i] > 0  # 过滤零分结果
        ]

        logger.info("BM25 检索完成: query='%s...', results=%d", query[:30], len(documents))
        return documents
