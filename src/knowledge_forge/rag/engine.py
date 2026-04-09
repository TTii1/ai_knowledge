"""RAG 问答引擎 — 串联完整的问答流水线

流程：
1. Query 改写（指代消解 + LLM 改写/HyDE/分解）
2. 多路召回（向量检索 + BM25 关键词检索，RRF 融合）
3. Rerank 重排序（BGE-reranker-v2-m3）
4. 上下文构建（Token 预算控制 + 来源注入）
5. LLM 生成（流式/非流式，来源引用标注）
"""

import logging
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional

from knowledge_forge.rag.query_rewriter import QueryRewriter, RewriteStrategy
from knowledge_forge.rag.retriever.base import BaseRetriever, RetrievedDocument
from knowledge_forge.rag.retriever.hybrid_retriever import HybridRetriever
from knowledge_forge.rag.retriever.vector_retriever import VectorRetriever
from knowledge_forge.rag.retriever.bm25_retriever import BM25Retriever
from knowledge_forge.rag.reranker import Reranker
from knowledge_forge.rag.context_builder import ContextBuilder
from knowledge_forge.rag.generator import Generator

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """RAG 问答结果"""
    query: str                          # 原始查询
    resolved_query: str = ""            # 指代消解后的查询
    rewritten_queries: list[str] = field(default_factory=list)  # 改写后的查询列表
    retrieved_documents: list[RetrievedDocument] = field(default_factory=list)  # 检索到的文档
    reranked_documents: list[RetrievedDocument] = field(default_factory=list)   # 重排序后的文档
    context: str = ""                   # 构建的上下文
    answer: str = ""                    # 生成的回答
    sources: list[dict] = field(default_factory=list)  # 引用来源
    latency_ms: float = 0.0             # 总耗时（毫秒）
    metadata: dict = field(default_factory=dict)       # 额外元数据


class RAGEngine:
    """RAG 问答引擎

    串联完整流水线：Query 改写 → 多路召回 → Rerank → 上下文构建 → LLM 生成
    """

    def __init__(
        self,
        query_rewriter: Optional[QueryRewriter] = None,
        retriever: Optional[BaseRetriever] = None,
        reranker: Optional[Reranker] = None,
        context_builder: Optional[ContextBuilder] = None,
        generator: Optional[Generator] = None,
        rewrite_strategy: RewriteStrategy = RewriteStrategy.LLM_REWRITE,
        enable_rerank: bool = True,
        retrieval_top_k: int = 20,
        rerank_top_k: int = 5,
    ):
        self.query_rewriter = query_rewriter
        self.retriever = retriever
        self.reranker = reranker
        self.context_builder = context_builder
        self.generator = generator
        self.rewrite_strategy = rewrite_strategy
        self.enable_rerank = enable_rerank
        self.retrieval_top_k = retrieval_top_k
        self.rerank_top_k = rerank_top_k

    async def answer(
        self,
        query: str,
        knowledge_base: str = "default",
        conversation_history: list[dict] | None = None,
        stream: bool = False,
    ) -> RAGResult | AsyncGenerator[str, None]:
        """执行完整的 RAG 问答流水线

        Args:
            query: 用户查询
            knowledge_base: 知识库名称
            conversation_history: 对话历史
            stream: 是否流式输出

        Returns:
            stream=False 时返回 RAGResult
            stream=True 时返回 AsyncGenerator[str, None]
        """
        start_time = time.time()
        result = RAGResult(query=query)

        try:
            # ========== 1. Query 改写 ==========
            resolved_query = query
            rewritten_queries = [query]

            if self.query_rewriter and conversation_history:
                try:
                    rewritten_queries = await self.query_rewriter.rewrite(
                        query=query,
                        strategy=self.rewrite_strategy,
                        conversation_history=conversation_history,
                    )
                    resolved_query = rewritten_queries[0] if rewritten_queries else query
                    result.resolved_query = resolved_query
                    result.rewritten_queries = rewritten_queries
                    logger.info("Query 改写完成: '%s' → %s", query, rewritten_queries)
                except Exception as e:
                    logger.warning("Query 改写失败，使用原始查询: %s", str(e))
                    resolved_query = query
                    result.resolved_query = query
            else:
                result.resolved_query = query

            # ========== 2. 多路召回 ==========
            all_documents: list[RetrievedDocument] = []

            if self.retriever:
                # 对每个改写后的 query 进行检索，合并去重
                seen_ids: set[str] = set()
                for q in rewritten_queries:
                    docs = await self.retriever.retrieve(
                        query=q,
                        top_k=self.retrieval_top_k,
                        knowledge_base=knowledge_base,
                    )
                    for doc in docs:
                        if doc.id not in seen_ids:
                            seen_ids.add(doc.id)
                            all_documents.append(doc)

                result.retrieved_documents = all_documents
                logger.info("多路召回完成: queries=%d, total_docs=%d", len(rewritten_queries), len(all_documents))

            # ========== 3. Rerank 重排序 ==========
            reranked_docs = all_documents

            if self.enable_rerank and self.reranker and all_documents:
                try:
                    reranked_docs = await self.reranker.rerank(
                        query=resolved_query,
                        documents=all_documents,
                        top_k=self.rerank_top_k,
                    )
                    result.reranked_documents = reranked_docs
                    logger.info("Rerank 完成: candidates=%d, top_k=%d", len(all_documents), len(reranked_docs))
                except Exception as e:
                    logger.warning("Rerank 失败，使用原始排序: %s", str(e))
                    reranked_docs = all_documents[:self.rerank_top_k]
                    result.reranked_documents = reranked_docs
            else:
                # 无 Reranker 时直接截取 top_k
                reranked_docs = all_documents[:self.rerank_top_k]
                result.reranked_documents = reranked_docs

            # ========== 4. 上下文构建 ==========
            context = ""
            if self.context_builder:
                context = self.context_builder.build(
                    query=resolved_query,
                    documents=reranked_docs,
                    conversation_history=conversation_history,
                )
            else:
                # 降级：简单拼接
                context = "\n\n".join(doc.content for doc in reranked_docs)

            result.context = context

            # 构建来源信息
            result.sources = self._build_sources(reranked_docs)

            # ========== 5. LLM 生成 ==========
            if self.generator:
                system_prompt = ""
                if self.context_builder:
                    system_prompt = self.context_builder.build_system_prompt()

                if stream:
                    # 流式输出
                    return self._stream_answer(
                        result=result,
                        query=resolved_query,
                        context=context,
                        system_prompt=system_prompt,
                        conversation_history=conversation_history,
                        start_time=start_time,
                    )
                else:
                    # 非流式输出
                    answer = await self.generator.generate(
                        query=resolved_query,
                        context=context,
                        system_prompt=system_prompt,
                        conversation_history=conversation_history,
                    )
                    result.answer = answer
            else:
                result.answer = "未配置 LLM 生成器，无法生成回答。"

            result.latency_ms = (time.time() - start_time) * 1000
            logger.info(
                "RAG 问答完成: query='%s...', answer_len=%d, latency=%.0fms",
                query[:30], len(result.answer), result.latency_ms,
            )

        except Exception as e:
            logger.error("RAG 问答失败: %s", str(e), exc_info=True)
            result.answer = f"抱歉，问答过程中发生错误：{str(e)}"
            result.metadata["error"] = str(e)
            result.latency_ms = (time.time() - start_time) * 1000

        return result

    async def _stream_answer(
        self,
        result: RAGResult,
        query: str,
        context: str,
        system_prompt: str,
        conversation_history: list[dict] | None,
        start_time: float,
    ) -> AsyncGenerator[str, None]:
        """流式生成回答"""
        try:
            stream = self.generator.generate_stream(
                query=query,
                context=context,
                system_prompt=system_prompt,
                conversation_history=conversation_history,
            )
            full_answer = []
            async for chunk in stream:
                full_answer.append(chunk)
                yield chunk

            result.answer = "".join(full_answer)
            result.latency_ms = (time.time() - start_time) * 1000
            logger.info("流式 RAG 问答完成: answer_len=%d, latency=%.0fms", len(result.answer), result.latency_ms)

        except Exception as e:
            logger.error("流式生成失败: %s", str(e), exc_info=True)
            yield f"\n\n[错误] 生成过程中发生异常：{str(e)}"

    @staticmethod
    def _build_sources(documents: list[RetrievedDocument]) -> list[dict]:
        """构建来源引用信息"""
        sources = []
        seen = set()
        for doc in documents:
            source_key = f"{doc.metadata.get('source_file', '')}:{doc.id}"
            if source_key not in seen:
                seen.add(source_key)
                sources.append({
                    "document_id": doc.metadata.get("document_id", ""),
                    "source_file": doc.metadata.get("source_file", ""),
                    "heading_chain": doc.heading_chain,
                    "score": doc.score,
                    "content_preview": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                })
        return sources
