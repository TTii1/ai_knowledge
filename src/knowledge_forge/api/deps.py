"""API 依赖注入 - 统一管理各服务实例的生命周期"""

from functools import lru_cache

from knowledge_forge.config import Settings, get_settings
from knowledge_forge.storage.vector_store import VectorStore
from knowledge_forge.storage.metadata_store import MetadataStore
from knowledge_forge.storage.cache_store import CacheStore
from knowledge_forge.storage.file_store import FileStore
from knowledge_forge.embedding.openai_embedding import OpenAIEmbedding
from knowledge_forge.document.pipeline import DocumentPipeline
from knowledge_forge.rag.query_rewriter import QueryRewriter
from knowledge_forge.rag.retriever.vector_retriever import VectorRetriever
from knowledge_forge.rag.retriever.bm25_retriever import BM25Retriever
from knowledge_forge.rag.retriever.hybrid_retriever import HybridRetriever
from knowledge_forge.rag.reranker import Reranker
from knowledge_forge.rag.context_builder import ContextBuilder
from knowledge_forge.rag.generator import Generator
from knowledge_forge.rag.engine import RAGEngine
from knowledge_forge.rag.conversation_memory import (
    ConversationMemory, InMemoryMemoryStore, RedisMemoryStore,
)


# ============ 全局服务实例 ============

_vector_store: VectorStore | None = None
_metadata_store: MetadataStore | None = None
_cache_store: CacheStore | None = None
_file_store: FileStore | None = None
_embedding_service: OpenAIEmbedding | None = None
_pipeline: DocumentPipeline | None = None
_query_rewriter: QueryRewriter | None = None
_retriever: HybridRetriever | None = None
_reranker: Reranker | None = None
_context_builder: ContextBuilder | None = None
_generator: Generator | None = None
_rag_engine: RAGEngine | None = None
_conversation_memory: ConversationMemory | None = None


def get_app_settings() -> Settings:
    """获取应用配置"""
    return get_settings()


# ============ 基础设施 ============

def get_vector_store() -> VectorStore:
    """获取向量存储实例"""
    global _vector_store
    if _vector_store is None:
        settings = get_settings()
        _vector_store = VectorStore(
            host=settings.milvus_host,
            port=settings.milvus_port,
            collection_name=settings.milvus_collection,
            dimension=settings.embedding_dimension,
        )
    return _vector_store


def get_metadata_store() -> MetadataStore:
    """获取元数据存储实例"""
    global _metadata_store
    if _metadata_store is None:
        settings = get_settings()
        _metadata_store = MetadataStore(database_url=settings.postgres_url)
    return _metadata_store


def get_cache_store() -> CacheStore:
    """获取缓存存储实例"""
    global _cache_store
    if _cache_store is None:
        settings = get_settings()
        _cache_store = CacheStore(redis_url=settings.redis_url)
    return _cache_store


def get_file_store() -> FileStore:
    """获取文件存储实例"""
    global _file_store
    if _file_store is None:
        settings = get_settings()
        _file_store = FileStore(base_dir=settings.upload_dir)
    return _file_store


def get_embedding_service() -> OpenAIEmbedding:
    """获取 Embedding 服务实例"""
    global _embedding_service
    if _embedding_service is None:
        settings = get_settings()
        _embedding_service = OpenAIEmbedding(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            dimension=settings.embedding_dimension,
        )
    return _embedding_service


def get_pipeline() -> DocumentPipeline:
    """获取文档处理流水线实例"""
    global _pipeline
    if _pipeline is None:
        settings = get_settings()
        _pipeline = DocumentPipeline(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            min_chunk_size=settings.min_chunk_size,
        )
    return _pipeline


# ============ RAG 核心组件 ============

def get_query_rewriter() -> QueryRewriter:
    """获取 Query 改写器"""
    global _query_rewriter
    if _query_rewriter is None:
        settings = get_settings()
        _query_rewriter = QueryRewriter(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.llm_model,
        )
    return _query_rewriter


def get_retriever() -> HybridRetriever:
    """获取混合检索器"""
    global _retriever
    if _retriever is None:
        embedding = get_embedding_service()
        vector_store = get_vector_store()

        vector_retriever = VectorRetriever(
            embedding=embedding,
            vector_store=vector_store,
        )

        bm25_retriever = BM25Retriever()

        _retriever = HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
        )
    return _retriever


def get_reranker() -> Reranker:
    """获取 Reranker"""
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker


def get_context_builder() -> ContextBuilder:
    """获取上下文构建器"""
    global _context_builder
    if _context_builder is None:
        settings = get_settings()
        _context_builder = ContextBuilder(
            max_tokens=settings.context_max_tokens,
            model=settings.llm_model,
        )
    return _context_builder


def get_generator() -> Generator:
    """获取 LLM 生成器"""
    global _generator
    if _generator is None:
        settings = get_settings()
        _generator = Generator(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
    return _generator


def get_rag_engine() -> RAGEngine | None:
    """获取 RAG 问答引擎"""
    global _rag_engine
    if _rag_engine is None:
        try:
            settings = get_settings()
            _rag_engine = RAGEngine(
                query_rewriter=get_query_rewriter(),
                retriever=get_retriever(),
                reranker=get_reranker(),
                context_builder=get_context_builder(),
                generator=get_generator(),
                retrieval_top_k=settings.retrieval_top_k,
                rerank_top_k=settings.rerank_top_k,
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("RAG 引擎初始化失败: %s", str(e))
            return None
    return _rag_engine


def get_conversation_memory() -> ConversationMemory:
    """获取对话记忆管理器"""
    global _conversation_memory
    if _conversation_memory is None:
        settings = get_settings()
        # 优先使用 Redis，降级到内存
        try:
            cache_store = get_cache_store()
            if cache_store._client:
                redis_store = RedisMemoryStore(
                    redis_client=cache_store._client,
                    ttl=86400 * 7,  # 7 天
                )
                _conversation_memory = ConversationMemory(
                    store=redis_store,
                    max_turns=settings.max_conversation_turns,
                    max_tokens=settings.max_conversation_tokens,
                )
                import logging
                logging.getLogger(__name__).info("对话记忆: 使用 Redis 存储")
            else:
                raise RuntimeError("Redis 不可用")
        except Exception:
            in_memory_store = InMemoryMemoryStore()
            _conversation_memory = ConversationMemory(
                store=in_memory_store,
                max_turns=settings.max_conversation_turns,
                max_tokens=settings.max_conversation_tokens,
            )
            import logging
            logging.getLogger(__name__).info("对话记忆: 使用内存存储（降级模式）")
    return _conversation_memory


def reset_services() -> None:
    """重置所有服务实例（用于测试或应用关闭）"""
    global _vector_store, _metadata_store, _cache_store, _file_store
    global _embedding_service, _pipeline, _query_rewriter, _retriever
    global _reranker, _context_builder, _generator, _rag_engine
    global _conversation_memory
    _vector_store = None
    _metadata_store = None
    _cache_store = None
    _file_store = None
    _embedding_service = None
    _pipeline = None
    _query_rewriter = None
    _retriever = None
    _reranker = None
    _context_builder = None
    _generator = None
    _rag_engine = None
    _conversation_memory = None
