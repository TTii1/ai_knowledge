"""API 依赖注入 - 统一管理各服务实例的生命周期"""

from functools import lru_cache

from knowledge_forge.config import Settings, get_settings
from knowledge_forge.storage.vector_store import VectorStore
from knowledge_forge.storage.metadata_store import MetadataStore
from knowledge_forge.storage.cache_store import CacheStore
from knowledge_forge.storage.file_store import FileStore
from knowledge_forge.embedding.openai_embedding import OpenAIEmbedding
from knowledge_forge.document.pipeline import DocumentPipeline


# ============ 全局服务实例 ============

_vector_store: VectorStore | None = None
_metadata_store: MetadataStore | None = None
_cache_store: CacheStore | None = None
_file_store: FileStore | None = None
_embedding_service: OpenAIEmbedding | None = None
_pipeline: DocumentPipeline | None = None


def get_app_settings() -> Settings:
    """获取应用配置"""
    return get_settings()


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


def reset_services() -> None:
    """重置所有服务实例（用于测试或应用关闭）"""
    global _vector_store, _metadata_store, _cache_store, _file_store, _embedding_service, _pipeline
    _vector_store = None
    _metadata_store = None
    _cache_store = None
    _file_store = None
    _embedding_service = None
    _pipeline = None
