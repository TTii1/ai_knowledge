"""应用配置 - 基于 Pydantic Settings 的分层配置管理"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置，支持 .env 文件和环境变量"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ===== 应用配置 =====
    app_name: str = "KnowledgeForge"
    app_env: Literal["development", "staging", "production"] = "development"
    debug: bool = True
    log_level: str = "INFO"

    # ===== API 配置 =====
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"

    # ===== LLM 配置 =====
    openai_api_key: str = "sk-xxx"
    openai_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536

    # ===== 向量数据库 =====
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "knowledge_chunks"

    # ===== 关系数据库 =====
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "knowledge_forge"
    postgres_password: str = "changeme"
    postgres_db: str = "knowledge_forge"

    @property
    def postgres_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def postgres_sync_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # ===== Redis =====
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str | None = None

    @property
    def redis_url(self) -> str:
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"

    # ===== 文件存储 =====
    upload_dir: str = "./data/uploads"
    max_upload_size_mb: int = 50

    # ===== RAG 配置 =====
    chunk_size: int = 800
    chunk_overlap: int = 100
    min_chunk_size: int = 100
    retrieval_top_k: int = 20
    rerank_top_k: int = 5
    context_max_tokens: int = 4096

    # ===== 对话配置 =====
    max_conversation_turns: int = 10
    max_conversation_tokens: int = 4096

    # ===== Celery =====
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    @property
    def is_dev(self) -> bool:
        return self.app_env == "development"


@lru_cache
def get_settings() -> Settings:
    """获取全局配置单例"""
    return Settings()
