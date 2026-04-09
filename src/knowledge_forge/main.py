"""KnowledgeForge 应用入口"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from knowledge_forge.config import get_settings, setup_logging
from knowledge_forge.api.documents import router as documents_router
from knowledge_forge.api.knowledge import router as knowledge_router
from knowledge_forge.api.chat import router as chat_router
from knowledge_forge.api.admin import router as admin_router
from knowledge_forge.api.deps import (
    get_metadata_store,
    get_cache_store,
    get_vector_store,
    reset_services,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    settings = get_settings()
    setup_logging(log_level=settings.log_level)
    logger.info("🚀 KnowledgeForge 启动中... env=%s", settings.app_env)

    # 初始化元数据存储（PostgreSQL）
    try:
        metadata_store = get_metadata_store()
        await metadata_store.connect()
        logger.info("✅ PostgreSQL 连接成功")

        # 确保 default 知识库存在
        default_kb = await metadata_store.get_knowledge_base("default")
        if not default_kb:
            await metadata_store.create_knowledge_base(
                name="default",
                description="默认知识库",
                embedding_model=settings.embedding_model,
                embedding_dimension=settings.embedding_dimension,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
            logger.info("✅ 创建默认知识库 'default'")
    except Exception as e:
        logger.warning("⚠️ PostgreSQL 连接失败（部分功能不可用）: %s", str(e))

    # 初始化 Redis 缓存
    try:
        cache_store = get_cache_store()
        await cache_store.connect()
        logger.info("✅ Redis 连接成功")
    except Exception as e:
        logger.warning("⚠️ Redis 连接失败（会话缓存不可用）: %s", str(e))

    # 初始化 Milvus 向量存储
    try:
        vector_store = get_vector_store()
        vector_store.connect()
        logger.info("✅ Milvus 连接成功")
    except Exception as e:
        logger.warning("⚠️ Milvus 连接失败（向量检索不可用）: %s", str(e))

    logger.info("✅ KnowledgeForge 启动完成")
    yield

    # 清理资源
    logger.info("🔒 KnowledgeForge 关闭中...")
    try:
        metadata_store = get_metadata_store()
        await metadata_store.disconnect()
    except Exception:
        pass

    try:
        cache_store = get_cache_store()
        await cache_store.disconnect()
    except Exception:
        pass

    try:
        vector_store = get_vector_store()
        vector_store.disconnect()
    except Exception:
        pass

    reset_services()
    logger.info("✅ KnowledgeForge 已关闭")


def create_app() -> FastAPI:
    """创建 FastAPI 应用实例"""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description="企业级智能知识库问答系统",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.is_dev else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册路由
    app.include_router(documents_router, prefix=f"{settings.api_prefix}/documents", tags=["文档管理"])
    app.include_router(knowledge_router, prefix=f"{settings.api_prefix}/knowledge-bases", tags=["知识库管理"])
    app.include_router(chat_router, prefix=f"{settings.api_prefix}/chat", tags=["问答"])
    app.include_router(admin_router, prefix=f"{settings.api_prefix}/admin", tags=["管理"])

    # 健康检查
    @app.get("/health", tags=["健康检查"])
    async def health_check():
        """健康检查接口，返回服务状态"""
        from knowledge_forge.api.deps import get_metadata_store, get_cache_store, get_vector_store

        services = {}
        # 检查 PostgreSQL
        try:
            ms = get_metadata_store()
            if ms._engine:
                services["postgresql"] = "connected"
            else:
                services["postgresql"] = "disconnected"
        except Exception:
            services["postgresql"] = "error"

        # 检查 Redis
        try:
            cs = get_cache_store()
            if cs._client:
                services["redis"] = "connected"
            else:
                services["redis"] = "disconnected"
        except Exception:
            services["redis"] = "error"

        # 检查 Milvus
        try:
            vs = get_vector_store()
            if vs._collection:
                services["milvus"] = "connected"
            else:
                services["milvus"] = "disconnected"
        except Exception:
            services["milvus"] = "error"

        return {
            "status": "ok",
            "app": settings.app_name,
            "version": "0.1.0",
            "services": services,
        }

    return app


app = create_app()


def main():
    """启动服务"""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "knowledge_forge.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.is_dev,
    )


if __name__ == "__main__":
    main()
