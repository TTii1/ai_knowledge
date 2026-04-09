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

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    settings = get_settings()
    setup_logging(log_level=settings.log_level)
    logger.info("🚀 KnowledgeForge 启动中... env=%s", settings.app_env)

    # TODO: 初始化数据库连接
    # TODO: 初始化 Milvus 连接
    # TODO: 初始化 Redis 连接

    logger.info("✅ KnowledgeForge 启动完成")
    yield

    # 清理资源
    logger.info("🔒 KnowledgeForge 关闭中...")
    # TODO: 关闭数据库连接
    # TODO: 关闭 Milvus 连接
    # TODO: 关闭 Redis 连接
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
        return {"status": "ok", "app": settings.app_name, "version": "0.1.0"}

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
