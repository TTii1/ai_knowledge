"""文档处理异步任务 - Celery Task

完整流程：解析 → 切分 → 向量化 → 向量存储 → 元数据更新
"""

import asyncio
import logging
from pathlib import Path

from knowledge_forge.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


def _run_async(coro):
    """在同步 Celery 任务中安全运行异步协程

    使用新的事件循环避免与已有事件循环冲突
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # 如果已在异步上下文中，创建新线程运行
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


async def _process_document(file_path: str, knowledge_base: str, doc_id: str):
    """异步处理文档：解析 → 切分 → 向量化 → 存储"""
    from knowledge_forge.api.deps import (
        get_pipeline, get_embedding_service, get_vector_store, get_metadata_store,
    )

    pipeline = get_pipeline()
    embedding_service = get_embedding_service()
    vector_store = get_vector_store()
    metadata_store = get_metadata_store()

    # 更新状态：解析中
    await metadata_store.update_document_status(doc_id, status="processing")

    result = await pipeline.process_and_store(
        file_path=Path(file_path),
        knowledge_base=knowledge_base,
        embedding_service=embedding_service,
        vector_store=vector_store,
        metadata_store=metadata_store,
        doc_id=doc_id,
    )

    return result


@celery_app.task(bind=True, name="process_document", max_retries=3, default_retry_delay=60)
def process_document(self, file_path: str, knowledge_base: str = "default", doc_id: str = ""):
    """异步处理文档：解析 → 切分 → 向量化 → 存储

    Args:
        file_path: 文档路径
        knowledge_base: 知识库名称
        doc_id: 文档 ID（用于状态追踪）
    """
    logger.info("开始处理文档: %s (kb=%s, doc_id=%s)", file_path, knowledge_base, doc_id)

    try:
        # 更新任务状态
        self.update_state(state="PROCESSING", meta={"step": "parsing"})

        result = _run_async(_process_document(file_path, knowledge_base, doc_id))

        logger.info("文档处理完成: %s → %d chunks", file_path, result.get("chunks", 0))
        return result

    except Exception as e:
        logger.error("文档处理失败: %s, error=%s", file_path, str(e))

        # 更新元数据状态为失败
        if doc_id:
            try:
                _run_async(_update_doc_failed(doc_id, str(e)))
            except Exception:
                pass

        # 重试
        try:
            self.retry(exc=e)
        except self.MaxRetriesExceededError:
            logger.error("文档处理重试次数耗尽: %s", file_path)
            raise


async def _update_doc_failed(doc_id: str, error_message: str):
    """更新文档状态为失败"""
    from knowledge_forge.api.deps import get_metadata_store
    metadata_store = get_metadata_store()
    await metadata_store.update_document_status(
        doc_id, status="failed", error_message=error_message
    )
