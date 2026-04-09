"""文档处理异步任务"""

import logging
from pathlib import Path

from knowledge_forge.tasks.celery_app import celery_app
from knowledge_forge.document.pipeline import DocumentPipeline

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="process_document")
def process_document(self, file_path: str, knowledge_base: str = "default"):
    """异步处理文档：解析 → 切分 → 向量化 → 存储

    Args:
        file_path: 文档路径
        knowledge_base: 知识库名称
    """
    import asyncio

    logger.info("开始处理文档: %s (kb=%s)", file_path, knowledge_base)

    try:
        # 更新任务状态
        self.update_state(state="PROCESSING", meta={"step": "parsing"})

        pipeline = DocumentPipeline()
        chunks = asyncio.run(pipeline.process(Path(file_path)))

        self.update_state(state="PROCESSING", meta={"step": "embedding", "chunks": len(chunks)})

        # TODO: 向量化 + 存储
        # embedding_service = ...
        # vector_store = ...
        # embeddings = await embedding_service.embed_texts([c.content for c in chunks])
        # await vector_store.insert_chunks(chunks, embeddings, knowledge_base)

        logger.info("文档处理完成: %s → %d chunks", file_path, len(chunks))
        return {"file_path": file_path, "chunks": len(chunks), "status": "completed"}

    except Exception as e:
        logger.error("文档处理失败: %s, error=%s", file_path, str(e))
        self.update_state(state="FAILURE", meta={"error": str(e)})
        raise
