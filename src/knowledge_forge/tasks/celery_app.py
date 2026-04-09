"""Celery 配置"""

from celery import Celery
from knowledge_forge.config import get_settings

settings = get_settings()

celery_app = Celery(
    "knowledge_forge",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)

# 自动发现任务模块
celery_app.autodiscover_tasks(["knowledge_forge.tasks"])
