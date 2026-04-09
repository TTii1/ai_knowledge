"""PostgreSQL 元数据存储（预留）"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class MetadataStore:
    """PostgreSQL 元数据存储

    存储：文档信息、知识库配置、评估数据、对话日志等
    """

    def __init__(self, database_url: str):
        self.database_url = database_url
        self._engine = None

    async def connect(self) -> None:
        """连接数据库并初始化表结构"""
        # TODO: 使用 SQLAlchemy async engine 连接
        # TODO: 使用 Alembic 管理数据库迁移
        logger.info("元数据存储连接（待实现）: %s", self.database_url)

    async def disconnect(self) -> None:
        """断开连接"""
        if self._engine:
            await self._engine.dispose()
            logger.info("元数据存储连接已关闭")
