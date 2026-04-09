"""Redis 缓存存储（预留）"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class CacheStore:
    """Redis 缓存

    用途：会话缓存、对话历史窗口管理、查询缓存
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self._client = None

    async def connect(self) -> None:
        """连接 Redis"""
        import redis.asyncio as aioredis
        self._client = aioredis.from_url(self.redis_url)
        await self._client.ping()
        logger.info("Redis 连接成功: %s", self.redis_url)

    async def disconnect(self) -> None:
        """断开连接"""
        if self._client:
            await self._client.close()
            logger.info("Redis 连接已关闭")

    async def get(self, key: str) -> Optional[str]:
        """获取缓存"""
        if not self._client:
            return None
        return await self._client.get(key)

    async def set(self, key: str, value: str, ttl: int = 3600) -> None:
        """设置缓存"""
        if self._client:
            await self._client.set(key, value, ex=ttl)

    async def delete(self, key: str) -> None:
        """删除缓存"""
        if self._client:
            await self._client.delete(key)
