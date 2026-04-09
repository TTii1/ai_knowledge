"""RAG 查询缓存 — 减少重复查询的延迟

支持：
1. 内存缓存（LRU）：开发/测试用
2. Redis 缓存：生产环境
3. 语义缓存：相似查询命中（预留）
"""

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Optional

logger = logging.getLogger(__name__)


class BaseCache(ABC):
    """缓存基类"""

    @abstractmethod
    async def get(self, key: str) -> Optional[str]:
        ...

    @abstractmethod
    async def set(self, key: str, value: str, ttl: int = 3600) -> None:
        ...

    @abstractmethod
    async def delete(self, key: str) -> bool:
        ...


class LRUCache(BaseCache):
    """LRU 内存缓存"""

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, tuple[str, float]] = OrderedDict()

    async def get(self, key: str) -> Optional[str]:
        if key in self._cache:
            value, expire_at = self._cache[key]
            if time.time() < expire_at:
                self._cache.move_to_end(key)
                return value
            else:
                del self._cache[key]
        return None

    async def set(self, key: str, value: str, ttl: int = 3600) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (value, time.time() + (ttl or self.default_ttl))
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    async def delete(self, key: str) -> bool:
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        self._cache.clear()


class RedisCache(BaseCache):
    """Redis 缓存"""

    def __init__(self, redis_client=None, prefix: str = "rag_cache:", default_ttl: int = 3600):
        self._redis = redis_client
        self.prefix = prefix
        self.default_ttl = default_ttl

    async def get(self, key: str) -> Optional[str]:
        if not self._redis:
            return None
        value = await self._redis.get(f"{self.prefix}{key}")
        return value.decode("utf-8") if value else None

    async def set(self, key: str, value: str, ttl: int = 3600) -> None:
        if not self._redis:
            return
        await self._redis.setex(f"{self.prefix}{key}", ttl or self.default_ttl, value)

    async def delete(self, key: str) -> bool:
        if not self._redis:
            return False
        await self._redis.delete(f"{self.prefix}{key}")
        return True


class QueryCache:
    """RAG 查询缓存管理器

    用法：
        cache = QueryCache(store=LRUCache())
        # 检查缓存
        cached = await cache.get_answer("什么是 RAG？", "default")
        if cached:
            return cached
        # 计算并缓存
        answer = await rag_engine.answer(...)
        await cache.set_answer("什么是 RAG？", "default", answer)
    """

    def __init__(self, store: BaseCache):
        self.store = store

    @staticmethod
    def _make_key(query: str, knowledge_base: str, **kwargs) -> str:
        """生成缓存 key"""
        raw = json.dumps({"query": query, "kb": knowledge_base, **kwargs}, sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()

    async def get_answer(
        self, query: str, knowledge_base: str = "default", **kwargs
    ) -> Optional[dict]:
        """获取缓存的回答"""
        key = self._make_key(query, knowledge_base, **kwargs)
        value = await self.store.get(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return None
        return None

    async def set_answer(
        self, query: str, knowledge_base: str, answer: dict,
        ttl: int = 3600, **kwargs,
    ) -> None:
        """缓存回答"""
        key = self._make_key(query, knowledge_base, **kwargs)
        await self.store.set(key, json.dumps(answer, ensure_ascii=False), ttl=ttl)

    async def invalidate(self, query: str, knowledge_base: str = "default", **kwargs) -> bool:
        """使缓存失效"""
        key = self._make_key(query, knowledge_base, **kwargs)
        return await self.store.delete(key)

    async def clear(self) -> None:
        """清空缓存"""
        if isinstance(self.store, LRUCache):
            self.store.clear()
