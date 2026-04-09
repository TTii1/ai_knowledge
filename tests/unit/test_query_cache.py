"""查询缓存单元测试"""

import pytest
import asyncio
import time

from knowledge_forge.rag.query_cache import LRUCache, QueryCache


class TestLRUCache:
    """LRU 缓存测试"""

    @pytest.mark.asyncio
    async def test_set_and_get(self):
        """基本存取"""
        cache = LRUCache()
        await cache.set("key1", "value1")
        result = await cache.get("key1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self):
        """获取不存在的 key"""
        cache = LRUCache()
        result = await cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self):
        """删除"""
        cache = LRUCache()
        await cache.set("key1", "value1")
        deleted = await cache.delete("key1")
        assert deleted is True
        result = await cache.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self):
        """删除不存在的 key"""
        cache = LRUCache()
        deleted = await cache.delete("nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_ttl_expired(self):
        """TTL 过期"""
        cache = LRUCache(default_ttl=1)
        await cache.set("key1", "value1", ttl=1)
        result = await cache.get("key1")
        assert result == "value1"

        # 等待过期
        await asyncio.sleep(1.1)
        result = await cache.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        """LRU 淘汰"""
        cache = LRUCache(max_size=3)
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")
        await cache.set("key4", "value4")  # 淘汰 key1

        assert await cache.get("key1") is None
        assert await cache.get("key2") == "value2"
        assert await cache.get("key3") == "value3"
        assert await cache.get("key4") == "value4"

    @pytest.mark.asyncio
    async def test_clear(self):
        """清空缓存"""
        cache = LRUCache()
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        cache.clear()
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None


class TestQueryCache:
    """查询缓存管理器测试"""

    @pytest.mark.asyncio
    async def test_cache_miss(self):
        """缓存未命中"""
        cache = QueryCache(store=LRUCache())
        result = await cache.get_answer("测试问题", "default")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        """缓存命中"""
        cache = QueryCache(store=LRUCache())
        answer = {"answer": "测试答案", "sources": []}
        await cache.set_answer("测试问题", "default", answer)

        result = await cache.get_answer("测试问题", "default")
        assert result is not None
        assert result["answer"] == "测试答案"

    @pytest.mark.asyncio
    async def test_cache_different_kb(self):
        """不同知识库不同缓存"""
        cache = QueryCache(store=LRUCache())
        await cache.set_answer("问题", "kb1", {"answer": "答案1"})
        await cache.set_answer("问题", "kb2", {"answer": "答案2"})

        r1 = await cache.get_answer("问题", "kb1")
        r2 = await cache.get_answer("问题", "kb2")
        assert r1["answer"] == "答案1"
        assert r2["answer"] == "答案2"

    @pytest.mark.asyncio
    async def test_cache_invalidate(self):
        """缓存失效"""
        cache = QueryCache(store=LRUCache())
        await cache.set_answer("问题", "default", {"answer": "答案"})
        deleted = await cache.invalidate("问题", "default")
        assert deleted is True
        result = await cache.get_answer("问题", "default")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_with_ttl(self):
        """带 TTL 的缓存"""
        cache = QueryCache(store=LRUCache(default_ttl=1))
        await cache.set_answer("问题", "default", {"answer": "答案"}, ttl=1)
        result = await cache.get_answer("问题", "default")
        assert result is not None

        await asyncio.sleep(1.1)
        result = await cache.get_answer("问题", "default")
        assert result is None
