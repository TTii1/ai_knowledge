"""Query 改写器单元测试"""

import pytest

from knowledge_forge.rag.query_rewriter import QueryRewriter, RewriteStrategy


def test_rewrite_strategy_enum():
    """测试改写策略枚举"""
    assert RewriteStrategy.NONE == "none"
    assert RewriteStrategy.LLM_REWRITE == "llm_rewrite"
    assert RewriteStrategy.HYDE == "hyde"
    assert RewriteStrategy.DECOMPOSE == "decompose"


def test_query_rewriter_no_api_key():
    """测试无 API Key 时的安全降级"""
    rewriter = QueryRewriter(api_key=None)
    assert not rewriter.is_available


def test_query_rewriter_invalid_api_key():
    """测试无效 API Key 时不创建客户端"""
    rewriter = QueryRewriter(api_key="sk-xxx")
    assert not rewriter.is_available


@pytest.mark.asyncio
async def test_rewrite_no_client_returns_original():
    """测试无 LLM 客户端时返回原始查询"""
    rewriter = QueryRewriter(api_key=None)
    result = await rewriter.rewrite("什么是 RAG？")
    assert result == ["什么是 RAG？"]


@pytest.mark.asyncio
async def test_rewrite_none_strategy_returns_original():
    """测试 NONE 策略时返回原始查询"""
    rewriter = QueryRewriter(api_key="sk-test-fake")
    result = await rewriter.rewrite("什么是 RAG？", strategy=RewriteStrategy.NONE)
    assert result == ["什么是 RAG？"]


def test_contains_pronouns_heuristic():
    """测试代词检测启发式（内部方法）"""
    rewriter = QueryRewriter(api_key=None)

    # 包含代词的查询
    history = [{"role": "user", "content": "RAG 是什么"}, {"role": "assistant", "content": "RAG 是..."}]

    # 直接测试指代消解（无 LLM 时直接返回原始查询）
    # 有代词 + 有历史，但由于没有客户端，不会调用 LLM
    import asyncio
    result = asyncio.get_event_loop().run_until_complete(
        rewriter._resolve_references("它的优势是什么？", history)
    )
    assert result == "它的优势是什么？"  # 无客户端，返回原始查询


def test_no_pronouns_skip_resolution():
    """测试无代词时跳过指代消解"""
    rewriter = QueryRewriter(api_key=None)
    import asyncio
    result = asyncio.get_event_loop().run_until_complete(
        rewriter._resolve_references("RAG 技术的核心流程是什么？", [])
    )
    assert result == "RAG 技术的核心流程是什么？"
