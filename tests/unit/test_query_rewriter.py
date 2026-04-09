"""Query 改写器单元测试"""

import pytest

from knowledge_forge.conversation.reference_resolver import ReferenceResolver


def test_contains_pronouns():
    """测试代词检测"""
    resolver = ReferenceResolver()

    assert resolver.contains_pronouns("它的优势是什么？")
    assert resolver.contains_pronouns("这个技术怎么样？")
    assert not resolver.contains_pronouns("RAG 技术的核心流程是什么？")
    assert not resolver.contains_pronouns("什么是向量数据库？")


def test_needs_resolution():
    """测试指代消解需求判断"""
    resolver = ReferenceResolver()

    # 有历史 + 有代词 → 需要消解
    assert resolver.needs_resolution("它的优势是什么？", has_history=True)

    # 有历史 + 无代词 → 不需要消解
    assert not resolver.needs_resolution("RAG 的核心流程是什么？", has_history=True)

    # 无历史 + 有代词 → 不需要消解（没有上下文）
    assert not resolver.needs_resolution("它的优势是什么？", has_history=False)
