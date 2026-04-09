"""测试配置和 fixtures"""

import pytest
from pathlib import Path


@pytest.fixture
def test_data_dir() -> Path:
    """测试数据目录"""
    return Path(__file__).parent.parent / "data" / "test_docs"


@pytest.fixture
def sample_txt_file(tmp_path) -> Path:
    """创建临时 TXT 测试文件"""
    txt_file = tmp_path / "test.txt"
    txt_file.write_text(
        "RAG 技术概述\n\n"
        "RAG（Retrieval-Augmented Generation）是一种结合检索和生成的技术。\n\n"
        "核心流程包括：文档解析、向量化存储、相似度检索、上下文注入和 LLM 生成。\n\n"
        "RAG 的优势在于可以利用外部知识库，减少 LLM 的幻觉问题，"
        "同时保持回答的时效性和准确性。\n\n"
        "多路召回是 RAG 的重要优化手段，通过结合向量检索和关键词检索，"
        "可以显著提高检索的召回率和准确率。\n\n"
        "Rerank 重排序则是对多路召回的结果进行精排，"
        "使用交叉编码器计算 query 与文档的相关性分数，"
        "从而提升最终结果的质量。",
        encoding="utf-8",
    )
    return txt_file


@pytest.fixture
def sample_markdown_file(tmp_path) -> Path:
    """创建临时 Markdown 测试文件"""
    md_file = tmp_path / "test.md"
    md_file.write_text(
        "# RAG 技术指南\n\n"
        "## 1. 什么是 RAG\n\n"
        "RAG 是一种检索增强生成技术。\n\n"
        "## 2. RAG 的核心流程\n\n"
        "### 2.1 文档处理\n\n"
        "文档处理包括解析、切分和向量化三个步骤。\n\n"
        "### 2.2 检索与生成\n\n"
        "检索阶段使用向量相似度搜索，生成阶段使用 LLM。\n\n"
        "## 3. RAG 的优化策略\n\n"
        "- Query 改写\n"
        "- 多路召回\n"
        "- Rerank 重排序\n"
        "- 上下文压缩\n",
        encoding="utf-8",
    )
    return md_file
