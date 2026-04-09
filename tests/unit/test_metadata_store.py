"""元数据存储单元测试"""

import pytest

from knowledge_forge.storage.metadata_store import MetadataStore


@pytest.fixture
def metadata_store():
    """创建使用 SQLite 内存数据库的 MetadataStore"""
    store = MetadataStore(database_url="sqlite+aiosqlite://")
    return store


@pytest.fixture
async def connected_store(metadata_store):
    """连接后的 MetadataStore"""
    await metadata_store.connect()
    yield metadata_store
    await metadata_store.disconnect()


class TestKnowledgeBaseCRUD:
    @pytest.mark.asyncio
    async def test_create_knowledge_base(self, connected_store):
        """测试创建知识库"""
        kb = await connected_store.create_knowledge_base(
            name="test_kb",
            description="测试知识库",
        )

        assert kb["name"] == "test_kb"
        assert kb["description"] == "测试知识库"
        assert kb["document_count"] == 0
        assert kb["chunk_count"] == 0
        assert kb["is_active"] is True

    @pytest.mark.asyncio
    async def test_get_knowledge_base(self, connected_store):
        """测试获取知识库"""
        await connected_store.create_knowledge_base(name="test_kb")
        kb = await connected_store.get_knowledge_base("test_kb")

        assert kb is not None
        assert kb["name"] == "test_kb"

    @pytest.mark.asyncio
    async def test_get_nonexistent_kb(self, connected_store):
        """测试获取不存在的知识库"""
        kb = await connected_store.get_knowledge_base("nonexistent")
        assert kb is None

    @pytest.mark.asyncio
    async def test_list_knowledge_bases(self, connected_store):
        """测试列出知识库"""
        await connected_store.create_knowledge_base(name="kb1")
        await connected_store.create_knowledge_base(name="kb2")

        result = await connected_store.list_knowledge_bases()

        assert result["total"] >= 2
        assert len(result["items"]) >= 2

    @pytest.mark.asyncio
    async def test_delete_knowledge_base(self, connected_store):
        """测试删除知识库（软删除）"""
        await connected_store.create_knowledge_base(name="to_delete")
        success = await connected_store.delete_knowledge_base("to_delete")

        assert success is True

        # 软删除后 active_only=True 应查不到
        kb = await connected_store.get_knowledge_base("to_delete")
        assert kb["is_active"] is False

    @pytest.mark.asyncio
    async def test_update_kb_stats(self, connected_store):
        """测试更新知识库统计"""
        await connected_store.create_knowledge_base(name="stats_kb")
        await connected_store.update_knowledge_base_stats("stats_kb", doc_delta=1, chunk_delta=10)

        kb = await connected_store.get_knowledge_base("stats_kb")
        assert kb["document_count"] == 1
        assert kb["chunk_count"] == 10


class TestDocumentCRUD:
    @pytest.mark.asyncio
    async def test_create_document(self, connected_store):
        """测试创建文档记录"""
        doc = await connected_store.create_document(
            filename="test.pdf",
            file_type="pdf",
            file_size=1024,
            file_path="/data/uploads/test.pdf",
            knowledge_base="default",
        )

        assert doc["id"] != ""
        assert doc["filename"] == "test.pdf"
        assert doc["status"] == "pending"
        assert doc["file_type"] == "pdf"

    @pytest.mark.asyncio
    async def test_get_document(self, connected_store):
        """测试获取文档详情"""
        created = await connected_store.create_document(
            filename="test.pdf",
            file_type="pdf",
            file_size=1024,
            file_path="/data/uploads/test.pdf",
        )

        doc = await connected_store.get_document(created["id"])

        assert doc is not None
        assert doc["filename"] == "test.pdf"

    @pytest.mark.asyncio
    async def test_update_document_status(self, connected_store):
        """测试更新文档状态"""
        created = await connected_store.create_document(
            filename="test.pdf",
            file_type="pdf",
            file_size=1024,
            file_path="/data/uploads/test.pdf",
        )

        await connected_store.update_document_status(
            created["id"],
            status="completed",
            chunk_count=15,
            total_tokens=12000,
        )

        doc = await connected_store.get_document(created["id"])
        assert doc["status"] == "completed"
        assert doc["chunk_count"] == 15
        assert doc["total_tokens"] == 12000

    @pytest.mark.asyncio
    async def test_list_documents(self, connected_store):
        """测试列出文档"""
        await connected_store.create_document(
            filename="doc1.pdf", file_type="pdf", file_size=100,
            file_path="/path/1", knowledge_base="kb_a",
        )
        await connected_store.create_document(
            filename="doc2.txt", file_type="txt", file_size=200,
            file_path="/path/2", knowledge_base="kb_b",
        )

        # 全部
        result = await connected_store.list_documents()
        assert result["total"] >= 2

        # 按知识库筛选
        result = await connected_store.list_documents(knowledge_base="kb_a")
        assert all(d["knowledge_base"] == "kb_a" for d in result["items"])

    @pytest.mark.asyncio
    async def test_delete_document(self, connected_store):
        """测试删除文档"""
        created = await connected_store.create_document(
            filename="to_delete.pdf", file_type="pdf", file_size=100,
            file_path="/path/del",
        )

        deleted = await connected_store.delete_document(created["id"])
        assert deleted is not None

        # 删除后应查不到
        doc = await connected_store.get_document(created["id"])
        assert doc is None

    @pytest.mark.asyncio
    async def test_document_status_failed(self, connected_store):
        """测试文档处理失败状态"""
        created = await connected_store.create_document(
            filename="error.pdf", file_type="pdf", file_size=100,
            file_path="/path/err",
        )

        await connected_store.update_document_status(
            created["id"],
            status="failed",
            error_message="解析失败：文件损坏",
        )

        doc = await connected_store.get_document(created["id"])
        assert doc["status"] == "failed"
        assert "文件损坏" in doc["error_message"]
