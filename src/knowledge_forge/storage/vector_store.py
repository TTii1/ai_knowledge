"""Milvus 向量存储"""

import logging
from typing import Optional

from pymilvus import CollectionSchema, DataType, FieldSchema, Collection, connections, utility

from knowledge_forge.document.chunker.base import Chunk

logger = logging.getLogger(__name__)


class VectorStore:
    """Milvus 向量数据库存储"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        collection_name: str = "knowledge_chunks",
        dimension: int = 1536,
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dimension = dimension
        self._collection: Optional[Collection] = None

    def connect(self) -> None:
        """连接 Milvus"""
        connections.connect(
            alias="default",
            host=self.host,
            port=self.port,
        )
        logger.info("Milvus 连接成功: %s:%d", self.host, self.port)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """确保 Collection 存在"""
        if utility.has_collection(self.collection_name):
            self._collection = Collection(self.collection_name)
            logger.info("Collection 已存在: %s", self.collection_name)
        else:
            self._create_collection()

    def _create_collection(self) -> None:
        """创建 Collection"""
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=36),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="knowledge_base", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="heading_chain", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
        ]
        schema = CollectionSchema(fields=fields, description="Knowledge chunks with embeddings")
        self._collection = Collection(name=self.collection_name, schema=schema)

        # 创建索引
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 256},
        }
        self._collection.create_index(field_name="embedding", index_params=index_params)
        self._collection.create_index(field_name="knowledge_base", index_name="kb_index")
        logger.info("Collection 创建成功: %s", self.collection_name)

    async def insert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        knowledge_base: str = "default",
    ) -> int:
        """插入 chunks 及其向量

        Args:
            chunks: Chunk 列表
            embeddings: 对应的向量列表
            knowledge_base: 知识库名称

        Returns:
            插入的记录数
        """
        import json

        if not self._collection:
            self.connect()

        data = [
            {
                "id": chunk.id,
                "content": chunk.content,
                "knowledge_base": knowledge_base,
                "document_id": chunk.metadata.document_id,
                "heading_chain": json.dumps(chunk.heading_chain, ensure_ascii=False),
                "chunk_index": chunk.metadata.chunk_index,
                "metadata": json.dumps({
                    "source_file": chunk.metadata.source_file,
                    "page_number": chunk.metadata.page_number,
                    "file_type": chunk.metadata.file_type,
                    "token_count": chunk.token_count,
                }, ensure_ascii=False),
                "embedding": embedding,
            }
            for chunk, embedding in zip(chunks, embeddings)
        ]

        self._collection.insert(data)
        self._collection.flush()
        logger.info("插入 %d chunks 到 %s", len(data), knowledge_base)
        return len(data)

    async def search(
        self,
        query_embedding: list[float],
        knowledge_base: str = "default",
        top_k: int = 20,
    ) -> list[dict]:
        """向量相似度搜索

        Args:
            query_embedding: 查询向量
            knowledge_base: 知识库名称
            top_k: 返回结果数

        Returns:
            搜索结果列表
        """
        import json

        if not self._collection:
            self.connect()

        self._collection.load()

        search_params = {"metric_type": "COSINE", "params": {"ef": 128}}
        results = self._collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=f'knowledge_base == "{knowledge_base}"',
            output_fields=["id", "content", "heading_chain", "metadata", "document_id", "chunk_index"],
        )

        output = []
        for hit in results[0]:
            entity = hit.entity
            metadata = json.loads(entity.get("metadata", "{}"))
            heading_chain = json.loads(entity.get("heading_chain", "[]"))
            output.append({
                "id": entity.get("id"),
                "content": entity.get("content"),
                "heading_chain": heading_chain,
                "metadata": metadata,
                "document_id": entity.get("document_id"),
                "chunk_index": entity.get("chunk_index"),
                "score": hit.score,
            })

        logger.info("向量搜索完成: top_k=%d, results=%d", top_k, len(output))
        return output

    def disconnect(self) -> None:
        """断开连接"""
        connections.disconnect("default")
        logger.info("Milvus 连接已关闭")
