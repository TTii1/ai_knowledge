# KnowledgeForge API 文档

> 完整 API 文档可在服务启动后访问：http://localhost:8000/docs

## 基础信息

- **Base URL**: `http://localhost:8000/api/v1`
- **Content-Type**: `application/json`
- **认证**: 暂无（开发环境），生产环境建议添加 API Key 认证

## 文档管理

### 上传文档

```
POST /documents/upload
Content-Type: multipart/form-data

参数:
- file: 文件（支持 .pdf, .docx, .md, .txt）
- knowledge_base: 知识库名称（默认 "default"）
```

**响应示例：**

```json
{
  "id": "doc-uuid-xxx",
  "filename": "report.pdf",
  "file_type": "pdf",
  "file_size": 1024000,
  "knowledge_base": "default",
  "status": "pending",
  "message": "文档上传成功，正在后台处理"
}
```

### 获取文档列表

```
GET /documents?knowledge_base=default&page=1&page_size=20
```

**响应示例：**

```json
{
  "items": [
    {
      "id": "doc-uuid-xxx",
      "filename": "report.pdf",
      "file_type": "pdf",
      "status": "completed",
      "chunk_count": 15,
      "total_tokens": 8000,
      "created_at": "2026-04-09T12:00:00"
    }
  ],
  "total": 1,
  "page": 1,
  "page_size": 20
}
```

### 删除文档

```
DELETE /documents/{doc_id}
```

## 知识库管理

### 创建知识库

```
POST /knowledge-bases

Body:
{
  "name": "tech-docs",
  "description": "技术文档知识库",
  "embedding_model": "text-embedding-3-small",
  "embedding_dimension": 1536,
  "chunk_size": 800,
  "chunk_overlap": 100
}
```

### 获取知识库列表

```
GET /knowledge-bases?page=1&page_size=20
```

### 删除知识库

```
DELETE /knowledge-bases/{kb_name}
```

## 问答 API

### 创建会话

```
POST /chat/sessions

Body:
{
  "knowledge_base": "default"
}
```

**响应：**

```json
{
  "session_id": "session-uuid-xxx",
  "knowledge_base": "default",
  "created_at": "2026-04-09T12:00:00"
}
```

### 发送消息（SSE 流式）

```
POST /chat/sessions/{session_id}/messages

Body:
{
  "query": "RAG 技术的核心流程是什么？",
  "knowledge_base": "default",
  "top_k": 5,
  "stream": true
}
```

**流式响应 (SSE)：**

```
data: {"type": "chunk", "content": "RAG"}
data: {"type": "chunk", "content": " 技术"}
data: {"type": "chunk", "content": " 的核心"}
...
data: {"type": "done", "sources": [...], "latency_ms": 1500}
```

**非流式响应：**

```json
{
  "answer": "RAG 技术的核心流程包括：1. 检索相关文档片段...",
  "sources": [
    {
      "document_id": "doc-xxx",
      "source_file": "rag_intro.pdf",
      "heading_chain": ["第1章", "1.1 概述"],
      "score": 0.95,
      "content_preview": "RAG（检索增强生成）是一种..."
    }
  ],
  "latency_ms": 1500
}
```

### 获取对话历史

```
GET /chat/sessions/{session_id}/history
```

## 管理与评估

### 系统统计

```
GET /admin/stats
```

**响应：**

```json
{
  "total_documents": 25,
  "total_chunks": 520,
  "total_knowledge_bases": 3,
  "total_sessions": 12,
  "rag_engine_available": true
}
```

### 运行评估

```
POST /admin/evaluation/run

Body:
{
  "name": "baseline_eval",
  "knowledge_base": "default",
  "experiment_id": "E0"
}
```

### A/B 对比实验

```
POST /admin/evaluation/ab

Body:
{
  "experiment_ids": ["E0", "E4"]
}
```

### 评估报告列表

```
GET /admin/evaluation/reports
```

### 对话日志

```
GET /admin/conversation-logs?page=1&page_size=20
```

## 错误处理

所有 API 遵循标准 HTTP 状态码：

| 状态码 | 描述 |
|--------|------|
| 200 | 成功 |
| 400 | 请求参数错误 |
| 404 | 资源不存在 |
| 422 | 数据验证失败 |
| 500 | 服务器内部错误 |
| 503 | 服务不可用（RAG 引擎未初始化等） |

错误响应格式：

```json
{
  "detail": "错误描述信息"
}
```
