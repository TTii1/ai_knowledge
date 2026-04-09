# KnowledgeForge MCP Server 使用文档

## 概述

KnowledgeForge 的 MCP Server 将知识库查询能力封装为标准 MCP 协议接口，可被 Cursor、Claude 等 AI 工具直接调用。

## 启动方式

### stdio 模式（默认）

供 Cursor/Claude 等桌面工具通过标准输入输出调用：

```bash
python -m knowledge_forge.mcp_server
```

### SSE 模式

供 HTTP 客户端调用：

```bash
python -m knowledge_forge.mcp_server --transport sse --host 0.0.0.0 --port 9000
```

### Streamable HTTP 模式

```bash
python -m knowledge_forge.mcp_server --transport streamable-http --port 9000
```

## 客户端配置

### Cursor

在 Cursor 的 Settings → MCP 中添加：

```json
{
  "mcpServers": {
    "knowledge-forge": {
      "command": "python",
      "args": ["-m", "knowledge_forge.mcp_server"],
      "cwd": "/path/to/ai_knowledge",
      "env": {
        "OPENAI_API_KEY": "sk-your-key"
      }
    }
  }
}
```

### Claude Desktop

编辑 `claude_desktop_config.json`：

```json
{
  "mcpServers": {
    "knowledge-forge": {
      "command": "python",
      "args": ["-m", "knowledge_forge.mcp_server"],
      "cwd": "C:\\path\\to\\ai_knowledge"
    }
  }
}
```

## 工具列表

### 1. knowledge_query

在知识库中查询相关内容并生成回答。

**参数：**

| 参数 | 类型 | 必填 | 默认值 | 描述 |
|------|------|------|--------|------|
| query | string | ✅ | - | 查询问题 |
| knowledge_base | string | ❌ | "default" | 知识库名称 |
| top_k | integer | ❌ | 5 | 返回结果数 |
| session_id | string | ❌ | null | 会话 ID（用于多轮对话） |

**示例调用：**

```
请使用 knowledge_query 工具查询："RAG 技术的核心流程是什么？"
```

**多轮对话示例：**

```
1. 先创建会话：session_create(knowledge_base="default")
   → 返回 session_id: "abc-123"

2. 第一轮查询：knowledge_query(query="RAG 是什么", session_id="abc-123")

3. 后续查询：knowledge_query(query="它的优势是什么？", session_id="abc-123")
   → 自动关联上下文，指代消解
```

### 2. document_search

搜索知识库中的相关文档片段，返回原始内容（不经过 LLM 生成）。

**参数：**

| 参数 | 类型 | 必填 | 默认值 | 描述 |
|------|------|------|--------|------|
| query | string | ✅ | - | 搜索关键词 |
| knowledge_base | string | ❌ | "default" | 知识库名称 |
| top_k | integer | ❌ | 10 | 返回文档片段数 |

**使用场景：** 当需要获取原始文档内容、查看具体段落时使用。

### 3. knowledge_list

列出所有可用的知识库及其基本信息。

**参数：** 无

**返回示例：**

```
共有 2 个知识库：

- **tech-docs** ✅ 启用
  描述: 技术文档知识库
  文档数: 15 | Chunk 数: 320
  Embedding: text-embedding-3-small (dim=1536)

- **product-faq** ✅ 启用
  描述: 产品常见问题
  文档数: 8 | Chunk 数: 150
```

### 4. session_create

创建新的对话会话，返回 session_id。

**参数：**

| 参数 | 类型 | 必填 | 默认值 | 描述 |
|------|------|------|--------|------|
| knowledge_base | string | ❌ | "default" | 知识库名称 |

**返回：** session_id，用于后续 knowledge_query 调用。

## 资源

### knowledge://{kb_name}/overview

返回知识库的概览信息（JSON 格式）。

**字段：**
- name: 知识库名称
- description: 描述
- document_count: 文档数
- chunk_count: Chunk 总数
- embedding_model: Embedding 模型
- embedding_dimension: 向量维度
- is_active: 是否启用
- created_at / updated_at: 时间戳

## Prompt 模板

### rag_qa_prompt

结构化的 RAG 问答提示模板。

**参数：**
- query: 查询问题
- knowledge_base: 知识库名称

## 注意事项

1. **API Key 配置**: MCP Server 启动时需要读取 `.env` 中的 `OPENAI_API_KEY`
2. **后端依赖**: 部分工具（knowledge_query、document_search）需要 Milvus 和数据库服务
3. **降级策略**: 如果后端不可用，工具会返回错误提示而非崩溃
4. **缓存**: 查询结果会自动缓存，重复查询会命中缓存减少延迟
