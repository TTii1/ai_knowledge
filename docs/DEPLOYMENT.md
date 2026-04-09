# KnowledgeForge 部署文档

## 环境要求

- Python 3.11+
- Docker & Docker Compose
- 4GB+ 内存（Milvus 推荐 8GB）
- 可选：GPU（用于本地 Reranker 模型）

## 方式一：Docker Compose 一键部署

### 1. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，配置：
# - OPENAI_API_KEY
# - 数据库密码
# - Redis 密码（生产环境）
```

### 2. 启动所有服务

```bash
docker compose up -d
```

这将启动：
- Milvus (向量数据库) - 端口 19530
- PostgreSQL (元数据) - 端口 5432
- Redis (缓存/会话) - 端口 6379

### 3. 启动 API 服务

```bash
# 安装依赖
uv sync --python python --index-url https://mirrors.aliyun.com/pypi/simple/ --default-index https://mirrors.aliyun.com/pypi/simple/

# 启动 API
uvicorn knowledge_forge.main:app --host 0.0.0.0 --port 8000
```

### 4. 启动 MCP Server

```bash
python -m knowledge_forge.mcp_server --transport sse --port 9000
```

### 5. 启动管理后台

```bash
streamlit run src/knowledge_forge/dashboard/app.py --server.port 8501
```

## 方式二：Docker 完整部署

### 构建 API 镜像

```bash
docker build -t knowledge-forge-api .
docker run -d --name kf-api \
  -p 8000:8000 \
  --env-file .env \
  --network host \
  knowledge-forge-api
```

### 构建 MCP Server 镜像

```bash
docker build -t knowledge-forge-mcp -f Dockerfile.mcp .
docker run -d --name kf-mcp \
  -p 9000:9000 \
  --env-file .env \
  --network host \
  knowledge-forge-mcp \
  python -m knowledge_forge.mcp_server --transport sse --host 0.0.0.0 --port 9000
```

## 生产环境配置

### 环境变量

```env
# 应用
APP_ENV=production
DEBUG=false
LOG_LEVEL=WARNING

# LLM
OPENAI_API_KEY=sk-your-production-key
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.1

# Milvus
MILVUS_HOST=milvus
MILVUS_PORT=19530

# PostgreSQL
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_USER=knowledge_forge
POSTGRES_PASSWORD=strong-password-here
POSTGRES_DB=knowledge_forge

# Redis
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=redis-password-here

# RAG
RETRIEVAL_TOP_K=20
RERANK_TOP_K=5
CONTEXT_MAX_TOKENS=4096
```

### 性能调优

| 参数 | 开发环境 | 生产环境 | 说明 |
|------|---------|---------|------|
| pool_size | 5 | 20 | PostgreSQL 连接池 |
| max_overflow | 10 | 30 | 连接池溢出 |
| retrieval_top_k | 20 | 20 | 检索候选数 |
| rerank_top_k | 5 | 8 | Rerank 后保留数 |
| context_max_tokens | 4096 | 4096 | 上下文 Token 预算 |
| max_conversation_turns | 10 | 20 | 最大对话轮数 |
| query_cache_ttl | 1800 | 3600 | 查询缓存 TTL（秒）|

### Nginx 反向代理

```nginx
upstream kf_api {
    server 127.0.0.1:8000;
}

upstream kf_dashboard {
    server 127.0.0.1:8501;
}

server {
    listen 80;
    server_name knowledge-forge.example.com;

    # API
    location /api/ {
        proxy_pass http://kf_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # SSE 支持
    location /api/v1/chat/ {
        proxy_pass http://kf_api;
        proxy_set_header Connection '';
        proxy_http_version 1.1;
        proxy_buffering off;
        proxy_cache off;
        chunked_transfer_encoding off;
    }

    # 管理后台
    location /dashboard/ {
        proxy_pass http://kf_dashboard/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # API 文档
    location /docs {
        proxy_pass http://kf_api;
    }
}
```

## 监控

### 健康检查

```bash
curl http://localhost:8000/health
```

### 日志

```bash
# API 日志
docker logs -f kf-api

# Milvus 日志
docker logs -f milvus
```

### 系统统计

```bash
curl http://localhost:8000/api/v1/admin/stats
```

## 备份与恢复

### PostgreSQL 备份

```bash
docker exec postgres pg_dump -U knowledge_forge knowledge_forge > backup.sql
```

### Milvus 备份

参考 [Milvus 备份文档](https://milvus.io/docs/backup_and_restore.md)

## 常见问题

### Q: Milvus 连接超时

确保 `docker compose up` 后等待 30 秒让 Milvus 完全启动。

### Q: OpenAI API 调用失败

1. 检查 `.env` 中的 `OPENAI_API_KEY` 是否正确
2. 如使用代理，配置 `OPENAI_BASE_URL`

### Q: Reranker 模型加载失败

Reranker 为可选依赖，未安装时系统会自动跳过重排序步骤。如需安装：

```bash
uv sync --extra reranker
```

### Q: 中文分词效果不佳

BM25 使用 jieba 分词，可通过自定义词典优化：

```python
import jieba
jieba.load_userdict("custom_dict.txt")
```
