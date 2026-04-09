# KnowledgeForge 📚

> 企业级智能知识库问答系统 — RAG 深度实战 + MCP Server + Agent 架构

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com)
[![MCP](https://img.shields.io/badge/MCP-1.0+-purple.svg)](https://modelcontextprotocol.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## ✨ 核心特性

| 维度 | 常见 RAG Demo | KnowledgeForge |
|------|--------------|----------------|
| 文档处理 | 仅支持 TXT/Markdown | PDF/Word/Markdown/TXT 多格式 |
| 切分策略 | 固定长度切割 | 语义段落切分 + 自定义 chunk/overlap |
| 检索方式 | 单路向量检索 | 多路召回（向量 + BM25）+ Rerank |
| 对话能力 | 单轮问答 | 多轮对话 + 指代消解 + 会话窗口 |
| 集成能力 | 无 | MCP Server 标准接口，可被 Cursor/Claude 调用 |
| 可观测性 | 无 | 评估面板 + A/B 对比实验 + 量化指标 |

## 🏗️ 系统架构

```
用户 Query
  ↓
Query 改写 (指代消解 → LLM改写/HyDE/分解)
  ↓
多路召回 (向量检索 + BM25，异步并行)
  ↓
RRF 融合 (合并去重)
  ↓
Rerank 重排序 (BGE-reranker，可选)
  ↓
上下文构建 (Token预算控制 + 来源注入)
  ↓
LLM 生成 (流式SSE / 非流式)
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repo-url>
cd ai_knowledge

# 安装 uv（Python 包管理器）
pip install uv

# 同步依赖
uv sync --python python --index-url https://mirrors.aliyun.com/pypi/simple/ --default-index https://mirrors.aliyun.com/pypi/simple/

# 可选：安装 Reranker（需要 GPU + torch）
uv sync --extra reranker
```

### 2. 配置

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env，至少配置：
# OPENAI_API_KEY=sk-your-key-here
```

### 3. 启动基础设施

```bash
docker compose up -d  # Milvus + PostgreSQL + Redis
```

### 4. 启动 API 服务

```bash
# 开发模式
uvicorn knowledge_forge.main:app --reload --port 8000

# 或通过 Python
python -m knowledge_forge.main
```

### 5. 启动 MCP Server

```bash
# stdio 模式（供 Cursor/Claude 调用）
python -m knowledge_forge.mcp_server

# SSE 模式（供 HTTP 客户端调用）
python -m knowledge_forge.mcp_server --transport sse --port 9000
```

### 6. 启动管理后台

```bash
streamlit run src/knowledge_forge/dashboard/app.py
```

## 📡 API 概览

### 文档管理

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/v1/documents/upload` | 上传文档 |
| GET | `/api/v1/documents` | 获取文档列表 |
| DELETE | `/api/v1/documents/{doc_id}` | 删除文档 |

### 知识库管理

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/v1/knowledge-bases` | 创建知识库 |
| GET | `/api/v1/knowledge-bases` | 获取知识库列表 |
| DELETE | `/api/v1/knowledge-bases/{kb_name}` | 删除知识库 |

### 问答

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/v1/chat/sessions` | 创建会话 |
| POST | `/api/v1/chat/sessions/{session_id}/messages` | 发送消息（SSE 流式） |
| GET | `/api/v1/chat/sessions/{session_id}/history` | 获取对话历史 |

### 评估

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/v1/admin/evaluation/run` | 运行评估 |
| POST | `/api/v1/admin/evaluation/ab` | A/B 对比实验 |
| GET | `/api/v1/admin/evaluation/reports` | 评估报告列表 |

完整 API 文档：启动服务后访问 `http://localhost:8000/docs`

## 🔧 MCP Server 配置

在 Cursor/Claude 的 MCP 配置中添加：

```json
{
  "mcpServers": {
    "knowledge-forge": {
      "command": "python",
      "args": ["-m", "knowledge_forge.mcp_server"],
      "cwd": "/path/to/ai_knowledge"
    }
  }
}
```

### MCP 工具列表

| 工具 | 描述 |
|------|------|
| `knowledge_query` | 在知识库中查询并生成回答（含 RAG 流水线） |
| `document_search` | 搜索原始文档片段（不经过 LLM） |
| `knowledge_list` | 列出所有知识库 |
| `session_create` | 创建多轮对话会话 |

### MCP 资源

| URI | 描述 |
|-----|------|
| `knowledge://{kb_name}/overview` | 知识库概览 JSON |

## 📊 评估体系

### A/B 对比实验

| 实验 | 配置 | 变量 |
|------|------|------|
| E0 (基线) | 单路向量检索 + 无改写 + 无Rerank | — |
| E1 | 向量检索 + Query改写 | Query 改写 |
| E2 | 混合检索 + 无Rerank | 多路召回 |
| E3 | 混合检索 + Rerank | Rerank |
| E4 | 混合检索 + Query改写 + Rerank | 完整流水线 |
| E5 | E4 + HyDE | HyDE 策略 |

### 评估指标

- **检索质量**: Recall@K, Precision@K, MRR, Hit Rate
- **生成质量**: 答案相关性, 忠实度, 完整性
- **系统性能**: 端到端延迟, 检索延迟, 生成延迟

## 📁 项目结构

```
knowledge-forge/
├── src/knowledge_forge/
│   ├── config/          # 配置管理 (Pydantic Settings)
│   ├── document/        # 文档处理 (解析器 + 切分器 + 流水线)
│   ├── embedding/       # 向量化 (OpenAI / BGE)
│   ├── storage/         # 存储层 (Milvus + PostgreSQL + Redis)
│   ├── rag/             # RAG 核心 (引擎 + 检索 + 重排 + 生成)
│   ├── mcp_server/      # MCP Server (工具 + 资源)
│   ├── evaluation/      # 评估模块 (指标 + 数据集 + 引擎 + 报告)
│   ├── api/             # API 路由 (FastAPI)
│   ├── dashboard/       # 管理后台 (Streamlit)
│   └── tasks/           # 异步任务 (Celery)
├── tests/               # 测试 (101 个单元测试)
├── docs/                # 项目文档
├── scripts/             # 脚本工具
└── data/                # 数据目录
```

## 🧪 测试

```bash
# 运行全部单元测试
pytest tests/unit/ -v

# 运行特定模块测试
pytest tests/unit/test_evaluation.py -v
```

## 🛠️ 技术栈

- **Web**: FastAPI + Uvicorn + SSE-Starlette
- **数据库**: Milvus (向量) + PostgreSQL (元数据) + Redis (缓存/会话)
- **AI**: OpenAI API (Embedding + LLM) + BGE-reranker-v2-m3
- **MCP**: mcp-python-sdk (FastMCP)
- **任务队列**: Celery + Redis
- **管理后台**: Streamlit + Plotly
- **包管理**: uv + pyproject.toml

## 📝 开发进度

- [x] Phase 0: 项目骨架搭建 (82 文件, 11 测试)
- [x] Phase 1: 文档处理管道 (32 测试)
- [x] Phase 2: RAG 问答核心 (61 测试)
- [x] Phase 3: MCP Server 开发 (76 测试)
- [x] Phase 4: 管理后台 + 评估 (101 测试)
- [x] Phase 5: 优化 + 文档 + 部署

## 📄 License

MIT License
