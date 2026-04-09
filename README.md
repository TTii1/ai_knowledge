# KnowledgeForge 🧠

> 企业级智能知识库问答系统 — RAG 技术深度实战 + MCP Server 开发 + 完整 Agent 架构

## 功能特性

- 🔍 **多格式文档处理**：PDF / Word / Markdown / TXT 自动解析
- ✂️ **智能语义切分**：递归语义切分 + 自定义 chunk/overlap
- 🎯 **深度 RAG 流水线**：Query 改写 → 多路召回 → Rerank 重排序 → 上下文注入 → LLM 生成
- 💬 **多轮对话记忆**：会话窗口管理 + 指代消解
- 🔌 **MCP Server**：标准 MCP 协议接口，可被 Cursor / Claude 直接调用
- 📊 **量化评估**：检索质量 + 生成质量 + 系统性能的完整评估体系

## 快速开始

### 1. 环境准备

```bash
# 安装 uv（Python 包管理器）
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 克隆项目
git clone <repo-url>
cd ai_knowledge

# 创建虚拟环境并安装依赖
uv sync
```

### 2. 启动基础设施

```bash
docker compose up -d
```

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 填入你的 API Key 等配置
```

### 4. 启动服务

```bash
# 启动 API 服务
uv run python -m knowledge_forge.main

# 启动 MCP Server
uv run python -m knowledge_forge.mcp_server
```

## 技术栈

| 类别 | 技术 |
|------|------|
| 语言 | Python 3.11+ |
| Web 框架 | FastAPI |
| 向量数据库 | Milvus |
| 关系数据库 | PostgreSQL |
| 缓存 | Redis |
| Embedding | OpenAI / BGE-M3 |
| Reranker | BGE-reranker-v2-m3 |
| MCP | mcp-python-sdk |

## 项目结构

```
src/knowledge_forge/
├── main.py              # FastAPI 应用入口
├── config/              # 配置管理
├── document/            # 文档处理（解析 + 切分）
├── embedding/           # 向量化
├── storage/             # 存储层（向量/元数据/缓存/文件）
├── rag/                 # RAG 核心（检索 + Rerank + 生成）
├── conversation/        # 对话管理
├── mcp_server/          # MCP Server
├── api/                 # API 路由
├── evaluation/          # 评估模块
└── tasks/               # 异步任务
```

## 开发

```bash
# 运行测试
uv run pytest

# 代码格式化
uv run ruff format .

# 代码检查
uv run ruff check .
```

## License

MIT
