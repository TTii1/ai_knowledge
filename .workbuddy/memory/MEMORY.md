# KnowledgeForge 项目记忆

## 项目概况
- **项目名称**: KnowledgeForge - 企业级智能知识库问答系统
- **定位**: RAG 深度实战 + MCP Server + Agent 架构
- **项目路径**: d:\AI_Project\ai_knowledge
- **Python 版本**: 3.13.9 (系统 Python)
- **包管理**: uv (0.11.6)
- **虚拟环境**: .venv (uv 管理)
- **项目状态**: ✅ 全部 5 个 Phase 开发完成

## 技术栈
- FastAPI + Pydantic Settings + uvicorn
- Milvus (向量) + PostgreSQL (元数据) + Redis (缓存)
- OpenAI API (Embedding + LLM)
- BGE-reranker-v2-m3 (Rerank, 可选依赖)
- mcp-python-sdk (MCP Server, FastMCP 高级 API)
- Celery + Redis (异步任务)
- Streamlit (管理后台)

## 关键配置
- PyPI 镜像: mirrors.aliyun.com/pypi/simple/ (pypi.org 直连不通)
- uv sync 命令: `uv sync --python python --index-url https://mirrors.aliyun.com/pypi/simple/ --default-index https://mirrors.aliyun.com/pypi/simple/`
- .env 文件需要手动配置 OPENAI_API_KEY
- FlagEmbedding 为可选依赖: `uv sync --extra reranker`

## 开发进度（全部完成）
- [x] Phase 0: 项目骨架搭建 (2026-04-09) - 11 测试 - Git: c09a29a
- [x] Phase 1: 文档处理管道 (2026-04-09) - 32 测试 - Git: 2981592
- [x] Phase 2: RAG 问答核心 (2026-04-09) - 61 测试 - Git: 24f7d14
- [x] Phase 3: MCP Server 开发 (2026-04-09) - 76 测试 - Git: 76836b6
- [x] Phase 4: 管理后台+评估 (2026-04-09) - 101 测试 - Git: 0d0a22f
- [x] Phase 5: 优化+文档+部署 (2026-04-09) - 113 测试 - Git: 1cdc7a9

## 项目结构
- src/knowledge_forge/ - 主包
  - config/ - 配置 (Pydantic Settings)
  - document/ - 文档处理 (parsers, chunker, pipeline)
  - embedding/ - 向量化 (openai_embedding, bge_embedding)
  - storage/ - 存储层 (vector_store, metadata_store, cache_store, file_store)
  - rag/ - RAG 核心模块 (engine, query_rewriter, retriever, reranker, context_builder, generator, conversation_memory, query_cache)
  - mcp_server/ - MCP Server (server, tools/, resources/, __main__.py)
  - evaluation/ - 评估模块 (engine, metrics, dataset, report)
  - api/ - API 层 (chat, documents, knowledge, admin, deps)
  - dashboard/ - Streamlit 管理后台 (app.py)
  - tasks/ - 异步任务 (document_tasks)
- docs/ - 项目文档 (PROJECT_PLAN.md, API.md, MCP.md, DEPLOYMENT.md)
- tests/ - 测试 (unit/ - 113 个单元测试)
- scripts/ - 脚本工具 (run_mcp_server.py, seed_data.py)
- data/ - 数据目录

## 开发注意
- git push 由用户手动操作
- Python 3.13 中 global 声明不支持多行括号语法，需写成多行 global
- Python 3.13 中 Optional 需要显式导入
- SQLite 不支持 pool_size/max_overflow 参数
- datetime.utcnow() 已弃用，使用 datetime.now(timezone.utc)
- Reranker (FlagEmbedding) 为可选依赖，未安装时安全跳过
- tiktoken 未安装时 ContextBuilder 降级为字符数估算
- MCP Server 使用 FastMCP 高级 API (@mcp.tool, @mcp.resource, @mcp.prompt)
- MCP 支持 stdio/SSE/streamable-http 三种传输
- 查询缓存支持 Redis 和 LRU 两种实现
