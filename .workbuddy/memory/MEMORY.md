# KnowledgeForge 项目记忆

## 项目概况
- **项目名称**: KnowledgeForge - 企业级智能知识库问答系统
- **定位**: RAG 深度实战 + MCP Server + Agent 架构
- **项目路径**: d:\AI_Project\ai_knowledge
- **Python 版本**: 3.13.9 (系统 Python)
- **包管理**: uv (0.11.6)
- **虚拟环境**: .venv (uv 管理)

## 技术栈
- FastAPI + Pydantic Settings + uvicorn
- Milvus (向量) + PostgreSQL (元数据) + Redis (缓存)
- OpenAI API (Embedding + LLM)
- BGE-reranker-v2-m3 (Rerank, 可选依赖)
- mcp-python-sdk (MCP Server)
- Celery + Redis (异步任务)
- Streamlit (管理后台)

## 关键配置
- PyPI 镜像: mirrors.aliyun.com/pypi/simple/ (pypi.org 直连不通)
- uv sync 命令: `uv sync --python python --index-url https://mirrors.aliyun.com/pypi/simple/ --default-index https://mirrors.aliyun.com/pypi/simple/`
- .env 文件需要手动配置 OPENAI_API_KEY
- FlagEmbedding 为可选依赖: `uv sync --extra reranker`

## 开发进度
- [x] Phase 0: 项目骨架搭建 (2026-04-09)
  - 82 个文件, 10043 行代码
  - 11 个单元测试全部通过
  - Git 提交: c09a29a
- [x] Phase 1: 文档处理管道 (2026-04-09)
  - Markdown 解析器修复（代码块内 # 不误判）
  - Word 解析器修复（中英文标题兼容）
  - Chunk overlap 修复（不污染 content）
  - MetadataStore 完整实现（SQLAlchemy ORM + CRUD）
  - 文档 API + 知识库 API 完善
  - Pipeline 串联 embedding + vector_store
  - Celery 任务完善（重试+错误处理）
  - 32 个单元测试全部通过
  - Git 提交: 2981592
- [x] Phase 2: RAG 问答核心 (2026-04-09)
  - RAGEngine 问答引擎（完整流水线串联）
  - QueryRewriter 完善（安全降级+4种策略）
  - VectorRetriever 完善（异常安全降级）
  - BM25Retriever 完善（延迟加载+从VS构建索引）
  - HybridRetriever 完善（异步并行+RRF融合）
  - Reranker 完善（异步执行+降级跳过）
  - ContextBuilder 完善（tiktoken降级+Token预算）
  - Generator 完善（API Key安全降级+流式SSE）
  - ConversationMemory 对话记忆（InMemory+Redis+滑动窗口）
  - Chat API 完善（SSE流式+完整RAG+会话CRUD）
  - 依赖注入更新（RAGEngine/Retriever/Reranker/Generator/Memory）
  - 61 个单元测试全部通过
  - Git 提交: 24f7d14
- [ ] Phase 3: MCP Server 开发
- [ ] Phase 4: 管理后台 + 评估
- [ ] Phase 5: 优化 + 文档 + 部署

## 项目结构
- src/knowledge_forge/ - 主包
  - rag/ - RAG 核心模块 (engine, query_rewriter, retriever, reranker, context_builder, generator, conversation_memory)
  - document/ - 文档处理 (parsers, chunker, pipeline)
  - storage/ - 存储层 (vector_store, metadata_store, cache_store, file_store)
  - embedding/ - 向量化 (openai_embedding, bge_embedding)
  - api/ - API 层 (chat, documents, knowledge, admin, deps)
  - config/ - 配置 (settings)
  - tasks/ - 异步任务 (document_tasks)
- docs/ - 项目文档 (PROJECT_PLAN.md)
- tests/ - 测试 (unit/integration/evaluation)
- scripts/ - 脚本工具
- data/ - 数据目录 (uploads/evaluation/test_docs)

## 开发注意
- git push 由用户手动操作
- Python 3.13 中 Optional 需要显式导入（不自动从 typing 导入）
- SQLite 不支持 pool_size/max_overflow 参数
- datetime.utcnow() 已弃用，使用 datetime.now(timezone.utc)
- Reranker (FlagEmbedding) 为可选依赖，未安装时安全跳过
- tiktoken 未安装时 ContextBuilder 降级为字符数估算
