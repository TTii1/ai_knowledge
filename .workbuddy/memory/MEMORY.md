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
- BGE-reranker-v2-m3 (Rerank)
- mcp-python-sdk (MCP Server)
- Celery + Redis (异步任务)
- Streamlit (管理后台)

## 关键配置
- PyPI 镜像: mirrors.aliyun.com/pypi/simple/ (pypi.org 直连不通)
- uv sync 命令: `uv sync --python python --index-url https://mirrors.aliyun.com/pypi/simple/ --default-index https://mirrors.aliyun.com/pypi/simple/`
- .env 文件需要手动配置 OPENAI_API_KEY

## 开发进度
- [x] Phase 0: 项目骨架搭建 (2026-04-09)
  - 82 个文件, 10043 行代码
  - 11 个单元测试全部通过
  - Git 提交: c09a29a
- [ ] Phase 1: 文档处理管道
- [ ] Phase 2: RAG 问答核心
- [ ] Phase 3: MCP Server 开发
- [ ] Phase 4: 管理后台 + 评估
- [ ] Phase 5: 优化 + 文档 + 部署

## 项目结构
- src/knowledge_forge/ - 主包
- docs/ - 项目文档 (PROJECT_PLAN.md)
- tests/ - 测试 (unit/integration/evaluation)
- scripts/ - 脚本工具
- data/ - 数据目录 (uploads/evaluation/test_docs)
