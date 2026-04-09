# 智能知识库助手 — 项目规划文档

> **项目代号**：KnowledgeForge  
> **文档版本**：v1.0  
> **创建日期**：2026-04-09  
> **技术栈**：Python 3.13 / FastAPI / Milvus / MCP SDK

---

## 一、项目概览

### 1.1 项目定位

一个企业级智能知识库问答系统，涵盖 **RAG 技术深度实战 + MCP Server 开发 + 完整 Agent 架构**，不是简单的「调 API + 存向量」demo，而是具备 Query 改写、多路召回、Rerank 重排序等深度优化的生产级项目。

### 1.2 核心价值主张

| 维度 | 常见 RAG Demo | 本项目 |
|------|--------------|--------|
| 文档处理 | 仅支持 TXT/Markdown | PDF/Word/Markdown/TXT 多格式，含表格与图片提取 |
| 切分策略 | 固定长度切割 | 语义段落切分 + 自定义 chunk/overlap + 上下文保留 |
| 检索方式 | 单路向量检索 | 多路召回（向量 + 关键词 + 知识图谱）+ Rerank 重排序 |
| 对话能力 | 单轮问答 | 多轮对话 + 指代消解 + 会话窗口管理 |
| 集成能力 | 无 | MCP Server 标准接口，可被 Cursor/Claude 等工具直接调用 |
| 可观测性 | 无 | 问答效果评估面板 + 检索质量量化指标 |

### 1.3 项目目标

- **功能完整性**：7 大核心模块全部可运行、可演示
- **工程质量**：虚拟环境隔离、配置分层、日志规范、异常处理、类型标注
- **量化验证**：每个 RAG 优化环节都有对比数据（召回率、准确率、响应延迟）
- **可集成性**：MCP Server 可被外部 AI 工具一键接入

---

## 二、系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        客户端层                              │
│   ┌──────────┐  ┌──────────────┐  ┌───────────────────────┐ │
│   │ Web 前端  │  │ MCP Client   │  │  管理 Dashboard       │ │
│   │ (Vue/React)│  │(Cursor/Claude)│  │  (Streamlit/Gradio)  │ │
│   └─────┬─────┘  └──────┬───────┘  └──────────┬───────────┘ │
│         │               │                      │             │
└─────────┼───────────────┼──────────────────────┼─────────────┘
          │               │                      │
          ▼               ▼                      ▼
┌─────────────────────────────────────────────────────────────┐
│                        API 网关层                            │
│              FastAPI + WebSocket + SSE                       │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌────────────────────────┐
│  文档处理服务  │ │  RAG 问答服务 │ │    MCP Server 服务     │
│              │ │              │ │                        │
│ ┌──────────┐ │ │ ┌──────────┐ │ │  ┌──────────────────┐  │
│ │文档解析器 │ │ │ │Query 改写│ │ │  │ knowledge_query  │  │
│ │PDF/Word  │ │ │ │  ↓       │ │ │  │ document_search  │  │
│ │MD/TXT    │ │ │ │多路召回  │ │ │  │ knowledge_list   │  │
│ └────┬─────┘ │ │ │  ↓       │ │ │  │ session_create   │  │
│      ▼       │ │ │Rerank    │ │ │  └──────────────────┘  │
│ ┌──────────┐ │ │ │  ↓       │ │ │                        │
│ │语义切分器 │ │ │ │上下文注入│ │ │  MCP Protocol (stdio)  │
│ └────┬─────┘ │ │ │  ↓       │ │ │                        │
│      ▼       │ │ │LLM 生成  │ │ └────────────────────────┘
│ ┌──────────┐ │ │ └──────────┘ │
│ │Embedding │ │ │              │
│ │向量化    │ │ │ ┌──────────┐ │
│ └──────────┘ │ │ │对话记忆  │ │
│              │ │ │管理器    │ │
│              │ │ └──────────┘ │
└──────┬───────┘ └──────┬───────┘
       │                │
       ▼                ▼
┌─────────────────────────────────────────────────────────────┐
│                        数据存储层                            │
│  ┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌─────────┐ │
│  │  Milvus  │  │  PostgreSQL  │  │  Redis   │  │ MinIO/  │ │
│  │ 向量数据库│  │  元数据存储   │  │ 会话缓存 │  │ 本地文件 │ │
│  └──────────┘  └──────────────┘  └──────────┘  └─────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 核心数据流

```
文档上传 → 解析 → 切分 → Embedding → 存入 Milvus
                                      ↓
用户提问 → Query改写 → 多路召回(向量+BM25) → Rerank → 上下文注入 → LLM生成 → 流式响应
                                      ↑
                              对话历史/会话记忆
```

---

## 三、技术选型

### 3.1 核心技术栈

| 类别 | 技术选型 | 选型理由 |
|------|---------|---------|
| **语言** | Python 3.11+ | AI 生态最完善，类型提示支持好 |
| **Web 框架** | FastAPI | 异步高性能，自动 OpenAPI 文档，SSE/WebSocket 原生支持 |
| **向量数据库** | Milvus (Docker) | 企业级，支持混合检索，社区活跃；开发期可用 Chroma 轻量替代 |
| **关系数据库** | PostgreSQL | 元数据存储（文档信息、用户配置、评估数据） |
| **缓存** | Redis | 会话缓存、对话历史窗口管理 |
| **对象存储** | 本地文件系统（MinIO 可选） | 文档原文存储 |
| **Embedding** | OpenAI text-embedding-3-small / BGE-M3 | 兼顾效果与成本，BGE-M3 支持中英双语 |
| **LLM** | OpenAI GPT-4o-mini / DeepSeek-V3 | 问答生成、Query 改写、摘要 |
| **Reranker** | BGE-reranker-v2-m3 | 中英双语重排序，开源可用 |
| **MCP SDK** | mcp-python-sdk | 官方 Python MCP 协议实现 |
| **前端** | Streamlit（MVP）/ Vue3（正式） | 快速验证用 Streamlit，正式版用 Vue |
| **任务队列** | Celery + Redis | 文档异步处理（解析、切分、向量化） |

### 3.2 开发工具链

| 工具 | 用途 |
|------|------|
| `uv` | Python 包管理器，替代 pip，速度快 |
| `ruff` | 代码格式化 + lint |
| `pytest` | 单元测试 + 集成测试 |
| `pre-commit` | Git 提交前自动检查 |
| `Docker Compose` | 本地开发环境编排（Milvus/PG/Redis） |

---

## 四、项目目录结构

```
knowledge-forge/
├── .venv/                          # 虚拟环境（uv 管理）
├── pyproject.toml                  # 项目配置 + 依赖声明
├── uv.lock                         # 锁定依赖版本
├── docker-compose.yml              # 本地基础设施编排
├── .env.example                    # 环境变量模板
├── .env                            # 环境变量（gitignore）
├── .gitignore
├── README.md
│
├── docs/                           # 项目文档
│   ├── PROJECT_PLAN.md             # 本规划文档
│   ├── API.md                      # API 接口文档
│   ├── MCP.md                      # MCP Server 使用文档
│   └── DEPLOYMENT.md               # 部署文档
│
├── src/
│   └── knowledge_forge/            # 主包
│       ├── __init__.py
│       ├── main.py                 # FastAPI 应用入口
│       ├── config/                 # 配置管理
│       │   ├── __init__.py
│       │   ├── settings.py         # Pydantic Settings 配置
│       │   └── logging.py          # 日志配置
│       │
│       ├── document/               # 文档处理模块
│       │   ├── __init__.py
│       │   ├── parsers/            # 文档解析器
│       │   │   ├── __init__.py
│       │   │   ├── base.py         # 解析器基类
│       │   │   ├── pdf_parser.py
│       │   │   ├── word_parser.py
│       │   │   ├── markdown_parser.py
│       │   │   └── txt_parser.py
│       │   ├── chunker/            # 文档切分器
│       │   │   ├── __init__.py
│       │   │   ├── base.py         # 切分器基类
│       │   │   ├── semantic_chunker.py    # 语义切分
│       │   │   └── recursive_chunker.py   # 递归字符切分
│       │   └── pipeline.py         # 文档处理流水线
│       │
│       ├── embedding/              # 向量化模块
│       │   ├── __init__.py
│       │   ├── base.py             # Embedding 基类
│       │   ├── openai_embedding.py
│       │   └── bge_embedding.py    # 本地 BGE 模型
│       │
│       ├── storage/                # 存储模块
│       │   ├── __init__.py
│       │   ├── vector_store.py     # Milvus 向量存储
│       │   ├── metadata_store.py   # PostgreSQL 元数据存储
│       │   ├── cache_store.py      # Redis 缓存
│       │   └── file_store.py       # 文件存储
│       │
│       ├── rag/                    # RAG 核心模块
│       │   ├── __init__.py
│       │   ├── query_rewriter.py   # Query 改写
│       │   ├── retriever/          # 检索器
│       │   │   ├── __init__.py
│       │   │   ├── base.py         # 检索器基类
│       │   │   ├── vector_retriever.py    # 向量检索
│       │   │   ├── bm25_retriever.py      # BM25 关键词检索
│       │   │   └── hybrid_retriever.py    # 混合检索
│       │   ├── reranker.py         # Rerank 重排序
│       │   ├── context_builder.py  # 上下文构建与注入
│       │   └── generator.py        # LLM 回答生成
│       │
│       ├── conversation/           # 对话管理模块
│       │   ├── __init__.py
│       │   ├── session.py          # 会话管理
│       │   ├── memory.py           # 对话记忆（窗口策略）
│       │   └── reference_resolver.py  # 指代消解
│       │
│       ├── mcp_server/             # MCP Server 模块
│       │   ├── __init__.py
│       │   ├── server.py           # MCP Server 主入口
│       │   ├── tools/              # MCP 工具定义
│       │   │   ├── __init__.py
│       │   │   ├── knowledge_query.py
│       │   │   ├── document_search.py
│       │   │   └── knowledge_manage.py
│       │   └── resources/          # MCP 资源定义
│       │       ├── __init__.py
│       │       └── knowledge_base.py
│       │
│       ├── api/                    # API 路由层
│       │   ├── __init__.py
│       │   ├── deps.py             # 依赖注入
│       │   ├── documents.py        # 文档管理 API
│       │   ├── knowledge.py        # 知识库 API
│       │   ├── chat.py             # 问答 API（含 SSE 流式）
│       │   └── admin.py            # 管理后台 API
│       │
│       ├── evaluation/             # 评估模块
│       │   ├── __init__.py
│       │   ├── metrics.py          # 评估指标计算
│       │   ├── dataset.py          # 评估数据集管理
│       │   └── report.py           # 评估报告生成
│       │
│       └── tasks/                  # 异步任务
│           ├── __init__.py
│           ├── celery_app.py       # Celery 配置
│           └── document_tasks.py   # 文档处理任务
│
├── tests/                          # 测试
│   ├── conftest.py                 # 测试配置 + fixtures
│   ├── unit/                       # 单元测试
│   │   ├── test_parsers.py
│   │   ├── test_chunkers.py
│   │   ├── test_retriever.py
│   │   ├── test_reranker.py
│   │   └── test_query_rewriter.py
│   ├── integration/                # 集成测试
│   │   ├── test_rag_pipeline.py
│   │   ├── test_mcp_server.py
│   │   └── test_api.py
│   └── evaluation/                 # RAG 效果评估测试
│       ├── test_recall_quality.py
│       └── test_answer_quality.py
│
├── scripts/                        # 脚本工具
│   ├── setup_dev.sh                # 开发环境初始化
│   ├── seed_data.py                # 种子数据导入
│   └── run_mcp_server.py           # 启动 MCP Server
│
├── data/                           # 数据目录（gitignore）
│   ├── uploads/                    # 上传文档
│   ├── evaluation/                 # 评估数据集
│   └── test_docs/                  # 测试文档
│
└── frontend/                       # 前端（后期）
    └── ...
```

---

## 五、核心模块设计

### 5.1 文档处理模块

#### 5.1.1 文档解析器

```python
# 设计模式：策略模式，统一接口，不同解析器可插拔
class BaseParser(ABC):
    @abstractmethod
    async def parse(self, file_path: Path) -> ParsedDocument:
        """解析文档，返回结构化文档对象"""
        ...

@dataclass
class ParsedDocument:
    title: str
    content: str                    # 纯文本内容
    sections: list[DocumentSection] # 分段信息
    metadata: dict                  # 元数据（页数、作者等）
    tables: list[Table]             # 提取的表格
    source_file: str
```

| 格式 | 解析库 | 特殊处理 |
|------|--------|---------|
| PDF | `pymupdf` (PyMuPDF) | 表格提取、OCR 图片文字 |
| Word | `python-docx` | 标题层级、表格 |
| Markdown | `mistune` | 标题层级、代码块保留 |
| TXT | 内置 | 段落识别 |

#### 5.1.2 智能文档切分器

```
核心策略：递归语义切分（Recursive Semantic Chunking）

1. 首先按文档结构（标题/章节）做一级切分
2. 对超长章节按段落做二级切分
3. 对超长段落按句子做三级切分
4. 每个chunk保留：
   - 自身内容
   - 前后overlap内容（可配置overlap_size）
   - 文档层级上下文（标题链：H1 > H2 > 当前段落）
   - 元数据（来源、页码、位置）
```

**Chunk 数据结构**：
```python
@dataclass
class Chunk:
    id: str
    content: str                    # chunk 主内容
    context_before: str             # 前置overlap
    context_after: str              # 后置overlap
    heading_chain: list[str]        # 标题层级链 ["第1章", "1.1 概述"]
    metadata: ChunkMetadata         # 来源、页码、位置
    token_count: int                # token 数量
```

**可配置参数**：
- `chunk_size`: 500 / 800 / 1000 tokens（默认 800）
- `chunk_overlap`: 50 / 100 / 200 tokens（默认 100）
- `min_chunk_size`: 100 tokens（过小的合并）

### 5.2 向量化与存储模块

#### 5.2.1 Embedding 抽象层

```python
class BaseEmbedding(ABC):
    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...

    @abstractmethod
    async def embed_query(self, query: str) -> list[float]:
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...
```

**支持的 Embedding 模型**：

| 模型 | 维度 | 特点 | 适用场景 |
|------|------|------|---------|
| text-embedding-3-small | 1536 | OpenAI，效果好 | 有 API Key 的场景 |
| BGE-M3 (本地) | 1024 | 开源，支持中英 | 离线/隐私敏感场景 |

#### 5.2.2 Milvus Collection 设计

```
Collection: knowledge_chunks
├── id             (VARCHAR, PRIMARY KEY)    # chunk UUID
├── content        (VARCHAR)                 # chunk 内容
├── knowledge_base (VARCHAR)                 # 所属知识库
├── document_id    (VARCHAR)                 # 来源文档ID
├── heading_chain  (VARCHAR)                 # 标题层级链 JSON
├── chunk_index    (INT64)                   # chunk 序号
├── metadata       (VARCHAR)                 # 元数据 JSON
├── embedding      (FLOAT_VECTOR, dim=1536)  # 向量
└── 索引:
    ├── embedding: IVF_FLAT / HNSW
    └── knowledge_base: 普通索引（标量过滤）
```

### 5.3 RAG 核心模块

#### 5.3.1 完整 RAG 流水线

```
用户 Query
    │
    ▼
┌─────────────────────────────┐
│ 1. Query 预处理              │
│   - 拼写纠正                 │
│   - 指代消解（结合对话历史）  │
│   - Query 扩展/改写          │
│     · LLM 改写：生成 2-3 个  │
│       同义/细化 query        │
│     · HyDE：生成假设性回答   │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ 2. 多路召回                  │
│   ┌─────────────────────┐   │
│   │ 路径A: 向量检索       │   │  top_k=20
│   │ (Milvus similarity)  │   │
│   ├─────────────────────┤   │
│   │ 路径B: BM25 关键词   │   │  top_k=20
│   │ (全文检索)           │   │
│   ├─────────────────────┤   │
│   │ 路径C: 知识图谱      │   │  (Phase 2)
│   │ (实体关联查询)       │   │
│   └─────────────────────┘   │
│          ↓ 合并去重          │
│     候选集 ~30-40 chunks     │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ 3. Rerank 重排序             │
│   - BGE-reranker-v2-m3      │
│   - 输入: (query, chunk) 对  │
│   - 输出: 重排序 + 相关性分数│
│   - 取 top_k=5-8            │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ 4. 上下文构建                │
│   - 拼接 top chunks          │
│   - 注入文档来源信息         │
│   - 构建系统 prompt          │
│   - Token 预算控制           │
│     (context:max 4096 tokens)│
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ 5. LLM 生成                 │
│   - 流式输出 (SSE)           │
│   - 引用来源标注             │
│   - 置信度提示               │
└─────────────────────────────┘
```

#### 5.3.2 Query 改写策略

| 策略 | 描述 | 适用场景 |
|------|------|---------|
| **LLM 改写** | 让 LLM 生成 2-3 个语义等价但更具体的 query | 查询模糊、太短 |
| **HyDE** | 先让 LLM 生成假设性答案，用答案去检索 | 查询与文档表述差异大 |
| **Query 分解** | 将复杂问题拆解为子问题分别检索 | 多跳问题 |
| **指代消解** | 结合对话历史替换代词 | 多轮对话场景 |

#### 5.3.3 Rerank 设计

```python
class Reranker:
    """基于 BGE-reranker 的重排序器"""

    async def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
        top_k: int = 5,
    ) -> list[RankedDocument]:
        """
        对候选文档重新排序
        - 输入：query + 候选文档列表
        - 处理：批量计算 (query, doc) 相关性分数
        - 输出：按分数降序排列的文档列表，取 top_k
        """
        ...
```

### 5.4 对话管理模块

#### 5.4.1 会话窗口策略

```python
class ConversationMemory:
    """基于滑动窗口的对话记忆管理"""

    def __init__(
        self,
        max_turns: int = 10,         # 最大保留轮数
        max_tokens: int = 4096,       # 最大 token 预算
        strategy: str = "sliding",    # sliding / summary
    ):
        ...

    async def get_context(
        self,
        session_id: str,
        current_query: str,
    ) -> ConversationContext:
        """
        获取当前对话上下文
        - sliding: 保留最近 N 轮
        - summary: 对早期对话做摘要压缩
        """
        ...
```

#### 5.4.2 指代消解

```
历史: "RAG 是一种检索增强生成技术，它结合了检索和生成..."
当前: "它的优势是什么？"

消解后: "RAG（检索增强生成技术）的优势是什么？"
```

实现方式：将最近 3 轮对话 + 当前 query 交给 LLM 做指代消解，替换代词后再检索。

### 5.5 MCP Server 模块

#### 5.5.1 MCP 工具定义

```python
# 工具1：知识库查询
@mcp_tool(name="knowledge_query")
async def knowledge_query(
    query: str,                    # 查询问题
    knowledge_base: str = "default", # 知识库名称
    top_k: int = 5,                # 返回结果数
) -> str:
    """在知识库中查询相关内容并生成回答。
    当用户需要查询文档、知识库中的信息时使用此工具。"""
    ...

# 工具2：文档搜索
@mcp_tool(name="document_search")
async def document_search(
    query: str,                    # 搜索关键词
    knowledge_base: str = "default",
    top_k: int = 10,               # 返回文档片段数
) -> list[dict]:
    """搜索知识库中的相关文档片段，返回原始内容（不经过LLM生成）。
    当需要获取原始文档内容、查看具体段落时使用此工具。"""
    ...

# 工具3：知识库列表
@mcp_tool(name="knowledge_list")
async def knowledge_list() -> list[dict]:
    """列出所有可用的知识库及其基本信息。"""
    ...

# 工具4：会话管理
@mcp_tool(name="session_create")
async def session_create(
    knowledge_base: str = "default",
) -> str:
    """创建新的对话会话，返回 session_id。
    用于需要多轮对话上下文的场景。"""
    ...
```

#### 5.5.2 MCP 资源定义

```python
# 资源：知识库概览
@mcp_resource(uri="knowledge://{knowledge_base}/overview")
async def knowledge_overview(knowledge_base: str) -> str:
    """返回知识库的概览信息：文档数、chunk数、最近更新时间等"""
    ...
```

#### 5.5.3 MCP Server 配置（供 Cursor/Claude 使用）

```json
{
  "mcpServers": {
    "knowledge-forge": {
      "command": "python",
      "args": ["-m", "knowledge_forge.mcp_server"],
      "cwd": "/path/to/knowledge-forge"
    }
  }
}
```

### 5.6 知识库管理后台

#### 5.6.1 核心功能

| 功能 | 描述 |
|------|------|
| 文档管理 | 上传/删除/重新索引文档，查看文档状态 |
| 知识库配置 | chunk_size / overlap / embedding 模型配置 |
| 问答测试 | 在线测试问答效果，查看检索过程 |
| 效果评估 | 召回率、准确率、响应延迟等指标看板 |
| 对话日志 | 查看历史对话、检索命中的 chunks |

#### 5.6.2 评估指标体系

```
检索质量:
├── 召回率 (Recall@K)        - K=5/10/20 下的召回率
├── 准确率 (Precision@K)     - K=5/10 下的准确率
├── MRR                      - 平均倒数排名
└── Hit Rate                 - 命中率

生成质量:
├── 答案相关性 (Relevance)    - 答案与问题的相关程度
├── 忠实度 (Faithfulness)     - 答案是否基于检索内容
└── 完整性 (Completeness)    - 答案是否完整覆盖问题

系统性能:
├── 端到端延迟               - 从提问到首 token 时间
├── 检索延迟                 - 向量检索 + BM25 检索时间
└── 生成延迟                 - LLM 生成时间
```

---

## 六、API 设计

### 6.1 核心 API 列表

#### 文档管理

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/v1/documents/upload` | 上传文档（支持多文件） |
| GET | `/api/v1/documents` | 获取文档列表 |
| GET | `/api/v1/documents/{doc_id}` | 获取文档详情 |
| DELETE | `/api/v1/documents/{doc_id}` | 删除文档 |
| POST | `/api/v1/documents/{doc_id}/reindex` | 重新索引文档 |

#### 知识库管理

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/v1/knowledge-bases` | 创建知识库 |
| GET | `/api/v1/knowledge-bases` | 获取知识库列表 |
| GET | `/api/v1/knowledge-bases/{kb_id}` | 获取知识库详情 |
| PUT | `/api/v1/knowledge-bases/{kb_id}` | 更新知识库配置 |
| DELETE | `/api/v1/knowledge-bases/{kb_id}` | 删除知识库 |

#### 问答

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/v1/chat/sessions` | 创建会话 |
| POST | `/api/v1/chat/sessions/{session_id}/messages` | 发送消息（SSE 流式） |
| GET | `/api/v1/chat/sessions/{session_id}/history` | 获取对话历史 |

#### 评估

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/v1/evaluation/datasets` | 创建评估数据集 |
| POST | `/api/v1/evaluation/run` | 运行评估 |
| GET | `/api/v1/evaluation/reports/{report_id}` | 获取评估报告 |

---

## 七、开发阶段规划

### Phase 0：环境搭建（2 天）

- [ ] 项目初始化：`uv init`、虚拟环境创建
- [ ] `pyproject.toml` 依赖声明
- [ ] Docker Compose 编排（Milvus / PostgreSQL / Redis）
- [ ] FastAPI 骨架 + 健康检查接口
- [ ] 配置管理（Pydantic Settings + .env）
- [ ] 日志配置
- [ ] Git 仓库初始化 + .gitignore + pre-commit

**交付物**：可运行的空项目骨架，`docker compose up` 一键启动基础设施

### Phase 1：文档处理管道（3 天）

- [ ] PDF 解析器（pymupdf，含表格提取）
- [ ] Word 解析器（python-docx）
- [ ] Markdown / TXT 解析器
- [ ] 语义切分器（Recursive Semantic Chunking）
- [ ] Embedding 抽象层 + OpenAI 实现
- [ ] Milvus 向量存储 CRUD
- [ ] 文档上传 API + 异步处理流水线
- [ ] 单元测试

**交付物**：文档上传 → 解析 → 切分 → 向量化 → 存入 Milvus 完整流程

### Phase 2：RAG 问答核心（4 天）

- [ ] 向量检索器（Milvus similarity search）
- [ ] BM25 关键词检索器
- [ ] 混合检索器（结果合并去重）
- [ ] Query 改写（LLM 改写 + HyDE）
- [ ] Reranker（BGE-reranker-v2-m3）
- [ ] 上下文构建器（token 预算控制）
- [ ] LLM 生成器（流式输出 SSE）
- [ ] 对话记忆管理（滑动窗口）
- [ ] 指代消解
- [ ] 问答 API（创建会话 + 发送消息）
- [ ] 集成测试

**交付物**：完整 RAG 问答流程，支持流式输出和多轮对话

### Phase 3：MCP Server 开发（2 天）

- [ ] MCP Server 骨架（基于 mcp-python-sdk）
- [ ] knowledge_query 工具实现
- [ ] document_search 工具实现
- [ ] knowledge_list 工具实现
- [ ] session_create 工具实现
- [ ] 知识库资源定义
- [ ] MCP Server 启动脚本
- [ ] MCP 集成测试
- [ ] MCP 使用文档

**交付物**：可被 Cursor/Claude 调用的 MCP Server

### Phase 4：管理后台 + 评估（3 天）

- [ ] Streamlit 管理后台
  - [ ] 文档管理页
  - [ ] 知识库配置页
  - [ ] 问答测试页（含检索过程展示）
  - [ ] 评估看板页
- [ ] 评估模块
  - [ ] 评估数据集管理
  - [ ] 评估指标计算
  - [ ] 评估报告生成
- [ ] 对话日志查看

**交付物**：可视化管理和评估界面

### Phase 5：优化 + 文档 + 部署（3 天）

- [ ] RAG 优化对比实验
  - [ ] 基线 vs 多路召回
  - [ ] 基线 vs Rerank
  - [ ] 基线 vs Query 改写
  - [ ] 基线 vs 完整流水线
- [ ] 性能优化（缓存、批处理、连接池）
- [ ] API 文档完善
- [ ] README + 部署文档
- [ ] Docker 镜像打包
- [ ] 项目演示视频/截图

**交付物**：完整可部署项目 + 优化对比数据 + 完善文档

---

## 八、量化评估方案

### 8.1 评估数据集

构建一个包含 **50-100** 个问答对的评估数据集：

```json
{
  "id": "q_001",
  "question": "RAG 技术的核心流程是什么？",
  "ground_truth": "RAG 的核心流程包括：1. 检索相关文档片段...",
  "relevant_chunks": ["chunk_id_1", "chunk_id_2"],
  "difficulty": "easy|medium|hard",
  "question_type": "factual|reasoning|multi_hop"
}
```

### 8.2 A/B 对比实验设计

| 实验编号 | 配置 | 变量 |
|---------|------|------|
| E0 (基线) | 单路向量检索 + 无改写 + 无Rerank | — |
| E1 | 向量检索 + Query改写 | Query 改写 |
| E2 | 混合检索 + 无Rerank | 多路召回 |
| E3 | 混合检索 + Rerank | Rerank |
| E4 | 混合检索 + Query改写 + Rerank | 完整流水线 |
| E5 | E4 + HyDE | HyDE 策略 |

### 8.3 预期优化效果

| 指标 | E0 (基线) | E4 (完整) | 提升 |
|------|-----------|-----------|------|
| Recall@5 | ~60% | ~85% | +25% |
| MRR | ~0.45 | ~0.72 | +60% |
| 答案忠实度 | ~70% | ~90% | +20% |
| 端到端延迟 | ~2s | ~3.5s | -1.5s（可接受的延迟换质量） |

---

## 九、环境配置

### 9.1 环境变量

```bash
# .env.example

# ===== 应用配置 =====
APP_NAME=KnowledgeForge
APP_ENV=development
DEBUG=true
LOG_LEVEL=INFO

# ===== LLM 配置 =====
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

# ===== 向量数据库 =====
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=knowledge_chunks

# ===== 关系数据库 =====
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=knowledge_forge
POSTGRES_PASSWORD=changeme
POSTGRES_DB=knowledge_forge

# ===== Redis =====
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# ===== 文件存储 =====
UPLOAD_DIR=./data/uploads
MAX_UPLOAD_SIZE_MB=50

# ===== RAG 配置 =====
CHUNK_SIZE=800
CHUNK_OVERLAP=100
RETRIEVAL_TOP_K=20
RERANK_TOP_K=5
CONTEXT_MAX_TOKENS=4096

# ===== Celery =====
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2
```

### 9.2 Docker Compose 服务

```yaml
# docker-compose.yml 核心服务
services:
  milvus:
    image: milvusdb/milvus:v2.4-latest
    ports: ["19530:19530"]
    volumes: [milvus_data:/var/lib/milvus]

  postgres:
    image: postgres:16-alpine
    ports: ["5432:5432"]
    environment:
      POSTGRES_USER: knowledge_forge
      POSTGRES_PASSWORD: changeme
      POSTGRES_DB: knowledge_forge

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
```

---

## 十、关键依赖清单

```toml
# pyproject.toml [project.dependencies]

[project]
name = "knowledge-forge"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    # Web 框架
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.34.0",
    "sse-starlette>=2.0.0",

    # 数据验证 & 配置
    "pydantic>=2.10.0",
    "pydantic-settings>=2.7.0",

    # 文档解析
    "pymupdf>=1.25.0",
    "python-docx>=1.1.0",
    "mistune>=3.0.0",

    # 向量数据库
    "pymilvus>=2.5.0",

    # 关系数据库
    "sqlalchemy>=2.0.0",
    "asyncpg>=0.30.0",
    "alembic>=1.14.0",

    # Redis
    "redis>=5.0.0",

    # Embedding & LLM
    "openai>=1.60.0",
    "tiktoken>=0.9.0",

    # Reranker
    "FlagEmbedding>=1.2.0",

    # BM25
    "rank-bm25>=0.2.2",
    "jieba>=0.42.1",              # 中文分词

    # MCP
    "mcp>=1.0.0",

    # 异步任务
    "celery>=5.4.0",

    # 评估 & 可视化
    "streamlit>=1.40.0",
    "plotly>=5.24.0",

    # 工具
    "httpx>=0.28.0",
    "python-multipart>=0.0.18",
    "tenacity>=9.0.0",            # 重试
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "pytest-cov>=6.0.0",
    "ruff>=0.8.0",
    "pre-commit>=4.0.0",
    "ipykernel>=6.29.0",
]
```

---

## 十一、风险与应对

| 风险 | 影响 | 应对措施 |
|------|------|---------|
| Milvus 本地部署复杂 | 开发效率降低 | 开发期先用 Chroma 替代，生产再切 Milvus |
| BGE-reranker GPU 依赖 | 本地无GPU时推理慢 | 可用 Jina Reranker API 替代，或小模型 bge-reranker-base |
| OpenAI API 不稳定 | 问答功能中断 | 支持切换 DeepSeek / 本地 Ollama 作为备选 |
| Embedding 模型切换成本高 | 需要全量重新向量化 | 抽象层设计，切换模型只需改配置 + 重建索引 |
| 文档解析质量参差 | 后续检索效果差 | 每种格式有独立的预处理/清洗逻辑，可人工校验 |

---

## 十二、里程碑总结

| 阶段 | 时间 | 交付物 | 验收标准 |
|------|------|--------|---------|
| Phase 0 | 第 1-2 天 | 项目骨架 + 基础设施 | `docker compose up` 启动所有服务，API 健康检查通过 |
| Phase 1 | 第 3-5 天 | 文档处理管道 | 上传 PDF → 自动解析切分向量化存入 Milvus |
| Phase 2 | 第 6-9 天 | RAG 问答核心 | 流式问答可用，多轮对话正常，检索过程可视化 |
| Phase 3 | 第 10-11 天 | MCP Server | Cursor 可通过 MCP 查询知识库 |
| Phase 4 | 第 12-14 天 | 管理后台 + 评估 | Streamlit 面板可用，评估数据可量化 |
| Phase 5 | 第 15-17 天 | 优化 + 文档 | 完整对比数据，项目可部署 |

**总工期估算：约 17 天（2.5 周）**

---

> **下一步**：确认规划后，从 Phase 0 开始搭建项目骨架和虚拟环境。
