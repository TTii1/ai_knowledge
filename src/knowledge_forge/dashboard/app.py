"""KnowledgeForge 管理后台 — Streamlit 应用

启动方式：streamlit run src/knowledge_forge/dashboard/app.py

功能：
1. 系统概览仪表板
2. 文档管理
3. 知识库配置
4. 问答测试（含检索过程展示）
5. 评估看板
6. 对话日志
"""

import streamlit as st
import pandas as pd
import json
import sys
import os
from datetime import datetime

# 确保项目路径可用
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from knowledge_forge.config import get_settings

settings = get_settings()

# ============ 页面配置 ============

st.set_page_config(
    page_title="KnowledgeForge 管理后台",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============ API 客户端 ============

API_BASE = f"http://{settings.api_host}:{settings.api_port}{settings.api_prefix}"


def api_get(path: str):
    """调用 GET API"""
    try:
        import httpx
        resp = httpx.get(f"{API_BASE}{path}", timeout=10.0)
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception as e:
        st.error(f"API 调用失败: {e}")
        return None


def api_post(path: str, json_data: dict = None):
    """调用 POST API"""
    try:
        import httpx
        resp = httpx.post(f"{API_BASE}{path}", json=json_data, timeout=30.0)
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception as e:
        st.error(f"API 调用失败: {e}")
        return None


# ============ 侧边栏导航 ============

page = st.sidebar.selectbox(
    "导航",
    ["📊 系统概览", "📄 文档管理", "📚 知识库", "💬 问答测试", "📈 评估看板", "📝 对话日志"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**环境**: {settings.app_env}")
st.sidebar.markdown(f"**LLM**: {settings.llm_model}")
st.sidebar.markdown(f"**Embedding**: {settings.embedding_model}")


# ============ 系统概览 ============

if page == "📊 系统概览":
    st.title("📊 系统概览")

    stats = api_get("/admin/stats")
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("知识库数", stats.get("total_knowledge_bases", 0))
        col2.metric("文档数", stats.get("total_documents", 0))
        col3.metric("Chunk 数", stats.get("total_chunks", 0))
        col4.metric("会话数", stats.get("total_sessions", 0))

        rag_status = "✅ 可用" if stats.get("rag_engine_available") else "❌ 不可用"
        st.info(f"RAG 引擎状态: {rag_status}")
    else:
        st.warning("无法连接后端 API，请确保服务已启动。")

    # 知识库列表
    st.subheader("知识库概览")
    kb_list = api_get("/knowledge-bases")
    if kb_list and kb_list.get("items"):
        df = pd.DataFrame(kb_list["items"])
        display_cols = ["name", "description", "document_count", "chunk_count", "is_active"]
        available_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[available_cols], use_container_width=True)
    else:
        st.info("暂无知识库数据。")


# ============ 文档管理 ============

elif page == "📄 文档管理":
    st.title("📄 文档管理")

    # 文档列表
    kb_filter = st.selectbox("按知识库筛选", ["all"], key="doc_kb_filter")
    doc_list = api_get("/documents")
    if doc_list and doc_list.get("items"):
        docs = doc_list["items"]
        df = pd.DataFrame(docs)
        display_cols = ["filename", "file_type", "status", "chunk_count", "total_tokens", "created_at"]
        available_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[available_cols], use_container_width=True)

        st.caption(f"共 {doc_list.get('total', 0)} 个文档")
    else:
        st.info("暂无文档。请通过 API 上传文档。")

    # 上传区域
    st.subheader("上传文档")
    uploaded_files = st.file_uploader(
        "选择文件（支持 PDF/Word/Markdown/TXT）",
        accept_multiple_files=True,
        type=["pdf", "docx", "md", "txt"],
    )
    if uploaded_files:
        kb_name = st.text_input("目标知识库", value="default")
        if st.button("上传"):
            for f in uploaded_files:
                st.info(f"上传 {f.name}（需要通过 API 上传，此功能需要后端支持 multipart）")


# ============ 知识库 ============

elif page == "📚 知识库":
    st.title("📚 知识库管理")

    # 知识库列表
    kb_list = api_get("/knowledge-bases")
    if kb_list and kb_list.get("items"):
        for kb in kb_list["items"]:
            with st.expander(f"**{kb['name']}** — {kb.get('description', '无描述')}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"- 文档数: {kb.get('document_count', 0)}")
                    st.write(f"- Chunk 数: {kb.get('chunk_count', 0)}")
                    st.write(f"- Embedding: {kb.get('embedding_model', 'N/A')}")
                with col2:
                    st.write(f"- 维度: {kb.get('embedding_dimension', 0)}")
                    st.write(f"- Chunk Size: {kb.get('chunk_size', 800)}")
                    st.write(f"- Overlap: {kb.get('chunk_overlap', 100)}")
    else:
        st.info("暂无知识库。")

    # 创建知识库
    st.subheader("创建知识库")
    with st.form("create_kb"):
        kb_name = st.text_input("知识库名称")
        kb_desc = st.text_area("描述")
        submitted = st.form_submit_button("创建")
        if submitted and kb_name:
            result = api_post("/knowledge-bases", {"name": kb_name, "description": kb_desc})
            if result:
                st.success(f"知识库 '{kb_name}' 创建成功！")
            else:
                st.error("创建失败")


# ============ 问答测试 ============

elif page == "💬 问答测试":
    st.title("💬 问答测试")

    col1, col2 = st.columns([2, 1])
    with col1:
        query = st.text_area("输入问题", height=100)
    with col2:
        kb_name = st.text_input("知识库", value="default")
        top_k = st.slider("Top K", 1, 20, 5)
        session_id = st.text_input("Session ID（可选）", value="")

    if st.button("🔍 查询", type="primary") and query:
        with st.spinner("正在检索和生成回答..."):
            # 调用 Chat API
            payload = {
                "query": query,
                "knowledge_base": kb_name,
                "top_k": top_k,
            }
            if session_id:
                payload["session_id"] = session_id

            result = api_post("/chat/messages", payload)
            if result:
                st.subheader("回答")
                st.markdown(result.get("answer", ""))

                # 来源引用
                sources = result.get("sources", [])
                if sources:
                    with st.expander(f"📎 参考来源 ({len(sources)})"):
                        for i, src in enumerate(sources, 1):
                            st.markdown(
                                f"**{i}.** {src.get('source_file', '未知')} "
                                f"({' > '.join(src.get('heading_chain', []))}) "
                                f"[score: {src.get('score', 0):.4f}]"
                            )
                            st.caption(src.get("content_preview", ""))

                # 性能信息
                latency = result.get("latency_ms", 0)
                st.caption(f"⏱ 耗时: {latency:.0f}ms")
            else:
                st.error("查询失败，请检查后端服务。")


# ============ 评估看板 ============

elif page == "📈 评估看板":
    st.title("📈 RAG 评估看板")

    # A/B 对比实验
    st.subheader("A/B 对比实验")
    exp_ids = st.multiselect(
        "选择实验配置",
        ["E0", "E1", "E2", "E3", "E4", "E5"],
        default=["E0", "E4"],
        help="E0: 基线 | E1: +Query改写 | E2: +多路召回 | E3: +Rerank | E4: 完整流水线 | E5: +HyDE",
    )

    if st.button("🚀 运行评估", type="primary"):
        with st.spinner("正在运行评估，请稍候..."):
            result = api_post("/admin/evaluation/ab", {"experiment_ids": exp_ids})
            if result and "comparison" in result:
                comparison = result["comparison"]

                # 对比表格
                rows = []
                for exp_id, data in comparison.items():
                    rm = data.get("retrieval_metrics", {})
                    gm = data.get("generation_metrics", {})
                    pm = data.get("performance_metrics", {})
                    rows.append({
                        "实验": exp_id,
                        "Recall@5": f"{rm.get('recall_at_5', 0):.2%}",
                        "Recall@10": f"{rm.get('recall_at_10', 0):.2%}",
                        "MRR": f"{rm.get('mrr', 0):.4f}",
                        "Hit Rate": f"{rm.get('hit_rate', 0):.2%}",
                        "相关性": f"{gm.get('relevance', 0):.2%}",
                        "忠实度": f"{gm.get('faithfulness', 0):.2%}",
                        "延迟(ms)": f"{pm.get('e2e_latency_ms', 0):.0f}",
                    })

                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

                # 详细报告
                for exp_id, data in comparison.items():
                    with st.expander(f"实验 {exp_id} 详细报告"):
                        st.markdown(data.get("summary", ""))
            else:
                st.error("评估运行失败。")

    # 历史报告
    st.subheader("历史评估报告")
    reports = api_get("/admin/evaluation/reports")
    if reports and reports.get("items"):
        for r in reports["items"][:10]:
            st.markdown(f"- **{r.get('name', '')}** ({r.get('created_at', '')})")
    else:
        st.info("暂无历史报告。")


# ============ 对话日志 ============

elif page == "📝 对话日志":
    st.title("📝 对话日志")

    logs = api_get("/admin/conversation-logs")
    if logs and logs.get("items"):
        for log in logs["items"]:
            with st.expander(f"会话 {log.get('session_id', '')[:8]}... — {log.get('knowledge_base', '')}"):
                st.write(f"- 消息数: {log.get('message_count', 0)}")
                st.write(f"- 创建时间: {datetime.fromtimestamp(log.get('created_at', 0)).strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"- 更新时间: {datetime.fromtimestamp(log.get('updated_at', 0)).strftime('%Y-%m-%d %H:%M:%S')}")

        st.caption(f"共 {logs.get('total', 0)} 个会话")
    else:
        st.info("暂无对话日志。")


# ============ 底部 ============

st.sidebar.markdown("---")
st.sidebar.caption("KnowledgeForge v0.1.0")
st.sidebar.caption("Powered by RAG + MCP")
