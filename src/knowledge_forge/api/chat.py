"""问答 API - 支持流式输出（SSE）+ 完整 RAG 流水线

接口：
- POST /sessions — 创建会话
- POST /sessions/{id}/messages — 发送消息（支持流式/非流式）
- GET /sessions/{id}/history — 获取对话历史
- DELETE /sessions/{id} — 删除会话
"""

import json
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()


# ============ 请求/响应模型 ============

class CreateSessionRequest(BaseModel):
    """创建会话请求"""
    knowledge_base: str = "default"
    system_prompt: Optional[str] = None


class ChatMessage(BaseModel):
    """聊天消息"""
    query: str = Field(..., min_length=1, max_length=2000)
    stream: bool = True
    rewrite_strategy: str = "llm_rewrite"  # none / llm_rewrite / hyde / decompose
    top_k: int = Field(default=5, ge=1, le=20)


class SessionResponse(BaseModel):
    """会话响应"""
    session_id: str
    knowledge_base: str
    status: str = "created"


# ============ API 接口 ============

@router.post("/sessions", summary="创建会话", response_model=SessionResponse)
async def create_session(request: CreateSessionRequest):
    """创建新的对话会话"""
    from knowledge_forge.api.deps import get_conversation_memory

    try:
        memory = get_conversation_memory()
        session = await memory.create_session(
            knowledge_base=request.knowledge_base,
            system_prompt=request.system_prompt,
        )
        return SessionResponse(
            session_id=session.session_id,
            knowledge_base=session.knowledge_base,
        )
    except Exception as e:
        logger.error("创建会话失败: %s", str(e))
        raise HTTPException(status_code=500, detail=f"创建会话失败: {str(e)}")


@router.post("/sessions/{session_id}/messages", summary="发送消息")
async def send_message(session_id: str, message: ChatMessage):
    """发送消息并获取回答

    当 stream=True 时返回 SSE 事件流，否则返回完整回答。
    """
    from knowledge_forge.api.deps import get_rag_engine, get_conversation_memory
    from knowledge_forge.rag.query_rewriter import RewriteStrategy

    # 验证会话
    memory = get_conversation_memory()
    session = await memory.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"会话不存在: {session_id}")

    knowledge_base = session.knowledge_base

    # 记录用户消息
    await memory.add_user_message(session_id, message.query)

    # 获取对话历史（OpenAI 格式）
    conversation_history = await memory.get_openai_history(session_id)

    # 解析改写策略
    try:
        strategy = RewriteStrategy(message.rewrite_strategy)
    except ValueError:
        strategy = RewriteStrategy.LLM_REWRITE

    # 获取 RAG 引擎
    rag_engine = get_rag_engine()
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG 服务不可用")

    if message.stream:
        # 流式 SSE 输出
        return StreamingResponse(
            _stream_rag_answer(
                rag_engine=rag_engine,
                query=message.query,
                knowledge_base=knowledge_base,
                conversation_history=conversation_history,
                strategy=strategy,
                top_k=message.top_k,
                session_id=session_id,
                memory=memory,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        # 非流式
        result = await rag_engine.answer(
            query=message.query,
            knowledge_base=knowledge_base,
            conversation_history=conversation_history,
            stream=False,
        )

        # 记录助手回答
        answer = result.answer if hasattr(result, 'answer') else str(result)
        await memory.add_assistant_message(
            session_id, answer,
            metadata={"sources": result.sources} if hasattr(result, 'sources') else {},
        )

        return {
            "session_id": session_id,
            "query": message.query,
            "answer": answer,
            "sources": result.sources if hasattr(result, 'sources') else [],
            "latency_ms": result.latency_ms if hasattr(result, 'latency_ms') else 0,
        }


@router.get("/sessions/{session_id}/history", summary="获取对话历史")
async def get_session_history(session_id: str):
    """获取会话的对话历史"""
    from knowledge_forge.api.deps import get_conversation_memory

    memory = get_conversation_memory()
    session = await memory.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"会话不存在: {session_id}")

    messages = await memory.get_history(session_id)
    return {
        "session_id": session_id,
        "knowledge_base": session.knowledge_base,
        "messages": [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp,
            }
            for msg in messages
        ],
    }


@router.delete("/sessions/{session_id}", summary="删除会话")
async def delete_session(session_id: str):
    """删除对话会话"""
    from knowledge_forge.api.deps import get_conversation_memory

    memory = get_conversation_memory()
    success = await memory.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"会话不存在: {session_id}")

    return {"session_id": session_id, "status": "deleted"}


# ============ 辅助函数 ============

async def _stream_rag_answer(
    rag_engine,
    query: str,
    knowledge_base: str,
    conversation_history: list[dict],
    strategy,
    top_k: int,
    session_id: str,
    memory,
):
    """流式 RAG 问答 SSE 生成器

    SSE 事件格式：
    - event: metadata  (检索元信息)
    - event: delta     (回答片段)
    - event: sources   (引用来源)
    - event: done      (完成)
    - event: error     (错误)
    """
    from knowledge_forge.rag.engine import RAGResult

    try:
        # 发送 metadata 事件
        yield f"event: metadata\ndata: {json.dumps({'query': query, 'knowledge_base': knowledge_base}, ensure_ascii=False)}\n\n"

        result_or_stream = await rag_engine.answer(
            query=query,
            knowledge_base=knowledge_base,
            conversation_history=conversation_history,
            stream=True,
        )

        # 流式输出
        if hasattr(result_or_stream, '__aiter__'):
            # 这是一个异步生成器
            full_answer = []
            async for chunk in result_or_stream:
                full_answer.append(chunk)
                data = json.dumps({"content": chunk}, ensure_ascii=False)
                yield f"event: delta\ndata: {data}\n\n"

            answer = "".join(full_answer)
        else:
            # 非流式结果（降级）
            answer = result_or_stream.answer if hasattr(result_or_stream, 'answer') else str(result_or_stream)
            data = json.dumps({"content": answer}, ensure_ascii=False)
            yield f"event: delta\ndata: {data}\n\n"

        # 记录助手回答
        await memory.add_assistant_message(session_id, answer)

        # 发送来源信息
        sources = result_or_stream.sources if hasattr(result_or_stream, 'sources') else []
        if sources:
            yield f"event: sources\ndata: {json.dumps(sources, ensure_ascii=False)}\n\n"

        # 发送完成事件
        latency = result_or_stream.latency_ms if hasattr(result_or_stream, 'latency_ms') else 0
        yield f"event: done\ndata: {json.dumps({'latency_ms': latency}, ensure_ascii=False)}\n\n"

    except Exception as e:
        logger.error("流式 RAG 问答失败: %s", str(e), exc_info=True)
        error_data = json.dumps({"error": str(e)}, ensure_ascii=False)
        yield f"event: error\ndata: {error_data}\n\n"
