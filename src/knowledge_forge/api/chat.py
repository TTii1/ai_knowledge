"""问答 API - 支持流式输出（SSE）"""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()


class CreateSessionRequest(BaseModel):
    """创建会话请求"""
    knowledge_base: str = "default"
    system_prompt: Optional[str] = None


class ChatMessage(BaseModel):
    """聊天消息"""
    session_id: str
    query: str = Field(..., min_length=1, max_length=2000)
    stream: bool = True


@router.post("/sessions", summary="创建会话")
async def create_session(request: CreateSessionRequest):
    """创建新的对话会话"""
    session_id = str(uuid.uuid4())
    # TODO: 实现会话创建，初始化对话记忆
    return {
        "session_id": session_id,
        "knowledge_base": request.knowledge_base,
        "status": "created",
    }


@router.post("/sessions/{session_id}/messages", summary="发送消息")
async def send_message(session_id: str, message: ChatMessage):
    """发送消息并获取回答（支持 SSE 流式输出）

    当 stream=True 时返回 SSE 事件流，否则返回完整回答。
    """
    # TODO: 实现完整 RAG 问答流程
    # 1. Query 改写
    # 2. 多路召回
    # 3. Rerank 重排序
    # 4. 上下文注入
    # 5. LLM 生成（流式/非流式）
    return {
        "session_id": session_id,
        "answer": "TODO: RAG 问答待实现",
        "sources": [],
        "status": "not_implemented",
    }


@router.get("/sessions/{session_id}/history", summary="获取对话历史")
async def get_session_history(session_id: str):
    """获取会话的对话历史"""
    # TODO: 实现对话历史查询
    return {"session_id": session_id, "messages": []}
