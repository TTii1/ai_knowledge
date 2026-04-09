"""对话记忆管理单元测试"""

import pytest

from knowledge_forge.rag.conversation_memory import (
    ConversationMemory, InMemoryMemoryStore, ConversationMessage,
)


@pytest.fixture
def memory():
    """创建内存对话存储"""
    store = InMemoryMemoryStore()
    return ConversationMemory(store=store, max_turns=5)


@pytest.mark.asyncio
async def test_create_session(memory):
    """测试创建会话"""
    session = await memory.create_session(knowledge_base="test")
    assert session.session_id
    assert session.knowledge_base == "test"
    assert len(session.messages) == 0


@pytest.mark.asyncio
async def test_add_messages(memory):
    """测试添加消息"""
    session = await memory.create_session()
    sid = session.session_id

    await memory.add_user_message(sid, "你好")
    await memory.add_assistant_message(sid, "你好！有什么可以帮你的？")

    history = await memory.get_history(sid)
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[0].content == "你好"
    assert history[1].role == "assistant"
    assert history[1].content == "你好！有什么可以帮你的？"


@pytest.mark.asyncio
async def test_openai_history_format(memory):
    """测试 OpenAI 格式的对话历史"""
    session = await memory.create_session()
    sid = session.session_id

    await memory.add_user_message(sid, "什么是 RAG？")
    await memory.add_assistant_message(sid, "RAG 是检索增强生成技术")

    history = await memory.get_openai_history(sid)
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "什么是 RAG？"}
    assert history[1] == {"role": "assistant", "content": "RAG 是检索增强生成技术"}


@pytest.mark.asyncio
async def test_sliding_window(memory):
    """测试滑动窗口（只保留最近 N 轮）"""
    session = await memory.create_session()
    sid = session.session_id

    # 添加 8 轮对话（16 条消息），max_turns=5，应该只保留最近 5 轮（10 条）
    for i in range(8):
        await memory.add_user_message(sid, f"问题{i}")
        await memory.add_assistant_message(sid, f"回答{i}")

    history = await memory.get_history(sid)
    # max_turns=5，每轮2条，应保留最近10条
    assert len(history) == 10
    # 第一条应该是第3轮的问题
    assert history[0].content == "问题3"
    assert history[1].content == "回答3"
    # 最后一条是第7轮的回答
    assert history[-1].content == "回答7"


@pytest.mark.asyncio
async def test_delete_session(memory):
    """测试删除会话"""
    session = await memory.create_session()
    sid = session.session_id

    await memory.add_user_message(sid, "测试")
    success = await memory.delete_session(sid)
    assert success

    # 会话已删除，无法获取历史
    history = await memory.get_history(sid)
    assert len(history) == 0


@pytest.mark.asyncio
async def test_nonexistent_session(memory):
    """测试不存在的会话"""
    session = await memory.get_session("nonexistent-id")
    assert session is None

    history = await memory.get_history("nonexistent-id")
    assert len(history) == 0


@pytest.mark.asyncio
async def test_get_session(memory):
    """测试获取会话信息"""
    session = await memory.create_session(knowledge_base="my-kb")
    sid = session.session_id

    fetched = await memory.get_session(sid)
    assert fetched is not None
    assert fetched.session_id == sid
    assert fetched.knowledge_base == "my-kb"
