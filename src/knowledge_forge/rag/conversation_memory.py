"""对话记忆管理模块

支持：
1. Redis 存储（生产环境）：会话持久化，支持分布式
2. 内存存储（开发/测试）：简单的 dict 存储

策略：
- sliding: 滑动窗口，保留最近 N 轮
- summary: 摘要压缩（预留，Phase 3 实现）
"""

import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ConversationMessage:
    """对话消息"""
    role: str          # user / assistant
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)  # 来源、检索结果等


@dataclass
class ConversationSession:
    """对话会话"""
    session_id: str
    knowledge_base: str = "default"
    messages: list[ConversationMessage] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


class BaseMemoryStore(ABC):
    """对话记忆存储基类"""

    @abstractmethod
    async def create_session(self, knowledge_base: str, **kwargs) -> ConversationSession:
        ...

    @abstractmethod
    async def get_session(self, session_id: str) -> Optional[ConversationSession]:
        ...

    @abstractmethod
    async def add_message(self, session_id: str, message: ConversationMessage) -> None:
        ...

    @abstractmethod
    async def get_history(
        self, session_id: str, max_turns: int = 10
    ) -> list[ConversationMessage]:
        ...

    @abstractmethod
    async def delete_session(self, session_id: str) -> bool:
        ...


class InMemoryMemoryStore(BaseMemoryStore):
    """内存对话存储（开发/测试用）"""

    def __init__(self):
        self._sessions: dict[str, ConversationSession] = {}

    async def create_session(self, knowledge_base: str = "default", **kwargs) -> ConversationSession:
        session_id = str(uuid.uuid4())
        session = ConversationSession(
            session_id=session_id,
            knowledge_base=knowledge_base,
            metadata=kwargs,
        )
        self._sessions[session_id] = session
        logger.info("创建会话(内存): session_id=%s, kb=%s", session_id, knowledge_base)
        return session

    async def get_session(self, session_id: str) -> Optional[ConversationSession]:
        return self._sessions.get(session_id)

    async def add_message(self, session_id: str, message: ConversationMessage) -> None:
        session = self._sessions.get(session_id)
        if session:
            session.messages.append(message)
            session.updated_at = time.time()
        else:
            logger.warning("会话不存在: %s", session_id)

    async def get_history(
        self, session_id: str, max_turns: int = 10
    ) -> list[ConversationMessage]:
        session = self._sessions.get(session_id)
        if not session:
            return []
        # 滑动窗口：取最近 max_turns 轮（每轮 = user + assistant = 2 条消息）
        max_messages = max_turns * 2
        return session.messages[-max_messages:]

    async def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False


class RedisMemoryStore(BaseMemoryStore):
    """Redis 对话存储（生产环境）

    数据结构：
    - session:{id} -> Hash (session 元信息)
    - session:{id}:messages -> List (消息列表)
    """

    def __init__(self, redis_client=None, ttl: int = 86400 * 7):
        """
        Args:
            redis_client: Redis 异步客户端实例
            ttl: 会话过期时间（秒），默认 7 天
        """
        self._redis = redis_client
        self.ttl = ttl

    async def create_session(self, knowledge_base: str = "default", **kwargs) -> ConversationSession:
        session_id = str(uuid.uuid4())
        session = ConversationSession(
            session_id=session_id,
            knowledge_base=knowledge_base,
            metadata=kwargs,
        )

        if self._redis:
            import json
            # 存储会话元信息
            session_data = {
                "session_id": session_id,
                "knowledge_base": knowledge_base,
                "created_at": str(session.created_at),
                "updated_at": str(session.updated_at),
                "metadata": json.dumps(kwargs),
            }
            await self._redis.hset(f"session:{session_id}", mapping=session_data)
            await self._redis.expire(f"session:{session_id}", self.ttl)
            logger.info("创建会话(Redis): session_id=%s, kb=%s", session_id, knowledge_base)
        else:
            logger.warning("Redis 不可用，会话仅存于内存")

        return session

    async def get_session(self, session_id: str) -> Optional[ConversationSession]:
        if not self._redis:
            return None

        import json
        data = await self._redis.hgetall(f"session:{session_id}")
        if not data:
            return None

        return ConversationSession(
            session_id=data.get("session_id", session_id),
            knowledge_base=data.get("knowledge_base", "default"),
            created_at=float(data.get("created_at", 0)),
            updated_at=float(data.get("updated_at", 0)),
            metadata=json.loads(data.get("metadata", "{}")),
        )

    async def add_message(self, session_id: str, message: ConversationMessage) -> None:
        if not self._redis:
            logger.warning("Redis 不可用，无法保存消息")
            return

        import json
        msg_data = json.dumps({
            "role": message.role,
            "content": message.content,
            "timestamp": message.timestamp,
            "metadata": message.metadata,
        }, ensure_ascii=False)
        await self._redis.rpush(f"session:{session_id}:messages", msg_data)
        await self._redis.expire(f"session:{session_id}:messages", self.ttl)
        await self._redis.hset(f"session:{session_id}", "updated_at", str(time.time()))

    async def get_history(
        self, session_id: str, max_turns: int = 10
    ) -> list[ConversationMessage]:
        if not self._redis:
            return []

        import json
        max_messages = max_turns * 2

        # 获取最近的消息
        messages_raw = await self._redis.lrange(
            f"session:{session_id}:messages", -max_messages, -1
        )

        messages = []
        for msg_bytes in messages_raw:
            msg_str = msg_bytes if isinstance(msg_bytes, str) else msg_bytes.decode("utf-8")
            try:
                data = json.loads(msg_str)
                messages.append(ConversationMessage(
                    role=data["role"],
                    content=data["content"],
                    timestamp=data.get("timestamp", 0),
                    metadata=data.get("metadata", {}),
                ))
            except (json.JSONDecodeError, KeyError):
                continue

        return messages

    async def delete_session(self, session_id: str) -> bool:
        if not self._redis:
            return False

        await self._redis.delete(f"session:{session_id}", f"session:{session_id}:messages")
        return True


class ConversationMemory:
    """对话记忆管理器

    策略：
    - sliding: 滑动窗口，保留最近 N 轮
    - 支持将消息列表转换为 OpenAI 格式的 messages
    """

    def __init__(
        self,
        store: BaseMemoryStore,
        max_turns: int = 10,
        max_tokens: int = 4096,
        strategy: str = "sliding",
    ):
        self.store = store
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.strategy = strategy

    async def create_session(self, knowledge_base: str = "default", **kwargs) -> ConversationSession:
        """创建新的对话会话"""
        return await self.store.create_session(knowledge_base=knowledge_base, **kwargs)

    async def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """获取会话信息"""
        return await self.store.get_session(session_id)

    async def add_user_message(self, session_id: str, content: str, metadata: dict | None = None) -> None:
        """添加用户消息"""
        msg = ConversationMessage(
            role="user",
            content=content,
            metadata=metadata or {},
        )
        await self.store.add_message(session_id, msg)

    async def add_assistant_message(self, session_id: str, content: str, metadata: dict | None = None) -> None:
        """添加助手消息"""
        msg = ConversationMessage(
            role="assistant",
            content=content,
            metadata=metadata or {},
        )
        await self.store.add_message(session_id, msg)

    async def get_history(self, session_id: str) -> list[ConversationMessage]:
        """获取对话历史"""
        return await self.store.get_history(session_id, max_turns=self.max_turns)

    async def get_openai_history(self, session_id: str) -> list[dict]:
        """获取 OpenAI 格式的对话历史

        Returns:
            [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
        """
        messages = await self.get_history(session_id)
        return [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

    async def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        return await self.store.delete_session(session_id)
