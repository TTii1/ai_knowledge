"""会话管理"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """对话会话"""
    id: str = field(default_factory=lambda: str(uuid4()))
    knowledge_base: str = "default"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    message_count: int = 0
    metadata: dict = field(default_factory=dict)


class SessionManager:
    """会话管理器

    管理对话会话的创建、获取、删除
    """

    def __init__(self):
        self._sessions: dict[str, Session] = {}

    async def create_session(
        self,
        knowledge_base: str = "default",
        system_prompt: Optional[str] = None,
    ) -> Session:
        """创建新会话"""
        session = Session(
            knowledge_base=knowledge_base,
            metadata={"system_prompt": system_prompt} if system_prompt else {},
        )
        self._sessions[session.id] = session
        logger.info("会话创建: id=%s, kb=%s", session.id, knowledge_base)
        return session

    async def get_session(self, session_id: str) -> Optional[Session]:
        """获取会话"""
        return self._sessions.get(session_id)

    async def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info("会话删除: id=%s", session_id)
            return True
        return False

    async def list_sessions(self) -> list[Session]:
        """列出所有会话"""
        return list(self._sessions.values())
