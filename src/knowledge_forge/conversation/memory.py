"""对话记忆管理 - 基于滑动窗口"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """对话轮次"""
    role: str  # user / assistant
    content: str
    token_count: int = 0


class ConversationMemory:
    """对话记忆管理器

    策略：
    - sliding: 保留最近 N 轮对话
    - summary: 对早期对话做摘要压缩（TODO）
    """

    def __init__(
        self,
        max_turns: int = 10,
        max_tokens: int = 4096,
        strategy: str = "sliding",
    ):
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.strategy = strategy
        self._memories: dict[str, list[ConversationTurn]] = {}

    def _estimate_tokens(self, text: str) -> int:
        """粗略估算 token 数"""
        chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)

    async def add_message(self, session_id: str, role: str, content: str) -> None:
        """添加一条对话消息"""
        if session_id not in self._memories:
            self._memories[session_id] = []

        turn = ConversationTurn(
            role=role,
            content=content,
            token_count=self._estimate_tokens(content),
        )
        self._memories[session_id].append(turn)

        # 滑动窗口裁剪
        self._trim_memory(session_id)

    async def get_history(
        self,
        session_id: str,
        max_turns: Optional[int] = None,
    ) -> list[dict]:
        """获取对话历史

        Args:
            session_id: 会话ID
            max_turns: 最大轮数（None 使用默认值）

        Returns:
            对话历史列表 [{"role": ..., "content": ...}]
        """
        turns = self._memories.get(session_id, [])
        limit = max_turns or self.max_turns

        # 取最近 N 轮
        recent = turns[-(limit * 2):]  # 每轮 = 1 user + 1 assistant

        return [{"role": t.role, "content": t.content} for t in recent]

    async def clear(self, session_id: str) -> None:
        """清除对话记忆"""
        if session_id in self._memories:
            del self._memories[session_id]

    def _trim_memory(self, session_id: str) -> None:
        """裁剪对话记忆，保持在限制内"""
        turns = self._memories.get(session_id, [])
        if not turns:
            return

        # 按轮数裁剪
        max_messages = self.max_turns * 2
        if len(turns) > max_messages:
            self._memories[session_id] = turns[-max_messages:]
            turns = self._memories[session_id]

        # 按 token 预算裁剪
        total_tokens = sum(t.token_count for t in turns)
        while total_tokens > self.max_tokens and len(turns) > 2:
            removed = turns.pop(0)
            total_tokens -= removed.token_count
