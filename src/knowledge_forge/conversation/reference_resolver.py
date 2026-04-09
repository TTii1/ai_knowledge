"""指代消解模块"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ReferenceResolver:
    """指代消解器

    识别查询中的代词，结合对话历史替换为具体指代对象。
    实际的消解逻辑由 QueryRewriter 中的 LLM 完成，这里提供辅助功能。
    """

    # 中文常见代词
    PRONOUNS = {"它", "他", "她", "这个", "那个", "这些", "那些", "其", "这", "那"}

    def contains_pronouns(self, query: str) -> bool:
        """判断查询是否包含代词"""
        return any(p in query for p in self.PRONOUNS)

    def needs_resolution(self, query: str, has_history: bool) -> bool:
        """判断是否需要做指代消解"""
        return has_history and self.contains_pronouns(query)
