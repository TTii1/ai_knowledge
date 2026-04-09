"""Query 改写模块

支持策略：
1. LLM 改写：生成 2-3 个语义等价但更具体的 query
2. HyDE：生成假设性答案，用答案去检索
3. Query 分解：将复杂问题拆解为子问题
4. 指代消解：结合对话历史替换代词

所有 LLM 调用都有降级处理：API 不可用时返回原始查询
"""

import logging
from enum import Enum
from typing import Optional

import openai

logger = logging.getLogger(__name__)


class RewriteStrategy(str, Enum):
    """改写策略"""
    NONE = "none"            # 不改写
    LLM_REWRITE = "llm_rewrite"
    HYDE = "hyde"
    DECOMPOSE = "decompose"


class QueryRewriter:
    """Query 改写器

    支持可选改写 — 如果未配置 API Key 或 client，所有方法安全降级返回原始查询
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        self.model = model
        self._client: Optional[openai.AsyncOpenAI] = None

        # 只在有效 API Key 时创建客户端
        if api_key and api_key != "sk-xxx":
            try:
                self._client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
                logger.info("QueryRewriter 初始化完成: model=%s", model)
            except Exception as e:
                logger.warning("QueryRewriter 客户端创建失败: %s", str(e))
        else:
            logger.info("QueryRewriter: 未配置有效 API Key，改写功能禁用")

    @property
    def is_available(self) -> bool:
        """LLM 客户端是否可用"""
        return self._client is not None

    async def rewrite(
        self,
        query: str,
        strategy: RewriteStrategy = RewriteStrategy.LLM_REWRITE,
        conversation_history: Optional[list[dict]] = None,
    ) -> list[str]:
        """改写查询

        Args:
            query: 原始查询
            strategy: 改写策略
            conversation_history: 对话历史（用于指代消解）

        Returns:
            改写后的查询列表（第一个为指代消解后的查询，其余为改写变体）
        """
        # 无 LLM 时直接返回原始查询
        if not self.is_available or strategy == RewriteStrategy.NONE:
            return [query]

        try:
            # 先做指代消解
            resolved_query = await self._resolve_references(query, conversation_history)

            # 再做策略改写
            if strategy == RewriteStrategy.LLM_REWRITE:
                rewritten = await self._llm_rewrite(resolved_query)
            elif strategy == RewriteStrategy.HYDE:
                rewritten = await self._hyde(resolved_query)
            elif strategy == RewriteStrategy.DECOMPOSE:
                rewritten = await self._decompose(resolved_query)
            else:
                rewritten = []

            # 始终包含原始查询（指代消解后的版本）
            result = [resolved_query] + [q for q in rewritten if q != resolved_query]
            logger.info("Query 改写: '%s' → %s (strategy=%s)", query, result, strategy)
            return result

        except Exception as e:
            logger.warning("Query 改写失败，返回原始查询: %s", str(e))
            return [query]

    async def _resolve_references(
        self,
        query: str,
        history: Optional[list[dict]] = None,
    ) -> str:
        """指代消解 - 替换查询中的代词"""
        if not history or len(history) == 0:
            return query

        # 简单启发式：检查是否包含代词
        pronouns = ["它", "他", "她", "这个", "那个", "这些", "那些", "其", "这", "那"]
        if not any(p in query for p in pronouns):
            return query

        if not self.is_available:
            return query

        # 构建上下文
        context = "\n".join(
            f"{'用户' if m.get('role') == 'user' else '助手'}: {m.get('content', '')}"
            for m in history[-6:]  # 最近3轮
        )

        prompt = f"""请将下面查询中的代词替换为具体指代对象。只输出替换后的查询，不要解释。

对话历史：
{context}

当前查询：{query}

替换后的查询："""

        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200,
            )
            resolved = response.choices[0].message.content.strip()
            return resolved if resolved else query
        except Exception as e:
            logger.warning("指代消解失败: %s", str(e))
            return query

    async def _llm_rewrite(self, query: str) -> list[str]:
        """LLM 改写 - 生成多个语义等价的查询"""
        prompt = f"""请将以下查询改写为 2-3 个语义相同但更具体、更适合检索的版本。
每行一个改写结果，不要编号，不要解释。

原始查询：{query}

改写结果："""

        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300,
            )
            text = response.choices[0].message.content.strip()
            rewrites = [line.strip() for line in text.split("\n") if line.strip()]
            return rewrites
        except Exception as e:
            logger.warning("LLM 改写失败: %s", str(e))
            return []

    async def _hyde(self, query: str) -> list[str]:
        """HyDE - 生成假设性答案用于检索"""
        prompt = f"""请回答以下问题。即使你不确定，也请给出一个尽可能详细的回答。
这个回答将用于检索相关文档，所以请包含尽可能多的关键术语和概念。

问题：{query}

回答："""

        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500,
            )
            hyde_answer = response.choices[0].message.content.strip()
            return [hyde_answer] if hyde_answer else []
        except Exception as e:
            logger.warning("HyDE 生成失败: %s", str(e))
            return []

    async def _decompose(self, query: str) -> list[str]:
        """Query 分解 - 将复杂问题拆解为子问题"""
        prompt = f"""请将以下复杂问题拆解为 2-4 个简单的子问题。
每行一个子问题，不要编号，不要解释。

原始问题：{query}

子问题："""

        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=300,
            )
            text = response.choices[0].message.content.strip()
            sub_queries = [line.strip() for line in text.split("\n") if line.strip()]
            return sub_queries
        except Exception as e:
            logger.warning("Query 分解失败: %s", str(e))
            return []
