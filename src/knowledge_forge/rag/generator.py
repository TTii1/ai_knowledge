"""LLM 回答生成器 - 支持流式输出"""

import logging
from typing import AsyncGenerator, Optional

import openai

from knowledge_forge.rag.retriever.base import RetrievedDocument

logger = logging.getLogger(__name__)


class Generator:
    """LLM 回答生成器"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        self._client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def generate(
        self,
        query: str,
        context: str,
        system_prompt: str,
        conversation_history: list[dict] | None = None,
    ) -> str:
        """生成回答（非流式）

        Args:
            query: 用户查询
            context: 检索到的上下文
            system_prompt: 系统 prompt
            conversation_history: 对话历史

        Returns:
            生成的回答
        """
        messages = self._build_messages(query, context, system_prompt, conversation_history)

        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        answer = response.choices[0].message.content
        logger.info("回答生成完成: query='%s...', answer_len=%d", query[:30], len(answer))
        return answer

    async def generate_stream(
        self,
        query: str,
        context: str,
        system_prompt: str,
        conversation_history: list[dict] | None = None,
    ) -> AsyncGenerator[str, None]:
        """生成回答（流式输出）

        Args:
            query: 用户查询
            context: 检索到的上下文
            system_prompt: 系统 prompt
            conversation_history: 对话历史

        Yields:
            回答的文本片段
        """
        messages = self._build_messages(query, context, system_prompt, conversation_history)

        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _build_messages(
        self,
        query: str,
        context: str,
        system_prompt: str,
        conversation_history: list[dict] | None = None,
    ) -> list[dict]:
        """构建 LLM 消息列表"""
        messages = [{"role": "system", "content": system_prompt}]

        # 添加对话历史
        if conversation_history:
            messages.extend(conversation_history[-10:])  # 最近5轮

        # 添加当前查询（含上下文）
        user_message = f"""参考文档：
{context}

用户问题：{query}

请基于以上参考文档回答用户问题。"""

        messages.append({"role": "user", "content": user_message})
        return messages
