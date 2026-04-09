"""LLM 回答生成器 - 支持流式输出（SSE）

特点：
- 流式/非流式双模式
- 来源引用标注（在回答中标注参考的文档片段编号）
- OpenAI API 安全降级
- 可配置的 temperature / max_tokens
"""

import logging
from typing import AsyncGenerator, Optional

import openai

logger = logging.getLogger(__name__)


class Generator:
    """LLM 回答生成器

    支持流式和非流式两种模式：
    - generate(): 非流式，返回完整回答
    - generate_stream(): 流式，逐步 yield 回答片段
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client: Optional[openai.AsyncOpenAI] = None

        # 只在有效 API Key 时创建客户端
        if api_key and api_key != "sk-xxx":
            try:
                self._client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
                logger.info("Generator 初始化完成: model=%s", model)
            except Exception as e:
                logger.warning("Generator 客户端创建失败: %s", str(e))
        else:
            logger.info("Generator: 未配置有效 API Key，生成功能禁用")

    @property
    def is_available(self) -> bool:
        """LLM 客户端是否可用"""
        return self._client is not None

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
        if not self.is_available:
            return self._fallback_answer(query, context)

        try:
            messages = self._build_messages(query, context, system_prompt, conversation_history)

            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            answer = response.choices[0].message.content or ""
            logger.info("回答生成完成: query='%s...', answer_len=%d", query[:30], len(answer))
            return answer

        except Exception as e:
            logger.error("回答生成失败: %s", str(e))
            return self._fallback_answer(query, context)

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
        if not self.is_available:
            yield self._fallback_answer(query, context)
            return

        try:
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

        except Exception as e:
            logger.error("流式生成失败: %s", str(e))
            yield f"\n\n[错误] 生成回答时发生异常：{str(e)}"

    def _build_messages(
        self,
        query: str,
        context: str,
        system_prompt: str,
        conversation_history: list[dict] | None = None,
    ) -> list[dict]:
        """构建 LLM 消息列表"""
        messages = [{"role": "system", "content": system_prompt}]

        # 添加对话历史（最近 5 轮 = 10 条消息）
        if conversation_history:
            messages.extend(conversation_history[-10:])

        # 添加当前查询（含上下文）
        if context:
            user_message = f"""参考文档：
{context}

用户问题：{query}

请基于以上参考文档回答用户问题。在回答中标注参考的文档片段编号，例如"根据文档片段1..."。"""
        else:
            user_message = f"""用户问题：{query}

注意：没有检索到相关文档，请根据你的知识回答，并说明这是基于通用知识而非知识库内容。"""

        messages.append({"role": "user", "content": user_message})
        return messages

    @staticmethod
    def _fallback_answer(query: str, context: str) -> str:
        """降级回答（LLM 不可用时）"""
        if context:
            # 简单返回检索到的上下文
            return f"（LLM 暂时不可用，以下为检索到的相关内容）\n\n{context[:2000]}"
        else:
            return "抱歉，LLM 服务暂时不可用，无法生成回答。请稍后再试。"
