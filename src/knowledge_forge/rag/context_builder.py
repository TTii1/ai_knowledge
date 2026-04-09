"""上下文构建器 - 控制注入 LLM 的上下文内容与 Token 预算"""

import logging

import tiktoken

from knowledge_forge.rag.retriever.base import RetrievedDocument

logger = logging.getLogger(__name__)


class ContextBuilder:
    """上下文构建器

    职责：
    1. 拼接检索结果，控制总 Token 数在预算内
    2. 注入文档来源信息
    3. 构建结构化的上下文 prompt
    """

    def __init__(
        self,
        max_tokens: int = 4096,
        model: str = "gpt-4o-mini",
        include_sources: bool = True,
    ):
        self.max_tokens = max_tokens
        self.model = model
        self.include_sources = include_sources
        self._encoding = tiktoken.encoding_for_model(model)

    def build(
        self,
        query: str,
        documents: list[RetrievedDocument],
        conversation_history: list[dict] | None = None,
    ) -> str:
        """构建上下文

        Args:
            query: 用户查询
            documents: 检索到的文档列表
            conversation_history: 对话历史

        Returns:
            构建好的上下文字符串
        """
        # 构建 RAG 上下文部分
        context_parts = []
        used_tokens = 0

        # 预留系统 prompt 和对话历史的 token
        reserved_tokens = 1000  # 系统指令 + 墓地
        if conversation_history:
            history_text = "\n".join(m.get("content", "") for m in conversation_history)
            reserved_tokens += len(self._encoding.encode(history_text))

        available_tokens = self.max_tokens - reserved_tokens

        for i, doc in enumerate(documents):
            # 构建单个文档片段
            source_info = ""
            if self.include_sources and doc.heading_chain:
                source_info = f" [来源: {' > '.join(doc.heading_chain)}]"

            chunk_text = f"--- 文档片段 {i+1}{source_info} ---\n{doc.content}\n"
            chunk_tokens = len(self._encoding.encode(chunk_text))

            if used_tokens + chunk_tokens > available_tokens:
                # 截断该片段
                remaining = available_tokens - used_tokens
                if remaining > 50:
                    truncated = self._encoding.decode(
                        self._encoding.encode(chunk_text)[:remaining]
                    )
                    context_parts.append(truncated + "\n[...已截断]")
                break

            context_parts.append(chunk_text)
            used_tokens += chunk_tokens

        context = "\n".join(context_parts)

        logger.info(
            "上下文构建完成: docs=%d, used_tokens=%d/%d",
            len(context_parts), used_tokens, self.max_tokens,
        )
        return context

    def build_system_prompt(self) -> str:
        """构建系统 prompt"""
        return """你是一个专业的知识库问答助手。请基于提供的文档片段来回答用户的问题。

规则：
1. 只基于提供的文档片段回答，不要编造信息
2. 如果文档片段中没有相关信息，明确告知用户
3. 引用具体的文档来源
4. 回答要准确、完整、有条理
5. 使用中文回答"""
