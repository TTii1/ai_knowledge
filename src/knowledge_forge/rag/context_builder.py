"""上下文构建器 - 控制注入 LLM 的上下文内容与 Token 预算

职责：
1. 拼接检索结果，控制总 Token 数在预算内
2. 注入文档来源信息
3. 构建结构化的上下文 prompt
4. tiktoken 不可用时降级为字符数估算
"""

import logging

from knowledge_forge.rag.retriever.base import RetrievedDocument

logger = logging.getLogger(__name__)


class ContextBuilder:
    """上下文构建器

    Token 预算控制策略：
    - 预留系统 prompt 和对话历史的 token
    - 按顺序填充文档片段，超出预算时截断
    - tiktoken 不可用时按 1 token ≈ 1.5 字符估算
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
        self._encoding = None
        self._use_tiktoken = False

        # 尝试加载 tiktoken
        try:
            import tiktoken
            self._encoding = tiktoken.encoding_for_model(model)
            self._use_tiktoken = True
            logger.info("ContextBuilder: tiktoken 可用，精确 Token 计算")
        except Exception:
            logger.info("ContextBuilder: tiktoken 不可用，使用字符数估算（1 token ≈ 1.5 字符）")

    def _count_tokens(self, text: str) -> int:
        """计算文本的 token 数"""
        if self._use_tiktoken and self._encoding:
            return len(self._encoding.encode(text))
        else:
            # 简单估算：中文约 1.5 字符/token，英文约 4 字符/token
            # 取折中值
            return max(1, len(text) // 2)

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """截断文本到指定 token 数"""
        if self._use_tiktoken and self._encoding:
            tokens = self._encoding.encode(text)[:max_tokens]
            return self._encoding.decode(tokens)
        else:
            # 估算截断位置
            max_chars = max_tokens * 2
            return text[:max_chars]

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
        if not documents:
            logger.info("无检索结果，返回空上下文")
            return ""

        # 构建 RAG 上下文部分
        context_parts = []
        used_tokens = 0

        # 预留系统 prompt 和对话历史的 token
        reserved_tokens = 1000  # 系统指令 + 回答预留
        if conversation_history:
            history_text = "\n".join(m.get("content", "") for m in conversation_history)
            reserved_tokens += self._count_tokens(history_text)

        available_tokens = self.max_tokens - reserved_tokens
        if available_tokens < 200:
            available_tokens = 200  # 至少留 200 tokens 给上下文

        for i, doc in enumerate(documents):
            # 构建单个文档片段
            source_info = ""
            if self.include_sources and doc.heading_chain:
                source_info = f" [来源: {' > '.join(doc.heading_chain)}]"
            elif self.include_sources and doc.metadata.get("source_file"):
                source_info = f" [来源: {doc.metadata['source_file']}]"

            chunk_text = f"--- 文档片段 {i+1}{source_info} ---\n{doc.content}\n"
            chunk_tokens = self._count_tokens(chunk_text)

            if used_tokens + chunk_tokens > available_tokens:
                # 截断该片段
                remaining = available_tokens - used_tokens
                if remaining > 50:
                    truncated = self._truncate_to_tokens(chunk_text, remaining)
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
3. 引用具体的文档来源（标注引用的文档片段编号）
4. 回答要准确、完整、有条理
5. 使用中文回答
6. 如果有多个相关信息，请综合整理后给出完整回答"""
