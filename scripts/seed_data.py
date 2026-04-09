"""种子数据导入脚本"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from knowledge_forge.document.pipeline import DocumentPipeline
from knowledge_forge.config import get_settings


async def seed():
    """导入种子数据"""
    settings = get_settings()
    print(f"🚀 初始化知识库种子数据...")

    # TODO: 导入示例文档
    # pipeline = DocumentPipeline()
    # chunks = await pipeline.process(Path("data/test_docs/sample.pdf"))
    print("✅ 种子数据导入完成（待实现）")


if __name__ == "__main__":
    asyncio.run(seed())
