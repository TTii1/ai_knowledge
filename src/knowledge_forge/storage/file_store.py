"""本地文件存储"""

import logging
import shutil
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger(__name__)


class FileStore:
    """本地文件存储

    管理上传的文档原文
    """

    def __init__(self, base_dir: str = "./data/uploads"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    async def save(self, filename: str, content: bytes, knowledge_base: str = "default") -> str:
        """保存文件

        Args:
            filename: 原始文件名
            content: 文件内容
            knowledge_base: 知识库名称

        Returns:
            存储路径
        """
        # 按知识库分目录
        kb_dir = self.base_dir / knowledge_base
        kb_dir.mkdir(parents=True, exist_ok=True)

        # 生成唯一文件名
        file_id = str(uuid4())
        ext = Path(filename).suffix
        stored_name = f"{file_id}{ext}"
        stored_path = kb_dir / stored_name

        stored_path.write_bytes(content)
        logger.info("文件保存成功: %s → %s", filename, stored_path)
        return str(stored_path)

    async def delete(self, file_path: str) -> bool:
        """删除文件"""
        path = Path(file_path)
        if path.exists():
            path.unlink()
            logger.info("文件删除成功: %s", file_path)
            return True
        return False

    async def get(self, file_path: str) -> Path | None:
        """获取文件路径"""
        path = Path(file_path)
        return path if path.exists() else None
