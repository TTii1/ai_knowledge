"""文档解析器基类与数据模型"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Table:
    """表格数据"""
    headers: list[str]
    rows: list[list[str]]
    caption: str = ""


@dataclass
class DocumentSection:
    """文档段落/章节"""
    title: str
    content: str
    level: int = 0  # 标题级别 0=正文, 1=H1, 2=H2 ...
    page_number: int | None = None
    tables: list[Table] = field(default_factory=list)


@dataclass
class ParsedDocument:
    """解析后的文档"""
    title: str
    content: str
    sections: list[DocumentSection]
    metadata: dict = field(default_factory=dict)
    tables: list[Table] = field(default_factory=list)
    source_file: str = ""
    total_pages: int = 0
    file_size: int = 0
    file_type: str = ""


class BaseParser(ABC):
    """文档解析器基类"""

    # 支持的文件扩展名
    supported_extensions: list[str] = []

    @abstractmethod
    async def parse(self, file_path: Path) -> ParsedDocument:
        """解析文档，返回结构化文档对象

        Args:
            file_path: 文档文件路径

        Returns:
            ParsedDocument: 解析后的文档
        """
        ...

    def can_parse(self, file_path: Path) -> bool:
        """判断是否能解析该文件"""
        return file_path.suffix.lower() in self.supported_extensions
