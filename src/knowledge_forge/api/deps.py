"""API 依赖注入"""

from knowledge_forge.config import Settings, get_settings


def get_app_settings() -> Settings:
    """获取应用配置"""
    return get_settings()
