"""API 模块"""

from knowledge_forge.api.documents import router as documents_router
from knowledge_forge.api.knowledge import router as knowledge_router
from knowledge_forge.api.chat import router as chat_router
from knowledge_forge.api.admin import router as admin_router

__all__ = ["documents_router", "knowledge_router", "chat_router", "admin_router"]
