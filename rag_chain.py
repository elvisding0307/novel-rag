"""
小说 RAG 知识库 - 检索问答链

向后兼容层：保持原有 API 不变
"""
from services.qa_service import ask, reload_chain

__all__ = ["ask", "reload_chain"]
