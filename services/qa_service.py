"""
问答服务

处理用户问题，返回基于小说内容的回答
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from core.models import model_manager
from core.retriever import create_retriever, RAGRetriever
from core.vectorstore import vectorstore_manager
from core.prompts import Prompts, format_docs_for_context
from utils.logger import get_logger
from utils.exceptions import LLMError, ConfigurationError

logger = get_logger("novel_rag.qa")


@dataclass
class QAResponse:
    """问答响应结构"""
    answer: str
    sources: List[Dict[str, str]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": self.sources,
        }


class QAService:
    """问答服务"""
    
    _instance: Optional["QAService"] = None
    
    def __new__(cls) -> "QAService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._chain = None
        self._retriever: Optional[RAGRetriever] = None
        self._initialized = True
    
    def _ensure_initialized(self) -> None:
        """确保服务已初始化"""
        from config import config
        if not config.is_configured:
            raise ConfigurationError("API 密钥未配置")
        
        if self._chain is None:
            self._build_chain()
    
    def _build_chain(self) -> None:
        """构建 RAG 链"""
        logger.info("构建 RAG 问答链")
        
        self._retriever = create_retriever()
        llm = model_manager.llm
        
        self._chain = (
            {
                "context": RunnableLambda(self._retriever.retrieve) | format_docs_for_context,
                "question": RunnablePassthrough(),
            }
            | Prompts.NOVEL_QA
            | llm
            | StrOutputParser()
        )
        
        logger.info("RAG 链构建完成")
    
    def ask(self, question: str) -> QAResponse:
        """
        处理用户问题
        
        Args:
            question: 用户问题
            
        Returns:
            包含回答和来源的响应对象
        """
        self._ensure_initialized()
        
        logger.info(f"处理问题: {question[:50]}...")
        
        try:
            answer = self._chain.invoke(question)
            source_docs = self._retriever.retrieve(question)
            
            sources = [
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "未知来源"),
                }
                for doc in source_docs
            ]
            
            logger.info(f"回答生成完成，来源数: {len(sources)}")
            return QAResponse(answer=answer, sources=sources)
            
        except Exception as e:
            logger.error(f"问答失败: {e}")
            raise LLMError("回答生成失败", str(e))
    
    def reload(self) -> None:
        """重新加载服务（文档更新后调用）"""
        logger.info("重新加载问答服务")
        self._chain = None
        self._retriever = None
        model_manager.reset()
        vectorstore_manager.reset()
        self._ensure_initialized()


# 全局服务实例
qa_service = QAService()


def ask(question: str) -> Dict[str, Any]:
    """便捷函数：提问"""
    return qa_service.ask(question).to_dict()


def reload_chain() -> None:
    """便捷函数：重新加载"""
    qa_service.reload()
