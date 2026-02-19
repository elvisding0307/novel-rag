"""
检索器模块

集成向量检索和重排功能
"""
from typing import List, Optional

from langchain_core.documents import Document

from core.models import model_manager
from core.vectorstore import vectorstore_manager
from core.reranker import GeminiReranker, create_reranker
from utils.logger import get_logger
from utils.exceptions import RetrievalError

logger = get_logger("novel_rag.retriever")


class RAGRetriever:
    """
    RAG 检索器
    
    封装向量检索 + 重排的完整流程
    """
    
    def __init__(self, reranker: Optional[GeminiReranker] = None):
        """
        初始化检索器
        
        Args:
            reranker: 可选的重排器实例
        """
        self._reranker = reranker
        self._base_retriever = None
        logger.info(f"检索器初始化: 重排={'启用' if reranker else '禁用'}")
    
    @property
    def base_retriever(self):
        """获取基础向量检索器"""
        if self._base_retriever is None:
            from config import config
            search_k = (
                config.rerank.candidates 
                if config.rerank.enabled and self._reranker 
                else config.retrieval.search_k
            )
            self._base_retriever = vectorstore_manager.get_retriever(search_k)
            logger.info(f"基础检索器初始化: k={search_k}")
        return self._base_retriever
    
    def retrieve(self, question: str) -> List[Document]:
        """
        检索相关文档
        
        Args:
            question: 用户问题
            
        Returns:
            检索（并重排）后的文档列表
        """
        from config import config
        logger.info(f"检索问题: {question[:50]}...")
        
        try:
            docs = self.base_retriever.invoke(question)
            logger.info(f"向量检索返回 {len(docs)} 个文档")
            
            if self._reranker and config.rerank.enabled and len(docs) > config.retrieval.search_k:
                docs = self._reranker.rerank(question, docs)
            
            return docs
        except Exception as e:
            raise RetrievalError("文档检索失败", str(e))
    
    def reset(self) -> None:
        """重置检索器状态"""
        logger.info("重置检索器")
        self._base_retriever = None


def create_retriever() -> RAGRetriever:
    """工厂函数：创建检索器实例"""
    from config import config
    reranker = None
    if config.rerank.enabled:
        reranker = create_reranker(model_manager.llm)
    return RAGRetriever(reranker=reranker)
