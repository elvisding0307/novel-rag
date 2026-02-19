"""
向量库管理模块

管理 ChromaDB 向量库的创建、加载和操作
"""
from typing import Optional, List
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from core.models import model_manager
from utils.logger import get_logger
from utils.exceptions import VectorStoreError

logger = get_logger("novel_rag.vectorstore")


class VectorStoreManager:
    """向量库管理器"""
    
    _instance: Optional["VectorStoreManager"] = None
    _vectorstore: Optional[Chroma] = None
    
    def __new__(cls) -> "VectorStoreManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @property
    def vectorstore(self) -> Chroma:
        """获取向量库实例（懒加载）"""
        if self._vectorstore is None:
            self._load_vectorstore()
        return self._vectorstore
    
    def _load_vectorstore(self) -> None:
        """加载持久化的向量库"""
        from config import VECTORSTORE_DIR
        logger.info(f"加载向量库: {VECTORSTORE_DIR}")
        try:
            self._vectorstore = Chroma(
                persist_directory=str(VECTORSTORE_DIR),
                embedding_function=model_manager.embeddings,
            )
            logger.info("向量库加载成功")
        except Exception as e:
            raise VectorStoreError("向量库加载失败", str(e))
    
    def create_from_documents(
        self,
        documents: List[Document],
        persist_dir: Optional[Path] = None,
    ) -> Chroma:
        """从文档创建向量库"""
        from config import VECTORSTORE_DIR
        persist_dir = persist_dir or VECTORSTORE_DIR
        logger.info(f"创建向量库，文档数: {len(documents)}")
        try:
            self._vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=model_manager.embeddings,
                persist_directory=str(persist_dir),
            )
            logger.info(f"向量库创建成功: {persist_dir}")
            return self._vectorstore
        except Exception as e:
            raise VectorStoreError("向量库创建失败", str(e))
    
    def get_retriever(self, search_k: Optional[int] = None):
        """获取检索器"""
        from config import config
        k = search_k or config.retrieval.search_k
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )
    
    def reset(self) -> None:
        """重置向量库实例"""
        logger.info("重置向量库实例")
        self._vectorstore = None


# 全局向量库管理器实例
vectorstore_manager = VectorStoreManager()
