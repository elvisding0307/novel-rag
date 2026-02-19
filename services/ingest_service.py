"""
文档摄取服务

处理小说文档的加载、分块和向量化
"""
from pathlib import Path
from typing import List

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from core.vectorstore import vectorstore_manager
from utils.logger import get_logger
from utils.exceptions import IngestError, ConfigurationError

logger = get_logger("novel_rag.ingest")


class IngestService:
    """文档摄取服务"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
    
    def ingest(self) -> int:
        """
        执行完整的摄取流程
        
        Returns:
            摄取的文本块数量
        """
        self._validate()
        
        logger.info(f"开始摄取: {self.data_dir}")
        
        docs = self._load_documents()
        chunks = self._split_documents(docs)
        vectorstore_manager.create_from_documents(chunks)
        
        logger.info(f"摄取完成: {len(chunks)} 个文本块")
        return len(chunks)
    
    def _validate(self) -> None:
        """验证摄取条件"""
        from config import config
        if not config.is_configured:
            raise ConfigurationError("API 密钥未配置", "请设置环境变量 GOOGLE_API_KEY")
        
        txt_files = list(self.data_dir.glob("**/*.txt"))
        if not txt_files:
            raise IngestError("未找到文档", f"在 {self.data_dir} 中未找到 .txt 文件")
    
    def _load_documents(self) -> List[Document]:
        """加载文档"""
        logger.info(f"加载文档: {self.data_dir}")
        try:
            loader = DirectoryLoader(
                str(self.data_dir),
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
                show_progress=True,
            )
            docs = loader.load()
            logger.info(f"加载了 {len(docs)} 个文档")
            return docs
        except Exception as e:
            raise IngestError("文档加载失败", str(e))
    
    def _split_documents(self, docs: List[Document]) -> List[Document]:
        """分割文档"""
        from config import config
        logger.info("分割文档")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk.chunk_size,
            chunk_overlap=config.chunk.chunk_overlap,
            separators=list(config.chunk.separators),
        )
        chunks = splitter.split_documents(docs)
        logger.info(f"分割为 {len(chunks)} 个文本块")
        return chunks


def ingest(data_dir: Path) -> int:
    """便捷函数：执行摄取"""
    service = IngestService(data_dir)
    return service.ingest()
