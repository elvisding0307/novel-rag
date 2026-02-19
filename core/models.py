"""
模型管理模块

统一管理 LLM 和 Embedding 模型的创建和缓存
"""
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from utils.logger import get_logger
from utils.exceptions import ConfigurationError, LLMError

logger = get_logger("novel_rag.models")


class ModelManager:
    """模型管理器 - 单例模式管理模型实例"""
    
    _instance: Optional["ModelManager"] = None
    _llm: Optional[ChatGoogleGenerativeAI] = None
    _embeddings: Optional[GoogleGenerativeAIEmbeddings] = None
    
    def __new__(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _validate_config(self) -> None:
        """验证配置是否完整"""
        from config import config
        if not config.is_configured:
            raise ConfigurationError(
                "API 密钥未配置",
                "请设置环境变量 GOOGLE_API_KEY"
            )
    
    @property
    def llm(self) -> ChatGoogleGenerativeAI:
        """获取 LLM 实例（懒加载）"""
        if self._llm is None:
            from config import config
            self._validate_config()
            logger.info(f"初始化 LLM: {config.google.llm_model}")
            try:
                self._llm = ChatGoogleGenerativeAI(
                    model=config.google.llm_model,
                    google_api_key=config.google.api_key,
                    temperature=config.google.llm_temperature,
                )
            except Exception as e:
                raise LLMError("LLM 初始化失败", str(e))
        return self._llm
    
    @property
    def embeddings(self) -> GoogleGenerativeAIEmbeddings:
        """获取 Embedding 模型实例（懒加载）"""
        if self._embeddings is None:
            from config import config
            self._validate_config()
            logger.info(f"初始化 Embedding: {config.google.embedding_model}")
            try:
                self._embeddings = GoogleGenerativeAIEmbeddings(
                    model=config.google.embedding_model,
                    google_api_key=config.google.api_key,
                )
            except Exception as e:
                raise LLMError("Embedding 模型初始化失败", str(e))
        return self._embeddings
    
    def reset(self) -> None:
        """重置所有模型实例"""
        logger.info("重置模型实例")
        self._llm = None
        self._embeddings = None


# 全局模型管理器实例
model_manager = ModelManager()
