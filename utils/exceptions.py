"""
自定义异常模块

定义项目中使用的各类异常，便于统一错误处理
"""


class NovelRAGError(Exception):
    """项目基础异常类"""
    
    def __init__(self, message: str, details: str = ""):
        self.message = message
        self.details = details
        super().__init__(self.message)
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message}: {self.details}"
        return self.message


class ConfigurationError(NovelRAGError):
    """配置相关错误"""
    pass


class VectorStoreError(NovelRAGError):
    """向量库相关错误"""
    pass


class RerankerError(NovelRAGError):
    """重排器相关错误"""
    pass


class IngestError(NovelRAGError):
    """文档摄取相关错误"""
    pass


class RetrievalError(NovelRAGError):
    """检索相关错误"""
    pass


class LLMError(NovelRAGError):
    """LLM调用相关错误"""
    pass
