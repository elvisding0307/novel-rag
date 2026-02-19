"""
小说 RAG 知识库 - 配置文件

集中管理所有配置项，支持环境变量覆盖
"""
import os
from pathlib import Path
from dataclasses import dataclass, field

from utils.logger import setup_logger

# ── 项目路径 ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
LOG_DIR = BASE_DIR / "logs"


@dataclass
class GoogleConfig:
    """Google API 配置"""
    api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    embedding_model: str = "models/gemini-embedding-001"
    llm_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.3


@dataclass
class ChunkConfig:
    """文本分块配置"""
    chunk_size: int = 500
    chunk_overlap: int = 50
    separators: tuple = ("\n\n", "\n", "。", "！", "？", "；", "，", " ", "")


@dataclass
class RetrievalConfig:
    """检索配置"""
    search_k: int = 5  # 最终返回给 LLM 的文档数量


@dataclass  
class RerankConfig:
    """重排配置（核心功能）"""
    enabled: bool = True
    candidates: int = 15  # 初始检索的候选文档数量
    doc_preview_length: int = 300  # 重排时文档预览长度
    

@dataclass
class AppConfig:
    """应用配置"""
    google: GoogleConfig = field(default_factory=GoogleConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    
    @property
    def is_configured(self) -> bool:
        """检查必要配置是否完整"""
        return bool(self.google.api_key)


# ── 全局配置实例 ──────────────────────────────────────────
config = AppConfig()

# ── 初始化日志 ──────────────────────────────────────────────
logger = setup_logger("novel_rag", log_file=LOG_DIR / "app.log")

# ── 向后兼容：导出原有常量 ─────────────────────────────────
GOOGLE_API_KEY = config.google.api_key
EMBEDDING_MODEL = config.google.embedding_model
LLM_MODEL = config.google.llm_model
CHUNK_SIZE = config.chunk.chunk_size
CHUNK_OVERLAP = config.chunk.chunk_overlap
SEARCH_K = config.retrieval.search_k
RERANK_ENABLED = config.rerank.enabled
RERANK_CANDIDATES = config.rerank.candidates
