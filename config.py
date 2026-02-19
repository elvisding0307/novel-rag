"""
小说 RAG 知识库 - 配置文件
"""
import os
from pathlib import Path

# ── 项目路径 ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# ── Google Gemini API ─────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# 嵌入模型
EMBEDDING_MODEL = "models/gemini-embedding-001"

# 对话模型
LLM_MODEL = "gemini-2.5-flash"

# ── 文本分块参数 ──────────────────────────────────────────
CHUNK_SIZE = 500        # 每个文本块的最大字符数
CHUNK_OVERLAP = 50      # 相邻文本块的重叠字符数

# ── 检索参数 ──────────────────────────────────────────────
SEARCH_K = 5            # 检索返回的文档数量
