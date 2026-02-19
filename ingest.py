"""
小说 RAG 知识库 - 文档摄取模块

向后兼容层：保持原有 CLI 入口
"""
import sys
from pathlib import Path

from config import DATA_DIR
from services.ingest_service import ingest as do_ingest
from utils.exceptions import NovelRAGError


def ingest(data_dir: Path = DATA_DIR):
    """执行摄取（兼容原有接口）"""
    try:
        chunk_count = do_ingest(data_dir)
        print(f"🎉 摄取完成！共 {chunk_count} 个文本块")
    except NovelRAGError as e:
        print(f"❌ {e}")
        raise


if __name__ == "__main__":
    try:
        ingest()
    except NovelRAGError:
        sys.exit(1)
