"""
小说 RAG 知识库 - 文档摄取模块

职责：加载 .txt 小说文件 → 文本分块 → 嵌入向量化 → 存入 ChromaDB
"""
import sys
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

from config import (
    DATA_DIR,
    VECTORSTORE_DIR,
    GOOGLE_API_KEY,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def load_documents(data_dir: Path = DATA_DIR) -> list:
    """从 data/ 目录加载所有 .txt 文件"""
    loader = DirectoryLoader(
        str(data_dir),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    docs = loader.load()
    print(f"✅ 加载了 {len(docs)} 个文档")
    return docs


def split_documents(docs: list) -> list:
    """将文档分割成较小的文本块"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ 分割为 {len(chunks)} 个文本块")
    return chunks


def create_vectorstore(chunks: list) -> Chroma:
    """将文本块嵌入并存入 ChromaDB"""
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )
    print(f"✅ 向量库已创建，存储于 {VECTORSTORE_DIR}")
    return vectorstore


def ingest(data_dir: Path = DATA_DIR) -> Chroma:
    """执行完整的摄取流程：加载 → 分块 → 向量化"""
    if not GOOGLE_API_KEY:
        print("❌ 请设置环境变量 GOOGLE_API_KEY")
        sys.exit(1)

    txt_files = list(data_dir.glob("**/*.txt"))
    if not txt_files:
        print(f"❌ 在 {data_dir} 中未找到 .txt 文件，请先放入小说文件")
        sys.exit(1)

    print(f"📚 开始摄取 data/ 目录中的小说文件...")
    docs = load_documents(data_dir)
    chunks = split_documents(docs)
    vectorstore = create_vectorstore(chunks)
    print("🎉 摄取完成！")
    return vectorstore


if __name__ == "__main__":
    ingest()
