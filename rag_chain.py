"""
小说 RAG 知识库 - 检索问答链

职责：从 ChromaDB 检索相关段落 → 构造提示词 → Gemini 生成回答
"""
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config import (
    VECTORSTORE_DIR,
    GOOGLE_API_KEY,
    EMBEDDING_MODEL,
    LLM_MODEL,
    SEARCH_K,
)

# ── 小说问答专用 Prompt ──────────────────────────────────
NOVEL_QA_PROMPT = ChatPromptTemplate.from_template(
    """你是一个小说阅读助手。请根据以下从小说中检索到的原文段落来回答用户的问题。

要求：
1. 仅基于提供的原文段落进行回答，不要编造内容
2. 如果原文段落中没有相关信息，请明确告知用户
3. 回答时可以适当引用原文
4. 使用中文回答

--- 检索到的原文段落 ---
{context}
--- 原文段落结束 ---

用户问题：{question}

回答："""
)


def _format_docs(docs) -> str:
    """将检索到的文档格式化为字符串"""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        formatted.append(f"【段落 {i}】(来源: {source})\n{doc.page_content}")
    return "\n\n".join(formatted)


def build_rag_chain():
    """构建 RAG 检索问答链"""
    # 加载嵌入模型
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )

    # 加载持久化的向量库
    vectorstore = Chroma(
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=embeddings,
    )

    # 构造检索器
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": SEARCH_K},
    )

    # 构造 LLM
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3,
    )

    # 使用 LCEL 组装 RAG 链
    rag_chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | NOVEL_QA_PROMPT
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


# ── 全局链实例（延迟初始化）────────────────────────────────
_chain = None
_retriever = None


def _ensure_chain():
    """确保链已初始化"""
    global _chain, _retriever
    if _chain is None:
        _chain, _retriever = build_rag_chain()


def ask(question: str) -> dict:
    """
    提出问题并获取回答

    返回:
        {
            "answer": str,       # LLM 生成的回答
            "sources": list,     # 检索到的原文段落
        }
    """
    _ensure_chain()

    # 获取回答
    answer = _chain.invoke(question)

    # 获取检索到的源文档
    source_docs = _retriever.invoke(question)
    sources = []
    for doc in source_docs:
        sources.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "未知来源"),
        })

    return {
        "answer": answer,
        "sources": sources,
    }


def reload_chain():
    """重新加载 RAG 链（在新文档入库后调用）"""
    global _chain, _retriever
    _chain = None
    _retriever = None
    _ensure_chain()
