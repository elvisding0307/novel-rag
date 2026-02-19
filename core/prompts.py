"""
Prompt 模板管理模块

集中管理所有 LLM 提示词模板
"""
from langchain_core.prompts import ChatPromptTemplate


class Prompts:
    """Prompt 模板集合"""
    
    # ── 小说问答 Prompt ──────────────────────────────────────
    NOVEL_QA = ChatPromptTemplate.from_template(
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
    
    # ── 重排评分 Prompt ──────────────────────────────────────
    RERANK = ChatPromptTemplate.from_template(
        """你是一个文档相关性评估专家。请评估以下每个文档段落与用户问题的相关性。

用户问题：{question}

请为每个文档段落打分（0-10分），分数越高表示与问题越相关。
返回格式必须是 JSON 数组，例如：[{{"index": 1, "score": 8}}, {{"index": 2, "score": 3}}]

文档段落：
{documents}

请返回 JSON 格式的评分结果（只返回 JSON，不要其他内容）："""
    )


def format_docs_for_context(docs: list) -> str:
    """将文档格式化为上下文字符串"""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        formatted.append(f"【段落 {i}】(来源: {source})\n{doc.page_content}")
    return "\n\n".join(formatted)


def format_docs_for_rerank(docs: list, preview_length: int = 300) -> str:
    """将文档格式化为重排评估字符串"""
    doc_texts = []
    for i, doc in enumerate(docs, 1):
        preview = doc.page_content[:preview_length].replace("\n", " ")
        doc_texts.append(f"[文档 {i}]: {preview}")
    return "\n\n".join(doc_texts)
