"""
重排器模块（核心功能）

基于 Gemini LLM 对检索结果进行重排，提升相关性
"""
import json
import re
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

from core.prompts import Prompts, format_docs_for_rerank
from utils.logger import get_logger
from utils.exceptions import RerankerError

logger = get_logger("novel_rag.reranker")


class GeminiReranker:
    """
    基于 Gemini 的文档重排器
    
    使用 LLM 评估文档与查询的相关性，对检索结果重新排序
    """
    
    def __init__(
        self,
        llm: ChatGoogleGenerativeAI,
        top_k: int = 5,
        preview_length: int = 300,
    ):
        """
        初始化重排器
        
        Args:
            llm: 用于评分的 LLM 实例
            top_k: 重排后保留的文档数量
            preview_length: 文档预览长度
        """
        self._llm = llm
        self._top_k = top_k
        self._preview_length = preview_length
        logger.info(f"重排器初始化: top_k={top_k}, preview_length={preview_length}")
    
    def rerank(
        self,
        question: str,
        documents: List[Document],
    ) -> List[Document]:
        """
        对文档进行重排
        
        Args:
            question: 用户问题
            documents: 待重排的文档列表
            
        Returns:
            重排后的文档列表（最多 top_k 个）
        """
        if not documents:
            logger.warning("重排输入为空")
            return []
        
        if len(documents) <= self._top_k:
            logger.debug(f"文档数({len(documents)}) <= top_k({self._top_k})，跳过重排")
            return documents
        
        logger.info(f"开始重排: 问题长度={len(question)}, 文档数={len(documents)}")
        
        try:
            scores = self._get_relevance_scores(question, documents)
            reranked = self._sort_by_scores(documents, scores)
            logger.info(f"重排完成: 返回 {len(reranked)} 个文档")
            return reranked
        except Exception as e:
            logger.error(f"重排失败，使用原始顺序: {e}")
            return documents[:self._top_k]
    
    def _get_relevance_scores(
        self,
        question: str,
        documents: List[Document],
    ) -> List[Dict[str, Any]]:
        """调用 LLM 获取相关性评分"""
        documents_str = format_docs_for_rerank(documents, self._preview_length)
        prompt = Prompts.RERANK.format(question=question, documents=documents_str)
        
        logger.debug("调用 LLM 进行相关性评分")
        response = self._llm.invoke(prompt)
        response_text = response.content.strip()
        
        return self._parse_scores(response_text)
    
    def _parse_scores(self, response_text: str) -> List[Dict[str, Any]]:
        """解析 LLM 返回的评分 JSON"""
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if not json_match:
            raise RerankerError("无法解析评分结果", "未找到 JSON 数组")
        
        try:
            scores = json.loads(json_match.group())
            logger.debug(f"解析到 {len(scores)} 个评分")
            return scores
        except json.JSONDecodeError as e:
            raise RerankerError("JSON 解析失败", str(e))
    
    def _sort_by_scores(
        self,
        documents: List[Document],
        scores: List[Dict[str, Any]],
    ) -> List[Document]:
        """根据评分排序文档"""
        scores.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        reranked_docs = []
        for item in scores[:self._top_k]:
            idx = item.get("index", 1) - 1  # 转为 0-based 索引
            if 0 <= idx < len(documents):
                reranked_docs.append(documents[idx])
                logger.debug(f"选中文档 {idx+1}, 得分: {item.get('score', 0)}")
        
        return reranked_docs if reranked_docs else documents[:self._top_k]


def create_reranker(llm: ChatGoogleGenerativeAI) -> GeminiReranker:
    """工厂函数：创建重排器实例"""
    from config import config
    return GeminiReranker(
        llm=llm,
        top_k=config.retrieval.search_k,
        preview_length=config.rerank.doc_preview_length,
    )
