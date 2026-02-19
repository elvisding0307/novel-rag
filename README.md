# 📖 小说 RAG 知识库

基于 **LangChain + Google Gemini** 的小说 RAG（检索增强生成）知识库。上传小说文本文件，通过自然语言提问获取基于原文的智能回答。

## 功能

- 📥 上传 `.txt` 小说文件，自动分块向量化
- 💬 基于小说内容的智能问答
- 📖 回答附带原文参考段落
- 🌐 Gradio Web 界面

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 设置 API Key

```bash
export GOOGLE_API_KEY="your-gemini-api-key"
```

### 3. 摄取小说文件

将 `.txt` 小说文件放入 `data/` 目录，然后运行：

```bash
python ingest.py
```

### 4. 启动 Web 界面

```bash
python app.py
```

访问 `http://localhost:7860` 开始使用。

## 项目结构

```
├── config.py        # 配置文件
├── ingest.py        # 文档摄取（加载 → 分块 → 向量化）
├── rag_chain.py     # RAG 检索问答链
├── app.py           # Gradio Web 界面
├── data/            # 小说文件目录
└── vectorstore/     # ChromaDB 向量库
```

## 技术栈

- **LangChain** — 编排框架
- **Google Gemini** — LLM + 嵌入模型
- **ChromaDB** — 向量数据库
- **Gradio** — Web 界面
