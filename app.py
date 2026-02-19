"""
小说 RAG 知识库 - Gradio Web 界面

功能：
1. 文档管理：上传 .txt 小说文件并摄取入库
2. 问答对话：基于小说内容的智能问答
"""
import shutil
from pathlib import Path

import gradio as gr

from config import DATA_DIR, config
from services.ingest_service import ingest
from services.qa_service import ask, reload_chain
from utils.exceptions import NovelRAGError
from utils.logger import get_logger

logger = get_logger("novel_rag.app")


# ── 文档上传处理 ──────────────────────────────────────────
def handle_upload(files) -> str:
    """处理上传的 .txt 文件"""
    if not config.is_configured:
        return "❌ 请先设置环境变量 GOOGLE_API_KEY"

    if not files:
        return "❌ 请选择要上传的文件"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    uploaded = []

    for file in files:
        file_path = Path(file.name if hasattr(file, "name") else file)
        if file_path.suffix.lower() != ".txt":
            return f"❌ 仅支持 .txt 文件，收到: {file_path.name}"

        dest = DATA_DIR / file_path.name
        shutil.copy2(str(file_path), str(dest))
        uploaded.append(file_path.name)
        logger.info(f"文件已上传: {file_path.name}")

    try:
        ingest(DATA_DIR)
        reload_chain()
        file_list = "\n".join(f"  • {name}" for name in uploaded)
        return f"✅ 成功上传并摄取以下文件：\n{file_list}\n\n现在可以开始提问了！"
    except NovelRAGError as e:
        logger.error(f"摄取失败: {e}")
        return f"❌ 摄取过程出错：{e.message}"
    except Exception as e:
        logger.error(f"未知错误: {e}")
        return f"❌ 摄取过程出错：{str(e)}"


# ── 问答处理 ──────────────────────────────────────────────
def handle_question(question: str, history: list) -> str:
    """处理用户提问"""
    if not config.is_configured:
        return "❌ 请先设置环境变量 GOOGLE_API_KEY"

    if not question.strip():
        return "请输入您的问题"

    try:
        result = ask(question)
        answer = result["answer"]

        if result["sources"]:
            answer += "\n\n---\n📖 **参考段落：**\n"
            for i, src in enumerate(result["sources"], 1):
                source_file = Path(src["source"]).name
                content_preview = src["content"][:150].replace("\n", " ")
                answer += f"\n**[{i}]** `{source_file}`\n> {content_preview}...\n"

        return answer
    except NovelRAGError as e:
        logger.error(f"问答失败: {e}")
        return f"❌ 回答生成出错：{e.message}"
    except Exception as e:
        logger.error(f"未知错误: {e}")
        return f"❌ 回答生成出错：{str(e)}"


# ── 获取已有文档列表 ──────────────────────────────────────
def list_documents() -> str:
    """列出 data/ 目录中已有的文档"""
    if not DATA_DIR.exists():
        return "📂 暂无文档"

    txt_files = list(DATA_DIR.glob("**/*.txt"))
    if not txt_files:
        return "📂 暂无文档"

    file_list = "\n".join(
        f"  • {f.name} ({f.stat().st_size / 1024:.1f} KB)" 
        for f in txt_files
    )
    return f"📚 已有 {len(txt_files)} 个文档：\n{file_list}"


# ── 自定义样式 ────────────────────────────────────────────
CUSTOM_CSS = """
    .main-header {
        text-align: center;
        padding: 20px 0;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2em;
        margin-bottom: 5px;
    }
    .main-header p {
        color: #666;
        font-size: 1.1em;
    }
"""


# ── 构建 Gradio 界面 ─────────────────────────────────────
def create_app() -> gr.Blocks:
    with gr.Blocks(title="📖 小说 RAG 知识库") as app:
        gr.HTML("""
        <div class="main-header">
            <h1>📖 小说 RAG 知识库</h1>
            <p>上传小说，智能问答 —— 基于 LangChain + Gemini</p>
        </div>
        """)

        with gr.Tabs():
            with gr.Tab("💬 智能问答", id="qa"):
                chatbot = gr.Chatbot(
                    label="对话",
                    height=450,
                    placeholder="上传小说后，在下方输入您的问题开始对话...",
                )
                with gr.Row():
                    question_input = gr.Textbox(
                        label="提问",
                        placeholder="例如：这本小说的主角是谁？",
                        scale=4,
                        show_label=False,
                    )
                    submit_btn = gr.Button("发送", variant="primary", scale=1)

                def chat(question, history):
                    if not question.strip():
                        return history, ""
                    answer = handle_question(question, history)
                    history = history or []
                    history.append({"role": "user", "content": question})
                    history.append({"role": "assistant", "content": answer})
                    return history, ""

                submit_btn.click(
                    fn=chat,
                    inputs=[question_input, chatbot],
                    outputs=[chatbot, question_input],
                )
                question_input.submit(
                    fn=chat,
                    inputs=[question_input, chatbot],
                    outputs=[chatbot, question_input],
                )

            with gr.Tab("📁 文档管理", id="docs"):
                with gr.Row():
                    with gr.Column(scale=1):
                        file_upload = gr.File(
                            label="上传小说文件（.txt）",
                            file_types=[".txt"],
                            file_count="multiple",
                            type="filepath",
                        )
                        upload_btn = gr.Button("📥 上传并摄取", variant="primary")
                    with gr.Column(scale=1):
                        upload_result = gr.Textbox(
                            label="处理结果",
                            lines=8,
                            interactive=False,
                        )

                gr.Markdown("### 📋 已有文档")
                doc_list = gr.Textbox(
                    label="文档列表",
                    value=list_documents(),
                    lines=6,
                    interactive=False,
                )
                refresh_btn = gr.Button("🔄 刷新列表")

                upload_btn.click(
                    fn=handle_upload,
                    inputs=[file_upload],
                    outputs=[upload_result],
                ).then(
                    fn=list_documents,
                    outputs=[doc_list],
                )
                refresh_btn.click(fn=list_documents, outputs=[doc_list])

    return app


# ── 启动 ─────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("启动 Web 应用")
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="blue",
        ),
        css=CUSTOM_CSS,
    )
