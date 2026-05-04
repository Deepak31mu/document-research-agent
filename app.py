import gradio as gr
import hashlib
from typing import List, Dict, Union
import os

from document_processor.file_handler import DocumentProcessor
from retriever.builder import RetrieverBuilder
from agents.workflow import AgentWorkflow
from config import constants, settings
from utils.logger import logger

EXAMPLES = {
    "Google 2024 Environmental Report": {
        "question": "Retrieve the data center PUE efficiency values in Singapore 2nd facility in 2019 and 2022. Also retrieve regional average CFE in Asia pacific in 2023",
        "file_paths": ["examples/google-2024-environmental-report.pdf"]
    },
    "DeepSeek-R1 Technical Report": {
        "question": "Summarize DeepSeek-R1 model's performance evaluation on all coding tasks against OpenAI o1-mini model",
        "file_paths": ["examples/DeepSeek Technical Report.pdf"]
    }
}


def main():
    processor = DocumentProcessor()
    retriever_builder = RetrieverBuilder()
    workflow = AgentWorkflow()

    css = """
    .title {
        font-size: 1.5em !important;
        text-align: center !important;
        color: #FFD700;
    }
    .subtitle {
        font-size: 1em !important;
        text-align: center !important;
        color: #FFD700;
    }
    .text {
        text-align: center;
    }
    """

    js = """
    function createGradioAnimation() {
        var container = document.createElement('div');
        container.id = 'gradio-animation';
        container.style.fontSize = '2em';
        container.style.fontWeight = 'bold';
        container.style.textAlign = 'center';
        container.style.marginBottom = '20px';
        container.style.color = '#eba93f';

        var text = 'Welcome to My Document Chat 🐥!';
        for (var i = 0; i < text.length; i++) {
            (function(i){
                setTimeout(function(){
                    var letter = document.createElement('span');
                    letter.style.opacity = '0';
                    letter.style.transition = 'opacity 0.1s';
                    letter.innerText = text[i];
                    container.appendChild(letter);
                    setTimeout(function() { letter.style.opacity = '0.9'; }, 50);
                }, i * 250);
            })(i);
        }

        var gradioContainer = document.querySelector('.gradio-container');
        gradioContainer.insertBefore(container, gradioContainer.firstChild);
        return 'Animation created';
    }
    """

    with gr.Blocks(theme=gr.themes.Citrus(), title="My Document Chat 🐥", css=css, js=js) as demo:
        gr.Markdown("## My Document Chat: powered by Docling 🐥 and LangGraph", elem_classes="subtitle")
        gr.Markdown("# How it works ✨:", elem_classes="title")
        gr.Markdown("📤 Upload your document(s), enter your query then hit Submit 📝", elem_classes="text")
        gr.Markdown("Or you can select one of the examples from the drop-down menu, select Load Example then hit Submit 📝", elem_classes="text")
        gr.Markdown("⚠️ **Note:** My Document Chat only accepts documents in these formats: '.pdf', '.docx', '.txt', '.md'", elem_classes="text")

        session_state = gr.State({
            "file_hashes": frozenset(),
            "retriever": None
        })

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Example 📂")
                example_dropdown = gr.Dropdown(
                    label="Select an Example 🐥",
                    choices=list(EXAMPLES.keys()),
                    value=None,
                )
                load_example_btn = gr.Button("Load Example 🛠️")

                files = gr.Files(label="📄 Upload Documents", file_types=constants.ALLOWED_TYPES)
                question = gr.Textbox(label="❓ Question", lines=3)
                submit_btn = gr.Button("Submit 🚀")

            with gr.Column():
                answer_output = gr.Textbox(label="🐥 Answer", interactive=False)
                verification_output = gr.Textbox(label="✅ Verification Report")

        def load_example(example_key: str):
            if not example_key or example_key not in EXAMPLES:
                return [], ""
            ex_data = EXAMPLES[example_key]
            loaded_files = [p for p in ex_data["file_paths"] if os.path.exists(p)]
            missing = [p for p in ex_data["file_paths"] if not os.path.exists(p)]
            for p in missing:
                logger.warning(f"Example file not found: {p}")
            return loaded_files, ex_data["question"]

        load_example_btn.click(
            fn=load_example,
            inputs=[example_dropdown],
            outputs=[files, question]
        )

        def process_question(question_text: str, uploaded_files: List, state: Dict):
            """Handle questions with document caching."""
            try:
                if not question_text.strip():
                    raise ValueError("❌ Question cannot be empty")
                if not uploaded_files:
                    raise ValueError("❌ No documents uploaded")

                current_hashes = _get_file_hashes(uploaded_files)

                if state["retriever"] is None or current_hashes != state["file_hashes"]:
                    logger.info("Processing new/changed documents...")
                    chunks = processor.process(uploaded_files)
                    retriever = retriever_builder.build_hybrid_retriever(chunks)
                    state.update({"file_hashes": current_hashes, "retriever": retriever})

                result = workflow.full_pipeline(
                    question=question_text,
                    retriever=state["retriever"]
                )
                return result["draft_answer"], result["verification_report"], state

            except Exception as e:
                logger.error(f"Processing error: {e}")
                return f"❌ Error: {str(e)}", "", state

        submit_btn.click(
            fn=process_question,
            inputs=[question, files, session_state],
            outputs=[answer_output, verification_output, session_state]
        )

    share = settings.GRADIO_SHARE
    demo.launch(server_name="127.0.0.1", server_port=5000, share=share)


def _get_file_hashes(uploaded_files: List) -> frozenset:
    """Generate SHA-256 hashes for uploaded files (handles both path strings and file objects)."""
    hashes = set()
    for file in uploaded_files:
        path = file if isinstance(file, str) else file.name
        with open(path, "rb") as f:
            hashes.add(hashlib.sha256(f.read()).hexdigest())
    return frozenset(hashes)


if __name__ == "__main__":
    main()
