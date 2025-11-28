import json
import gradio as gr
from datetime import datetime

LABEL_FILE = "pi_labels_dataset.jsonl"


def save_label(subject, body, is_pi):
    label = "PI" if is_pi else "Non-PI"
    entry = {
        "timestamp": datetime.now().isoformat(),
        "subject": subject,
        "body": body,
        "label": label
    }

    with open(LABEL_FILE, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return f"Saved as {label}!"

# Interface de label gradio - chatgpt


def app_ui():
    with gr.Blocks() as app:
        gr.Markdown("# Labellisation PI / Non-PI")

        subject = gr.Textbox(label="Objet du mail")
        body = gr.Textbox(label="Résumé du mail", lines=10)

        pi_btn = gr.Button("✅ Propriété Intellectuelle (PI)")
        nonpi_btn = gr.Button("❌ Non PI")

        output = gr.Textbox(label="Status")

        pi_btn.click(save_label, inputs=[
                     subject, body, gr.State(True)], outputs=output)
        nonpi_btn.click(save_label, inputs=[
                        subject, body, gr.State(False)], outputs=output)

    return app


if __name__ == "__main__":
    ui = app_ui()
    ui.launch(server_name="0.0.0.0", server_port=7861)
