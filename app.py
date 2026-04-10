"""
Shadow - Poem Generator UI.
Run with: python app.py
Then open http://localhost:7860 in your browser.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Set

import gradio as gr
import torch

try:
    from shadow import generate_poems, generate_text, load_checkpoint
except ImportError:
    print("ERROR: shadow.py not found. Make sure app.py and shadow.py are in the same folder.")
    sys.exit(1)


BASE_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cpu")
MODEL = None
STOI: Dict[str, int] = {}
ITOS: Dict[int, str] = {}
KNOWN_WORDS: Set[str] = set()
CHECKPOINT_NAME = ""

DEFAULT_CHECKPOINT = os.environ.get("SHADOW_CHECKPOINT", "").strip()
CHECKPOINT_OPTIONS = [
    "poem_word_lstm_a1.pt",
    "poem_word_lstm.pt",
    "char_lstm_poem.pt",
]
PROMPT_EXAMPLES = [
    "In the quiet night",
    "At dawn I remember",
    "The rose and the thorn",
    "When summer fades",
    "O soul, where art thou",
]


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    repo_relative = BASE_DIR / path
    if repo_relative.exists():
        return repo_relative
    return Path.cwd() / path


def iter_checkpoint_candidates() -> list[str]:
    candidates = [DEFAULT_CHECKPOINT, *CHECKPOINT_OPTIONS]
    deduped: list[str] = []
    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        deduped.append(candidate)
        seen.add(candidate)
    return deduped


def load_model(checkpoint_path: str) -> str:
    global MODEL, STOI, ITOS, KNOWN_WORDS, CHECKPOINT_NAME

    checkpoint_path = checkpoint_path.strip()
    if not checkpoint_path:
        return "No checkpoint path provided."

    resolved_path = resolve_repo_path(checkpoint_path)
    if not resolved_path.exists():
        return f"Checkpoint not found: `{checkpoint_path}`"

    try:
        MODEL, STOI, ITOS, KNOWN_WORDS, _ = load_checkpoint(str(resolved_path), DEVICE)
        MODEL.eval()
        CHECKPOINT_NAME = resolved_path.name
        return f"Loaded **{resolved_path.name}**. Vocabulary size: {len(STOI)} words."
    except Exception as exc:
        return f"Failed to load checkpoint: `{exc}`"


def generate_free_text(
    prompt: str,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    num_tokens: int,
) -> str:
    if MODEL is None:
        return "No model loaded. Load a checkpoint first."
    if not prompt.strip():
        return "Please enter a prompt."

    try:
        return generate_text(
            model=MODEL,
            stoi=STOI,
            itos=ITOS,
            prompt=prompt.strip(),
            device=DEVICE,
            num_tokens=int(num_tokens),
            temperature=float(temperature),
            top_k=int(top_k),
            repetition_penalty=float(repetition_penalty),
            repetition_window=64,
        )
    except Exception as exc:
        return f"Generation error: {exc}"


def generate_poem_stanzas(
    prompt: str,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    poem_count: int,
    lines_per_poem: int,
    line_retries: int,
) -> str:
    if MODEL is None:
        return "No model loaded. Load a checkpoint first."
    if not prompt.strip():
        return "Please enter a prompt."

    try:
        poems = generate_poems(
            model=MODEL,
            stoi=STOI,
            itos=ITOS,
            prompt=prompt.strip(),
            known_words=KNOWN_WORDS,
            device=DEVICE,
            poem_count=int(poem_count),
            lines_per_poem=int(lines_per_poem),
            line_max_tokens=24,
            line_retries=int(line_retries),
            temperature=float(temperature),
            top_k=int(top_k),
            repetition_penalty=float(repetition_penalty),
            repetition_window=64,
        )
    except Exception as exc:
        return f"Generation error: {exc}"

    rendered_poems = []
    for index, stanza in enumerate(poems, start=1):
        rendered_poems.append(f"-- Poem {index} --")
        rendered_poems.extend(stanza)
        rendered_poems.append("")
    return "\n".join(rendered_poems).strip()


startup_status = "No checkpoint loaded yet. Open Model Settings to load one manually."
default_checkpoint_input = iter_checkpoint_candidates()[0]
for checkpoint_name in iter_checkpoint_candidates():
    resolved_checkpoint = resolve_repo_path(checkpoint_name)
    if resolved_checkpoint.exists():
        default_checkpoint_input = checkpoint_name
        startup_status = load_model(checkpoint_name)
        break


CSS = """
#title {
    text-align: center;
    font-size: 2em;
    font-weight: bold;
    margin-bottom: 0;
}

#subtitle {
    text-align: center;
    color: #777;
    margin-top: 0;
    margin-bottom: 1.5em;
}

.poem-output textarea {
    font-family: Georgia, serif;
    font-size: 1.05em;
    line-height: 1.8;
}
"""


with gr.Blocks(title="Shadow - Poem Generator") as demo:
    gr.Markdown("# Shadow", elem_id="title")
    gr.Markdown(
        "*A classical poetry generator trained on Shakespeare, Blake, and Dickinson.*",
        elem_id="subtitle",
    )

    with gr.Accordion("Model Settings", open=False):
        with gr.Row():
            ckpt_input = gr.Textbox(
                label="Checkpoint path",
                value=default_checkpoint_input,
                placeholder="poem_word_lstm_a1.pt",
                scale=4,
            )
            load_btn = gr.Button("Load", variant="secondary", scale=1)
        model_status = gr.Markdown(startup_status)
        load_btn.click(fn=load_model, inputs=ckpt_input, outputs=model_status)

    gr.Markdown("---")

    prompt_input = gr.Textbox(
        label="Opening prompt",
        placeholder="In the quiet night, At dawn I remember, The rose and the thorn...",
        value="In the quiet night",
        lines=2,
    )
    gr.Examples(examples=PROMPT_EXAMPLES, inputs=prompt_input)

    with gr.Tabs():
        with gr.Tab("Free Verse"):
            with gr.Row():
                with gr.Column(scale=1):
                    ft_temp = gr.Slider(
                        0.4,
                        1.2,
                        value=0.64,
                        step=0.02,
                        label="Temperature",
                        info="Lower is tighter, higher is wilder.",
                    )
                    ft_topk = gr.Slider(
                        5,
                        80,
                        value=24,
                        step=1,
                        label="Top-K",
                        info="Limit word choices at each step.",
                    )
                    ft_rep = gr.Slider(
                        1.0,
                        1.5,
                        value=1.16,
                        step=0.01,
                        label="Repetition penalty",
                    )
                    ft_tokens = gr.Slider(
                        60,
                        300,
                        value=160,
                        step=10,
                        label="Max tokens",
                    )
                    ft_btn = gr.Button("Generate", variant="primary")
                with gr.Column(scale=2):
                    ft_output = gr.Textbox(
                        label="Generated poem",
                        lines=18,
                        elem_classes=["poem-output"],
                        buttons=["copy"],
                    )
            ft_btn.click(
                fn=generate_free_text,
                inputs=[prompt_input, ft_temp, ft_topk, ft_rep, ft_tokens],
                outputs=ft_output,
            )

        with gr.Tab("Structured Stanzas"):
            with gr.Row():
                with gr.Column(scale=1):
                    sp_temp = gr.Slider(0.4, 1.2, value=0.64, step=0.02, label="Temperature")
                    sp_topk = gr.Slider(5, 80, value=24, step=1, label="Top-K")
                    sp_rep = gr.Slider(1.0, 1.5, value=1.16, step=0.01, label="Repetition penalty")
                    sp_count = gr.Slider(1, 6, value=3, step=1, label="Number of poems")
                    sp_lines = gr.Radio([2, 4], value=4, label="Lines per poem")
                    sp_retries = gr.Slider(
                        5,
                        60,
                        value=30,
                        step=5,
                        label="Retries per line",
                        info="Higher values are slower but usually cleaner.",
                    )
                    sp_btn = gr.Button("Generate", variant="primary")
                with gr.Column(scale=2):
                    sp_output = gr.Textbox(
                        label="Generated stanzas",
                        lines=18,
                        elem_classes=["poem-output"],
                        buttons=["copy"],
                    )
            sp_btn.click(
                fn=generate_poem_stanzas,
                inputs=[prompt_input, sp_temp, sp_topk, sp_rep, sp_count, sp_lines, sp_retries],
                outputs=sp_output,
            )

    with gr.Accordion("Tips", open=False):
        gr.Markdown(
            """
**Prompts that work well**
- `In the quiet night`
- `At dawn I remember`
- `The rose and the thorn`
- `When summer fades`
- `O soul, where art thou`

**Temperature guide**
- `0.55-0.65`: tighter and more classical
- `0.65-0.75`: balanced
- `0.75-0.90`: more surprising, sometimes stranger
"""
        )

    gr.Markdown(
        "<center><sub>Built with PyTorch + Gradio · Running on CPU</sub></center>"
    )


if __name__ == "__main__":
    print("\nShadow Poem Generator")
    print("Opening at http://localhost:7860\n")
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        share=False,
        css=CSS,
        theme=gr.themes.Soft(),
    )
