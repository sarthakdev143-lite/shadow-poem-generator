"""
Shadow — Poem Generator UI
Run with: python app.py
Then open http://localhost:7860 in your browser
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Set

import torch
import gradio as gr

# ── Import everything from shadow.py (must be in the same folder) ──────────────
try:
    from shadow import (
        WordLSTM,
        load_checkpoint,
        generate_text,
        generate_poems,
        normalize_text,
    )
except ImportError:
    print("ERROR: shadow.py not found. Make sure app.py and shadow.py are in the same folder.")
    sys.exit(1)

# ── Global model state (loaded once) ──────────────────────────────────────────
DEVICE = torch.device("cpu")
MODEL = None
STOI: Dict[str, int] = {}
ITOS: Dict[int, str] = {}
KNOWN_WORDS: Set[str] = set()
CHECKPOINT_NAME = ""

CHECKPOINT_OPTIONS = [
    "poem_word_lstm_a1.pt",
    "poem_word_lstm.pt",
    "char_lstm_poem.pt",
]


def load_model(checkpoint_path: str):
    global MODEL, STOI, ITOS, KNOWN_WORDS, CHECKPOINT_NAME
    p = Path(checkpoint_path)
    if not p.exists():
        return f"❌ Checkpoint not found: {checkpoint_path}"
    try:
        MODEL, STOI, ITOS, KNOWN_WORDS, _ = load_checkpoint(str(p), DEVICE)
        MODEL.eval()
        CHECKPOINT_NAME = p.name
        vocab_size = len(STOI)
        return f"✅ Loaded **{p.name}** — vocab size: {vocab_size} words"
    except Exception as e:
        return f"❌ Failed to load checkpoint: {e}"


def generate_free_text(
    prompt: str,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    num_tokens: int,
) -> str:
    if MODEL is None:
        return "⚠️ No model loaded. Load a checkpoint first."
    if not prompt.strip():
        return "⚠️ Please enter a prompt."
    try:
        result = generate_text(
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
        return result
    except Exception as e:
        return f"❌ Generation error: {e}"


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
        return "⚠️ No model loaded. Load a checkpoint first."
    if not prompt.strip():
        return "⚠️ Please enter a prompt."
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
        out = []
        for i, stanza in enumerate(poems, 1):
            out.append(f"── Poem {i} ──")
            out.extend(stanza)
            out.append("")
        return "\n".join(out).strip()
    except Exception as e:
        return f"❌ Generation error: {e}"


# ── Auto-load best available checkpoint on startup ─────────────────────────────
startup_status = "No checkpoint loaded yet."
for ckpt in CHECKPOINT_OPTIONS:
    if Path(ckpt).exists():
        startup_status = load_model(ckpt)
        break


# ── UI ─────────────────────────────────────────────────────────────────────────
CSS = """
#title { text-align: center; font-size: 2em; font-weight: bold; margin-bottom: 0; }
#subtitle { text-align: center; color: #888; margin-top: 0; margin-bottom: 1.5em; }
#output_box textarea { font-family: 'Georgia', serif; font-size: 1.05em; line-height: 1.8; }
"""

with gr.Blocks(title="Shadow — Poem Generator") as demo:

    gr.Markdown("# 🌙 Shadow", elem_id="title")
    gr.Markdown("*A classical poetry generator trained on Shakespeare, Blake & Dickinson*", elem_id="subtitle")

    # ── Model loader ────────────────────────────────────────────────────────────
    with gr.Accordion("⚙️ Model Settings", open=False):
        with gr.Row():
            ckpt_input = gr.Textbox(
                label="Checkpoint path",
                value=CHECKPOINT_OPTIONS[0],
                placeholder="poem_word_lstm_a1.pt",
                scale=4,
            )
            load_btn = gr.Button("Load", variant="secondary", scale=1)
        model_status = gr.Markdown(startup_status)
        load_btn.click(fn=load_model, inputs=ckpt_input, outputs=model_status)

    gr.Markdown("---")

    # ── Shared prompt ───────────────────────────────────────────────────────────
    with gr.Row():
        prompt_input = gr.Textbox(
            label="✍️ Opening prompt",
            placeholder="In the quiet night, At dawn I remember, The rose and the thorn...",
            value="In the quiet night",
            scale=5,
        )

    # ── Tabs ────────────────────────────────────────────────────────────────────
    with gr.Tabs():

        # Free text tab
        with gr.Tab("📜 Free Verse"):
            with gr.Row():
                with gr.Column(scale=1):
                    ft_temp   = gr.Slider(0.4, 1.2, value=0.64, step=0.02, label="Temperature  (lower = focused, higher = wild)")
                    ft_topk   = gr.Slider(5, 80, value=24, step=1, label="Top-K  (limits word choices at each step)")
                    ft_rep    = gr.Slider(1.0, 1.5, value=1.16, step=0.01, label="Repetition Penalty")
                    ft_tokens = gr.Slider(60, 300, value=160, step=10, label="Max tokens to generate")
                    ft_btn    = gr.Button("✨ Generate", variant="primary")
                with gr.Column(scale=2):
                    ft_output = gr.Textbox(
                        label="Generated poem",
                        lines=18,
                    )
            ft_btn.click(
                fn=generate_free_text,
                inputs=[prompt_input, ft_temp, ft_topk, ft_rep, ft_tokens],
                outputs=ft_output,
            )

        # Structured poem tab
        with gr.Tab("📖 Structured Stanzas"):
            with gr.Row():
                with gr.Column(scale=1):
                    sp_temp    = gr.Slider(0.4, 1.2, value=0.64, step=0.02, label="Temperature")
                    sp_topk    = gr.Slider(5, 80, value=24, step=1, label="Top-K")
                    sp_rep     = gr.Slider(1.0, 1.5, value=1.16, step=0.01, label="Repetition Penalty")
                    sp_count   = gr.Slider(1, 6, value=3, step=1, label="Number of poems")
                    sp_lines   = gr.Radio([2, 4], value=4, label="Lines per poem")
                    sp_retries = gr.Slider(5, 60, value=30, step=5, label="Retries per line  (higher = better quality, slower)")
                    sp_btn     = gr.Button("✨ Generate", variant="primary")
                with gr.Column(scale=2):
                    sp_output = gr.Textbox(
                        label="Generated stanzas",
                        lines=18,
                    )
            sp_btn.click(
                fn=generate_poem_stanzas,
                inputs=[prompt_input, sp_temp, sp_topk, sp_rep, sp_count, sp_lines, sp_retries],
                outputs=sp_output,
            )

    # ── Tips ────────────────────────────────────────────────────────────────────
    with gr.Accordion("💡 Tips for better poems", open=False):
        gr.Markdown("""
**Prompts that work well** (use words from classical poetry):
- `In the quiet night`
- `At dawn I remember`
- `The rose and the thorn`
- `When summer fades`
- `O soul, where art thou`

**Avoid** starting with `Moonlight` — it's rare in the vocab and may show as `<unk>`

**Temperature guide:**
- `0.55–0.65` → structured, coherent, Shakespearean
- `0.65–0.75` → balanced, poetic blend
- `0.75–0.90` → creative, surprising, sometimes weird

**Retries per line:** Higher = model tries more candidates and picks the best. 
Slower but noticeably better quality on structured stanzas.
        """)

    gr.Markdown(
        "<center><sub>Built with PyTorch + Gradio · Trained on Shakespeare, Blake & Dickinson · Running on CPU</sub></center>"
    )

if __name__ == "__main__":
    print("\n🌙 Shadow Poem Generator")
    print("Opening at http://localhost:7860\n")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft())