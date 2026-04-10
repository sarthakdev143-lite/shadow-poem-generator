---
title: Shadow Poem Generator
emoji: 🌙
colorFrom: indigo
colorTo: pink
sdk: gradio
python_version: "3.12"
sdk_version: 6.6.0
app_file: app.py
suggested_hardware: cpu-basic
pinned: false
license: mit
tags:
- gradio
- pytorch
- text-generation
- poetry
short_description: Classical poetry generator with an included LSTM checkpoint and Gradio UI
---

# Shadow Poem Generator

Shadow is a compact poetry-generation project built around a word-level PyTorch LSTM, with an additional nano-transformer training path for local experiments. The repository is now structured to work as both:

- A normal GitHub repo with training scripts, corpora, and local app entrypoints.
- A Hugging Face Gradio Space using `app.py` as the public entrypoint.

## Repo Layout

- `app.py`: Gradio UI and Hugging Face Space entrypoint.
- `server.py` and `index.html`: optional FastAPI API plus custom browser UI for local or Docker use.
- `shadow.py`: main LSTM script for corpus cleaning, training, free-text generation, and stanza generation.
- `shadow_transformer.py`: experimental nano-transformer training and generation script.
- `build_combined_v3.py`: helper for building expanded local poetry corpora.
- `poem_word_lstm_a1.pt`: included inference checkpoint, tracked with Git LFS.
- `poems.txt`, `poems_clean.txt`, `poems_combined_v3_clean.txt`: corpora kept in the repo for reproducibility and training.

## Setup

Use Python 3.12 for the same environment this repo was last smoke-tested with.

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
pip install -r requirements.txt
```

macOS or Linux:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

If you plan to clone or push the included checkpoint, enable Git LFS before your first push:

```bash
git lfs install
```

If you want `server.py` to prefer a local GPT-2 export, install `transformers` as an extra:

```bash
pip install transformers
```

## Run Locally

### Gradio app

```bash
python app.py
```

The default UI runs at `http://localhost:7860`.

You can point the UI at a different checkpoint with `SHADOW_CHECKPOINT`.

Windows PowerShell:

```powershell
$env:SHADOW_CHECKPOINT="poem_word_lstm_a1.pt"
python app.py
```

### FastAPI server

```bash
python server.py
```

The local API server runs at `http://localhost:8000`. If a local `shadow_gpt2/final` directory exists and `transformers` is installed, `server.py` tries GPT-2 first and otherwise falls back to the included LSTM checkpoint.

### Docker

```bash
docker build -t shadow-poem-generator .
docker run --rm -p 7860:7860 shadow-poem-generator
```

## CLI Examples

### Clean a corpus

```bash
python shadow.py --mode clean_corpus --clean-input poems.txt --clean-output poems_clean.txt
```

### Generate poems with the included LSTM checkpoint

```bash
python shadow.py --mode generate_poems --checkpoint poem_word_lstm_a1.pt --prompt "In the quiet night"
```

### Train or resume an LSTM checkpoint

```bash
python shadow.py --mode train --text-path poems_clean.txt --checkpoint poem_word_lstm_local.pt
python shadow.py --mode train --resume --text-path poems_clean.txt --checkpoint poem_word_lstm_local.pt
```

### Train or sample from the nano transformer

```bash
python shadow_transformer.py --mode train_and_generate_poems --text-path poems_clean.txt --checkpoint shadow_nano.pt
python shadow_transformer.py --mode generate_poems --checkpoint shadow_nano.pt --prompt "Death be not proud"
```

## Upload To GitHub

Create the GitHub repository first, then push this repo with Git LFS enabled:

```bash
git lfs install
git remote add origin https://github.com/<your-user>/shadow-poem-generator.git
git push -u origin main
```

The included `poem_word_lstm_a1.pt` checkpoint is already configured for Git LFS through `.gitattributes`.

## Upload To Hugging Face Spaces

1. Create a new **Gradio** Space in the Hugging Face UI.
2. Keep `README.md` in the repo root. The YAML front matter at the top is the Space config.
3. Push the same repository to the Space remote:

```bash
git lfs install
git remote add hf https://huggingface.co/spaces/<your-user>/shadow-poem-generator
git push hf main
```

The Space will start from `app.py` and use the included LSTM checkpoint. No extra download step is required as long as Git LFS uploads successfully.

## Notes

- `app.py` and `server.py` now resolve checkpoints relative to the repo, so they do not depend on the current working directory.
- Extra corpora, evaluation logs, and experimental checkpoints stay ignored unless explicitly whitelisted.
- For a public Space, keep the repo lean and avoid committing `venv`, local notebook checkpoints, or large experimental model exports.
