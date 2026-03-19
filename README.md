---
title: Shadow Poem Generator
emoji: 🌙
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.6.0
app_file: app.py
pinned: false
license: mit
short_description: Classical poetry generator with LSTM UI and training scripts
---

# Shadow Poem Generator

Shadow is a small poetry-generation project built around a word-level LSTM, with an additional nano-transformer training path for local experiments. The repository includes a Gradio app, a FastAPI server with a custom HTML frontend, corpus-cleaning utilities, and CLI scripts for training and poem generation.

## What is in this repo

- `app.py`: Gradio interface used by the Hugging Face Space entrypoint.
- `server.py` and `index.html`: FastAPI API plus a custom browser UI for local or Docker use.
- `shadow.py`: main LSTM script for corpus cleaning, training, free-text generation, and stanza generation.
- `shadow_transformer.py`: from-scratch nano transformer training and generation script in pure PyTorch.
- `shadow_colab_train.ipynb`: Colab notebook for transformer training runs.
- `build_combined_v3.py`: local corpus-building helper for larger public-domain poetry datasets.
- `poem_word_lstm_a1.pt`: included LSTM checkpoint for immediate local use.
- `poems.txt` and `poems_clean.txt`: raw and cleaned text corpora currently in the repo.

## Setup

Use Python 3.10+.

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

If you want the optional GPT-2 export path used by `server.py`, install `transformers` locally as well:

```bash
pip install transformers
```

## Run the apps

### Gradio app

```bash
python app.py
```

This is the default app path for Hugging Face Spaces and usually runs at `http://localhost:7860`.

### FastAPI server

```bash
python server.py
```

When run directly, the server starts at `http://localhost:8000`.

You can also run it explicitly through Uvicorn:

```bash
uvicorn server:app --host 0.0.0.0 --port 7860
```

If a local `shadow_gpt2/final` folder exists and `transformers` is installed, `server.py` will try that model first. Otherwise it falls back to the LSTM checkpoint.

### Docker

```bash
docker build -t shadow-poem-generator .
docker run --rm -p 7860:7860 shadow-poem-generator
```

## CLI examples

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

## Notes

- Large experimental corpora, extra checkpoints, and evaluation logs are intentionally treated as local artifacts rather than normal GitHub source files.
- `build_combined_v3.py` expects local corpus source folders that are not part of a fresh clone.
- The Gradio app is LSTM-first. The FastAPI server is the more flexible local entrypoint if you want model auto-selection.

## Prompt ideas

- `In the quiet night`
- `I miss her`
- `Death be not proud`
- `When she is gone`
- `O soul where art thou`
