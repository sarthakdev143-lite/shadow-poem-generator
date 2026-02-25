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
short_description: LSTM trained on the works of Shakespeare, Blake & Dickinson
---

# 🌙 Shadow [Classical Poetry Generator]

A word-level LSTM trained from scratch on the works of **Shakespeare, Blake & Dickinson**.

Built in one day on a 13-year-old laptop (HP ProBook 6470b, i5-3320M) as a personal AI/ML project.

## What it does
- Generates original poetry in the style of classical English poets
- Two modes: **Free Verse** (flowing text) and **Structured Stanzas** (curated quatrains)
- Fully controllable via temperature, top-k, and repetition penalty sliders

## How it was built
- **Architecture:** Single-layer word-level LSTM (~77k parameters)
- **Corpus:** ~1.4M characters of pure poetry — Shakespeare's Sonnets, Blake's Songs, Dickinson's Complete Poems
- **Training:** PyTorch, CPU-only, ~30 minutes per run
- **Quality filter:** Line-level scoring with vocabulary authenticity check and retry sampling

## Tech Stack
- PyTorch (model + training)
- Gradio (UI)
- Python

## Try these prompts
- `In the quiet night`
- `I miss her`
- `Death be not proud`
- `When she is gone`
- `O soul where art thou`