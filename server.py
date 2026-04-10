"""
Shadow - FastAPI server.
Supports both the original LSTM and an optional GPT-2 export.
Run with: python server.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Set

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cpu")
MODEL = None
MODEL_TYPE: str | None = None
LOAD_ERROR: str | None = None
STOI = {}
ITOS = {}
KNOWN_WORDS: Set[str] = set()


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    repo_relative = BASE_DIR / path
    if repo_relative.exists():
        return repo_relative
    return Path.cwd() / path


def unique_candidates(*groups: str) -> list[str]:
    candidates: list[str] = []
    seen = set()
    for candidate in groups:
        if not candidate or candidate in seen:
            continue
        candidates.append(candidate)
        seen.add(candidate)
    return candidates


def load_models() -> None:
    global MODEL, MODEL_TYPE, STOI, ITOS, KNOWN_WORDS, LOAD_ERROR

    load_failures: list[str] = []

    gpt2_candidates = unique_candidates(
        os.environ.get("SHADOW_GPT2_DIR", "").strip(),
        "shadow_gpt2/final",
    )
    for candidate in gpt2_candidates:
        gpt2_path = resolve_repo_path(candidate)
        if not gpt2_path.exists():
            continue
        try:
            from transformers import GPT2LMHeadModel, GPT2Tokenizer

            print(f"Loading GPT-2 model from {gpt2_path}...")
            MODEL = {
                "model": GPT2LMHeadModel.from_pretrained(str(gpt2_path)).to(DEVICE),
                "tokenizer": GPT2Tokenizer.from_pretrained(str(gpt2_path)),
            }
            MODEL["model"].eval()
            MODEL_TYPE = "gpt2"
            LOAD_ERROR = None
            print(f"Loaded GPT-2 model from {gpt2_path}.")
            return
        except Exception as exc:
            load_failures.append(f"GPT-2 load failed for {gpt2_path}: {exc}")

    try:
        from shadow import load_checkpoint

        checkpoint_candidates = unique_candidates(
            os.environ.get("SHADOW_CHECKPOINT", "").strip(),
            "poem_word_lstm_a1.pt",
            "poem_word_lstm_a1pp_curated.pt",
            "poem_word_lstm.pt",
        )
        for checkpoint_name in checkpoint_candidates:
            checkpoint_path = resolve_repo_path(checkpoint_name)
            if not checkpoint_path.exists():
                continue
            MODEL, STOI, ITOS, KNOWN_WORDS, _ = load_checkpoint(str(checkpoint_path), DEVICE)
            MODEL.eval()
            MODEL_TYPE = "lstm"
            LOAD_ERROR = None
            print(f"Loaded LSTM checkpoint from {checkpoint_path}.")
            return
    except Exception as exc:
        load_failures.append(f"LSTM load failed: {exc}")

    MODEL = None
    MODEL_TYPE = None
    LOAD_ERROR = (
        "No supported model was found. Set SHADOW_CHECKPOINT or add one of: "
        "poem_word_lstm_a1.pt, poem_word_lstm_a1pp_curated.pt, poem_word_lstm.pt."
    )
    if load_failures:
        LOAD_ERROR = f"{LOAD_ERROR} Details: {' | '.join(load_failures)}"
    print(LOAD_ERROR)


load_models()


def unavailable_response() -> JSONResponse | None:
    if MODEL_TYPE is not None and MODEL is not None:
        return None
    return JSONResponse(
        status_code=503,
        content={"error": LOAD_ERROR or "No model is currently available."},
    )


def gpt2_generate(prompt, max_tokens, temperature, top_k, top_p, rep_penalty):
    tokenizer = MODEL["tokenizer"]
    model = MODEL["model"]
    inputs = tokenizer.encode(f"<poem> {prompt}", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model.generate(
            inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=rep_penalty,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output[0], skip_special_tokens=False)
    text = text.replace("<poem>", "").replace("</poem>", "").replace("<line>", "\n")
    return re.sub(r"<[^>]+>", "", text).strip()


def gpt2_generate_poems(prompt, poem_count, lines_per_poem, temperature, top_k, top_p, rep_penalty):
    poems = []
    for _ in range(poem_count):
        raw = gpt2_generate(prompt, lines_per_poem * 20, temperature, top_k, top_p, rep_penalty)
        lines = [line.strip() for line in raw.split("\n") if line.strip()]
        good_lines = [line for line in lines if 4 <= len(line.split()) <= 14 and line[0].isalpha()]
        while len(good_lines) < lines_per_poem:
            good_lines.append("The quiet wind remembers what we are.")
        poems.append(good_lines[:lines_per_poem])
    return poems


app = FastAPI(title="Shadow Poem Generator")


@app.get("/", response_class=HTMLResponse)
def index():
    return (BASE_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/model_info")
def model_info():
    return {"model_type": MODEL_TYPE, "load_error": LOAD_ERROR}


class TextRequest(BaseModel):
    prompt: str = "In the quiet night"
    temperature: float = 0.74
    top_k: int = 40
    top_p: float = 0.92
    repetition_penalty: float = 1.15
    num_tokens: int = 160


class PoemRequest(BaseModel):
    prompt: str = "In the quiet night"
    temperature: float = 0.74
    top_k: int = 40
    top_p: float = 0.92
    repetition_penalty: float = 1.15
    poem_count: int = 3
    lines_per_poem: int = 4


@app.post("/generate")
def generate(req: TextRequest):
    unavailable = unavailable_response()
    if unavailable:
        return unavailable

    try:
        if MODEL_TYPE == "gpt2":
            text = gpt2_generate(
                req.prompt,
                req.num_tokens,
                req.temperature,
                req.top_k,
                req.top_p,
                req.repetition_penalty,
            )
        else:
            from shadow import generate_text

            text = generate_text(
                model=MODEL,
                stoi=STOI,
                itos=ITOS,
                prompt=req.prompt,
                device=DEVICE,
                num_tokens=req.num_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                repetition_penalty=req.repetition_penalty,
                repetition_window=64,
            )
        return {"text": text, "model": MODEL_TYPE}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.post("/generate_poems")
def gen_poems(req: PoemRequest):
    unavailable = unavailable_response()
    if unavailable:
        return unavailable

    try:
        if MODEL_TYPE == "gpt2":
            poems = gpt2_generate_poems(
                req.prompt,
                req.poem_count,
                req.lines_per_poem,
                req.temperature,
                req.top_k,
                req.top_p,
                req.repetition_penalty,
            )
        else:
            from shadow import generate_poems

            poems = generate_poems(
                model=MODEL,
                stoi=STOI,
                itos=ITOS,
                prompt=req.prompt,
                known_words=KNOWN_WORDS,
                device=DEVICE,
                poem_count=req.poem_count,
                lines_per_poem=req.lines_per_poem,
                line_max_tokens=24,
                line_retries=30,
                temperature=req.temperature,
                top_k=req.top_k,
                repetition_penalty=req.repetition_penalty,
                repetition_window=64,
            )
        return {"poems": poems, "model": MODEL_TYPE}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


if __name__ == "__main__":
    if MODEL_TYPE:
        print(f"\nShadow Poem Generator [{MODEL_TYPE.upper()}]")
    else:
        print("\nShadow Poem Generator [NO MODEL]")
    print("Open http://localhost:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
