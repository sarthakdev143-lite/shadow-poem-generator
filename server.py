"""
Shadow — FastAPI server
Supports both the original LSTM and the new GPT-2 model
Run: python server.py
"""

import sys
import re
from pathlib import Path
from typing import Set, Optional

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

DEVICE = torch.device("cpu")
MODEL = None
MODEL_TYPE = None  # "lstm" or "gpt2"
STOI = {}
ITOS = {}
KNOWN_WORDS: Set[str] = set()

def load_models():
    global MODEL, MODEL_TYPE, STOI, ITOS, KNOWN_WORDS

    # Try GPT-2 first
    gpt2_path = Path("shadow_gpt2/final")
    if gpt2_path.exists():
        try:
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            print("Loading GPT-2 model...")
            MODEL = {
                "model": GPT2LMHeadModel.from_pretrained(str(gpt2_path)).to(DEVICE),
                "tokenizer": GPT2Tokenizer.from_pretrained(str(gpt2_path)),
            }
            MODEL["model"].eval()
            MODEL_TYPE = "gpt2"
            print("✅ GPT-2 loaded!")
            return
        except Exception as e:
            print(f"GPT-2 load failed: {e}, falling back to LSTM...")

    # Fall back to LSTM
    try:
        from shadow import load_checkpoint
        for ckpt in ["poem_word_lstm_a1.pt", "poem_word_lstm_a1pp_curated.pt", "poem_word_lstm.pt"]:
            if Path(ckpt).exists():
                MODEL, STOI, ITOS, KNOWN_WORDS, _ = load_checkpoint(ckpt, DEVICE)
                MODEL.eval()
                MODEL_TYPE = "lstm"
                print(f"✅ LSTM loaded from {ckpt}")
                return
    except Exception as e:
        print(f"LSTM load failed: {e}")

    print("❌ No model found!")
    sys.exit(1)


load_models()

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
    text = re.sub(r"<[^>]+>", "", text).strip()
    return text


def gpt2_generate_poems(prompt, poem_count, lines_per_poem, temperature, top_k, top_p, rep_penalty):
    poems = []
    for _ in range(poem_count):
        raw = gpt2_generate(prompt, lines_per_poem * 20, temperature, top_k, top_p, rep_penalty)
        lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
        good = [ln for ln in lines if 4 <= len(ln.split()) <= 14 and ln[0].isalpha()]
        while len(good) < lines_per_poem:
            good.append("The quiet wind remembers what we are.")
        poems.append(good[:lines_per_poem])
    return poems


app = FastAPI(title="Shadow Poem Generator")


@app.get("/", response_class=HTMLResponse)
def index():
    return Path("index.html").read_text(encoding="utf-8")


@app.get("/model_info")
def model_info():
    return {"model_type": MODEL_TYPE}


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
    try:
        if MODEL_TYPE == "gpt2":
            text = gpt2_generate(req.prompt, req.num_tokens, req.temperature, req.top_k, req.top_p, req.repetition_penalty)
        else:
            from shadow import generate_text
            text = generate_text(
                model=MODEL, stoi=STOI, itos=ITOS, prompt=req.prompt,
                device=DEVICE, num_tokens=req.num_tokens, temperature=req.temperature,
                top_k=req.top_k, repetition_penalty=req.repetition_penalty, repetition_window=64,
            )
        return {"text": text, "model": MODEL_TYPE}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/generate_poems")
def gen_poems(req: PoemRequest):
    try:
        if MODEL_TYPE == "gpt2":
            poems = gpt2_generate_poems(req.prompt, req.poem_count, req.lines_per_poem,
                                         req.temperature, req.top_k, req.top_p, req.repetition_penalty)
        else:
            from shadow import generate_poems
            poems = generate_poems(
                model=MODEL, stoi=STOI, itos=ITOS, prompt=req.prompt,
                known_words=KNOWN_WORDS, device=DEVICE, poem_count=req.poem_count,
                lines_per_poem=req.lines_per_poem, line_max_tokens=24, line_retries=30,
                temperature=req.temperature, top_k=req.top_k,
                repetition_penalty=req.repetition_penalty, repetition_window=64,
            )
        return {"poems": poems, "model": MODEL_TYPE}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    print(f"\n🌙 Shadow Poem Generator [{MODEL_TYPE.upper()}]")
    print("Open http://localhost:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
