"""
Shadow â€” Nano Transformer (from scratch, pure PyTorch)
A tiny GPT-style transformer trained on classical poetry.

Architecture:
  - Word-level tokenization (same vocab as before)
  - Learned positional embeddings
  - Pre-norm transformer blocks (more stable than post-norm)
  - Multi-head self-attention
  - Weight-tied input/output embeddings (saves ~30% params)
  - Top-k + top-p + repetition penalty sampling

Run:
  python shadow_transformer.py --mode train_and_generate_poems
  python shadow_transformer.py --mode generate_poems --checkpoint shadow_nano.pt
"""

import argparse
import copy
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# â”€â”€ Special tokens (same as LSTM version for compatibility) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"
NL  = "<nl>"
SPECIAL_TOKENS = [PAD, UNK, BOS, EOS, NL]

WORD_RE       = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[.,;:!?()]")
ALPHA_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
PUNCT_NO_SPACE_BEFORE = {".", ",", ";", ":", "!", "?", ")"}
PUNCT_NO_SPACE_AFTER  = {"("}


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  MODEL
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product multi-head attention with causal mask."""

    def __init__(self, emb_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim  = emb_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.qkv  = nn.Linear(emb_dim, 3 * emb_dim, bias=False)
        self.out  = nn.Linear(emb_dim, emb_dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.num_heads, self.head_dim

        qkv = self.qkv(x).reshape(B, T, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]               # (B, H, T, D)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, H, T, T)

        # Causal mask â€” can't attend to future tokens
        causal = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(causal, float("-inf"))
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = self.drop(F.softmax(attn, dim=-1))
        out  = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class FeedForward(nn.Module):
    """Position-wise feed-forward with GELU activation."""

    def __init__(self, emb_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LayerNorm â†’ Attn â†’ residual, LayerNorm â†’ FFN â†’ residual."""

    def __init__(self, emb_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn  = MultiHeadSelfAttention(emb_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.ffn   = FeedForward(emb_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class NanoTransformer(nn.Module):
    """
    Tiny GPT-style decoder-only transformer.
      vocab=5000, emb=128, heads=4, layers=4, ffn=512, seq=128
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim:    int   = 128,
        num_heads:  int   = 4,
        num_layers: int   = 4,
        ffn_dim:    int   = 512,
        max_seq:    int   = 128,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.max_seq = max_seq

        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(max_seq, emb_dim)
        self.drop    = nn.Dropout(dropout)

        self.blocks  = nn.ModuleList([
            TransformerBlock(emb_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm    = nn.LayerNorm(emb_dim)
        self.head    = nn.Linear(emb_dim, vocab_size, bias=False)

        # Weight tying â€” output head shares weights with token embedding
        # This saves ~vocab_size * emb_dim params and improves quality
        self.head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        assert T <= self.max_seq, f"Sequence length {T} exceeds max_seq {self.max_seq}"

        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h   = self.drop(self.tok_emb(x) + self.pos_emb(pos))

        for block in self.blocks:
            h = block(h)

        h      = self.norm(h)
        logits = self.head(h)      # (B, T, vocab_size)
        return logits


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  TOKENIZATION  (identical to shadow.py for full compatibility)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def normalize_text(text: str) -> str:
    replacements = {
        "\r\n": "\n", "\r": "\n",
        "\u2019": "'", "\u2018": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2014": " ", "\u2013": " ",
        "\u00a0": " ", "\t": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"--+", " ", text)
    text = re.sub(r"[ ]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    for line in text.split("\n"):
        toks = WORD_RE.findall(line)
        tokens.extend(t.lower() for t in toks)
        tokens.append(NL)
    if tokens and tokens[-1] == NL:
        tokens.pop()
    return tokens


def build_vocab(tokens: Sequence[str], max_vocab: int, min_freq: int) -> Tuple[Dict[str, int], Dict[int, str]]:
    counter = Counter(tokens)
    keep = [t for t, f in counter.most_common() if f >= min_freq and t not in SPECIAL_TOKENS]
    keep = keep[: max(0, max_vocab - len(SPECIAL_TOKENS))]
    vocab = SPECIAL_TOKENS + keep
    stoi  = {t: i for i, t in enumerate(vocab)}
    itos  = {i: t for t, i in stoi.items()}
    return stoi, itos


def encode_tokens(tokens: Sequence[str], stoi: Dict[str, int]) -> torch.Tensor:
    unk = stoi[UNK]
    return torch.tensor([stoi.get(t, unk) for t in tokens], dtype=torch.long)


def detokenize(tokens: Sequence[str]) -> str:
    out: List[str] = []
    for tok in tokens:
        if tok in {PAD, UNK, BOS, EOS}:
            continue
        if tok == NL:
            if out:
                out[-1] = out[-1].rstrip()
            out.append("\n")
            continue
        if tok in PUNCT_NO_SPACE_BEFORE:
            if out:
                out[-1] = out[-1].rstrip() + tok + " "
            else:
                out.append(tok + " ")
            continue
        if tok in PUNCT_NO_SPACE_AFTER:
            out.append(tok)
            continue
        out.append(tok + " ")
    text = "".join(out).strip()
    text = re.sub(r"\bi\b", "I", text)
    return text


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  DATA
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def get_batch(data: torch.Tensor, seq_len: int, batch_size: int, device: torch.device):
    starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i     : i + seq_len    ] for i in starts])
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in starts])
    return x.to(device), y.to(device)


def split_train_val(data: torch.Tensor, val_split: float) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if val_split <= 0:
        return data, None
    n = int(len(data) * (1 - val_split))
    return data[:n], data[n:]


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  TRAINING
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def train_model(
    model:          NanoTransformer,
    train_data:     torch.Tensor,
    val_data:       Optional[torch.Tensor],
    vocab_size:     int,
    device:         torch.device,
    epochs:         int,
    steps_per_epoch:int,
    batch_size:     int,
    seq_len:        int,
    lr:             float,
    grad_clip:      float,
    warmup_steps:   int,
    patience:       int,
) -> List[float]:

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05, ignore_index=0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2, betas=(0.9, 0.95))

    # Cosine LR with linear warmup
    total_steps = epochs * steps_per_epoch
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val   = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    bad_epochs = 0
    losses: List[float] = []
    global_step = 0

    model.train()
    for epoch in range(1, epochs + 1):
        running = 0.0
        for _ in range(steps_per_epoch):
            xb, yb = get_batch(train_data, seq_len, batch_size, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)                              # (B, T, V)
            loss   = criterion(logits.reshape(-1, vocab_size), yb.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            running    += loss.item()
            global_step += 1

        train_loss = running / steps_per_epoch
        losses.append(train_loss)

        # Validation
        val_loss = train_loss
        if val_data is not None:
            model.eval()
            with torch.no_grad():
                vl = 0.0
                for _ in range(60):
                    xb, yb = get_batch(val_data, seq_len, batch_size, device)
                    logits = model(xb)
                    vl += criterion(logits.reshape(-1, vocab_size), yb.reshape(-1)).item()
                val_loss = vl / 60
            model.train()

        improved = val_loss < best_val - 1e-4
        if improved:
            best_val   = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train={train_loss:.4f} | val={val_loss:.4f} | "
            f"lr={current_lr:.5f}{' [best]' if improved else ''}"
        )

        if patience > 0 and bad_epochs >= patience:
            print(f"Early stop at epoch {epoch}.")
            break

    model.load_state_dict(best_state)
    return losses


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  GENERATION
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.numel():
        return logits
    vals, idx = torch.topk(logits, k)
    out = torch.full_like(logits, float("-inf"))
    out[idx] = vals
    return out


def _apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    remove = cumprobs > p
    remove[0] = False
    sorted_logits[remove] = float("-inf")
    out = torch.full_like(logits, float("-inf"))
    out[sorted_idx] = sorted_logits
    return out


def _apply_rep_penalty(logits: torch.Tensor, ids: List[int], penalty: float, window: int) -> torch.Tensor:
    if penalty <= 1.0 or not ids:
        return logits
    out    = logits.clone()
    recent = ids[-window:] if window > 0 else ids
    for tid in set(recent):
        out[tid] = out[tid] / penalty if out[tid] > 0 else out[tid] * penalty
    return out


def _no_repeat_ngram_ban(history: List[int], n: int) -> Set[int]:
    if n <= 1 or len(history) < n - 1:
        return set()
    prefix = tuple(history[-(n-1):])
    banned: Set[int] = set()
    for i in range(len(history) - n + 1):
        if tuple(history[i:i+n-1]) == prefix:
            banned.add(history[i+n-1])
    return banned


@torch.no_grad()
def generate(
    model:       NanoTransformer,
    stoi:        Dict[str, int],
    itos:        Dict[int, str],
    prompt:      str,
    device:      torch.device,
    max_new:     int   = 200,
    temperature: float = 0.74,
    top_k:       int   = 40,
    top_p:       float = 0.92,
    rep_penalty: float = 1.15,
    rep_window:  int   = 64,
    no_repeat_n: int   = 3,
    stop_id:     Optional[int] = None,
    banned:      Optional[Set[int]] = None,
    min_before_stop: int = 0,
) -> List[int]:
    model.eval()
    prompt_ids = [stoi.get(t, stoi[UNK]) for t in tokenize(normalize_text(prompt))] or [stoi[BOS]]
    history    = list(prompt_ids)
    generated: List[int] = []
    base_banned = set(banned or set()) | {stoi[PAD], stoi[UNK], stoi[BOS]}

    for step in range(max_new):
        # Use last max_seq tokens as context
        ctx = torch.tensor(history[-model.max_seq:], dtype=torch.long, device=device).unsqueeze(0)
        logits = model(ctx)[0, -1, :]    # logits for next token

        step_banned = set(base_banned)
        if stop_id is not None and step < min_before_stop:
            step_banned.add(stop_id)
        step_banned.update(_no_repeat_ngram_ban(history, no_repeat_n))

        logits = logits / temperature
        logits = _apply_rep_penalty(logits, history, rep_penalty, rep_window)
        logits = _apply_top_k(logits, top_k)
        logits = _apply_top_p(logits, top_p)
        for bid in step_banned:
            logits[bid] = float("-inf")

        # Fallback if all logits are -inf
        if torch.isinf(logits).all():
            logits = torch.zeros_like(logits)
            for bid in base_banned:
                logits[bid] = float("-inf")

        probs  = F.softmax(logits, dim=-1)
        next_id = int(torch.multinomial(probs, 1).item())

        generated.append(next_id)
        history.append(next_id)

        if stop_id is not None and next_id == stop_id:
            break

    return generated


def generate_text(
    model:       NanoTransformer,
    stoi:        Dict[str, int],
    itos:        Dict[int, str],
    prompt:      str,
    device:      torch.device,
    num_tokens:  int   = 200,
    temperature: float = 0.74,
    top_k:       int   = 40,
    repetition_penalty: float = 1.15,
    repetition_window:  int   = 64,
    top_p:       float = 0.92,
    no_repeat_ngram: int = 3,
) -> str:
    prompt_ids = [stoi.get(t, stoi[UNK]) for t in tokenize(normalize_text(prompt))] or [stoi[BOS]]
    gen_ids    = generate(
        model=model, stoi=stoi, itos=itos, prompt=prompt, device=device,
        max_new=num_tokens, temperature=temperature, top_k=top_k, top_p=top_p,
        rep_penalty=repetition_penalty, rep_window=repetition_window, no_repeat_n=no_repeat_ngram,
    )
    all_ids = prompt_ids + gen_ids
    return detokenize([itos.get(i, UNK) for i in all_ids])


# â”€â”€ Line scoring (same logic as shadow.py) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _words(line: str) -> List[str]:
    return [w.lower() for w in ALPHA_WORD_RE.findall(line)]


def score_line(line: str, known_words: Set[str]) -> float:
    if not line or not line[0].isalpha():
        return -1e9
    words = _words(line)
    if not words:
        return -1e9
    known_r  = sum(1 for w in words if w in known_words) / len(words)
    unique_r = len(set(words)) / len(words)
    score    = known_r * 4.0 + unique_r * 1.5
    score   += max(0.0, 1.0 - abs(len(words) - 8) / 8.0) * 1.5
    score   -= sum(v-1 for v in Counter(words).values() if v > 1) * 0.8
    if len(words) < 5:   score -= 1.5
    if known_r < 0.78:   score -= 1.2
    if len(line) < 20:   score -= 0.5
    return score


def _sim(a: str, b: str) -> float:
    aw, bw = set(_words(a)), set(_words(b))
    if not aw or not bw:
        return 0.0
    return len(aw & bw) / len(aw | bw)


def generate_poems(
    model:       NanoTransformer,
    stoi:        Dict[str, int],
    itos:        Dict[int, str],
    prompt:      str,
    known_words: Set[str],
    device:      torch.device,
    poem_count:  int   = 4,
    lines_per_poem: int = 4,
    line_max_tokens: int = 24,
    line_retries: int   = 30,
    temperature: float = 0.74,
    top_k:       int   = 40,
    repetition_penalty: float = 1.15,
    repetition_window:  int   = 64,
    top_p:       float = 0.92,
    no_repeat_ngram: int = 3,
    candidates: int = 12,
    **kwargs,   # absorb extra args for compatibility
) -> List[List[str]]:
    nl_id     = stoi[NL]
    base_ban  = {stoi[PAD], stoi[UNK], stoi[BOS]}
    poems: List[List[str]] = []

    for _ in range(poem_count):
        best_lines: List[str] = []
        best_score = -1e9

        for _ in range(max(1, line_retries)):
            # Generate a long chunk and extract good lines
            gen_ids = generate(
                model=model, stoi=stoi, itos=itos, prompt=prompt, device=device,
                max_new=line_max_tokens * lines_per_poem * 3,
                temperature=temperature, top_k=top_k, top_p=top_p,
                rep_penalty=repetition_penalty, rep_window=repetition_window,
                no_repeat_n=no_repeat_ngram, banned=base_ban,
            )
            raw = detokenize([itos.get(i, UNK) for i in gen_ids])
            line_candidates = []
            for ln in raw.split("\n"):
                ln = re.sub(r"\s+", " ", ln).strip()
                if not ln or not ln[0].isalpha():
                    continue
                words = _words(ln)
                if 5 <= len(words) <= 14:
                    # Capitalize first letter
                    ln = ln[0].upper() + ln[1:]
                    line_candidates.append(ln)

            if not line_candidates:
                continue

            # Greedy diverse selection
            ranked = sorted(line_candidates, key=lambda l: score_line(l, known_words), reverse=True)
            selected: List[str] = []
            for cand in ranked[: max(1, candidates)]:
                if all(_sim(cand, prev) < 0.65 for prev in selected):
                    selected.append(cand)
                if len(selected) >= lines_per_poem:
                    break

            if len(selected) < lines_per_poem:
                continue

            poem_score = sum(score_line(l, known_words) for l in selected) / len(selected)
            if poem_score > best_score:
                best_score = poem_score
                best_lines = selected[:lines_per_poem]

        if not best_lines:
            best_lines = ["The quiet wind remembers what we are."] * lines_per_poem
        poems.append(best_lines)

    return poems


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  CHECKPOINT
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def save_checkpoint(path: str, model: NanoTransformer, stoi, itos, known_words, config: dict, losses) -> None:
    torch.save({
        "model_state_dict": model.state_dict(),
        "stoi":        stoi,
        "itos":        itos,
        "known_words": sorted(known_words),
        "config":      config,
        "losses":      losses,
        "model_type":  "nano_transformer",
    }, path)
    print(f"Saved: {path}")


def load_checkpoint(path: str, device: torch.device):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    ckpt = torch.load(p, map_location=device)
    stoi = ckpt["stoi"]
    itos = {int(k): v for k, v in ckpt["itos"].items()} if isinstance(next(iter(ckpt["itos"])), str) else ckpt["itos"]
    cfg  = ckpt.get("config", {})
    model = NanoTransformer(
        vocab_size = len(stoi),
        emb_dim    = cfg.get("emb_dim",    128),
        num_heads  = cfg.get("num_heads",  4),
        num_layers = cfg.get("num_layers", 4),
        ffn_dim    = cfg.get("ffn_dim",    512),
        max_seq    = cfg.get("seq_len",    128),
        dropout    = cfg.get("dropout",    0.1),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    known_words = set(ckpt.get("known_words", []))
    return model, stoi, itos, known_words, ckpt


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  CLI
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def parse_args():
    p = argparse.ArgumentParser(description="Shadow Nano Transformer â€” from scratch poetry generator.")
    p.add_argument("--mode", default="train_and_generate_poems",
                   choices=["train","generate","generate_poems","train_and_generate","train_and_generate_poems"])

    # Data
    p.add_argument("--text-path",    default="poems_clean.txt")
    p.add_argument("--repeat",       type=int,   default=1)
    p.add_argument("--max-vocab",    type=int,   default=5000)
    p.add_argument("--min-freq",     type=int,   default=2)

    # Model
    p.add_argument("--emb-dim",      type=int,   default=128)
    p.add_argument("--num-heads",    type=int,   default=4)
    p.add_argument("--num-layers",   type=int,   default=4)
    p.add_argument("--ffn-dim",      type=int,   default=512)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--max-params",   type=int,   default=5_000_000)

    # Training
    p.add_argument("--epochs",           type=int,   default=30)
    p.add_argument("--steps-per-epoch",  type=int,   default=300)
    p.add_argument("--batch-size",       type=int,   default=32)
    p.add_argument("--seq-len",          type=int,   default=128)
    p.add_argument("--lr",               type=float, default=3e-4)
    p.add_argument("--grad-clip",        type=float, default=1.0)
    p.add_argument("--warmup-steps",     type=int,   default=200)
    p.add_argument("--patience",         type=int,   default=8)
    p.add_argument("--val-split",        type=float, default=0.03)
    p.add_argument("--seed",             type=int,   default=42)

    # Checkpoint
    p.add_argument("--checkpoint", default="shadow_nano.pt")
    p.add_argument("--resume",     action="store_true")

    # Generation
    p.add_argument("--prompt",       action="append", default=[])
    p.add_argument("--temperature",  action="append", type=float, default=[])
    p.add_argument("--num-tokens",   type=int,   default=200)
    p.add_argument("--num-samples",  type=int,   default=2)
    p.add_argument("--top-k",        type=int,   default=40)
    p.add_argument("--top-p",        type=float, default=0.92)
    p.add_argument("--rep-penalty",  type=float, default=1.15)
    p.add_argument("--rep-window",   type=int,   default=64)
    p.add_argument("--no-repeat-n",  type=int,   default=3)
    p.add_argument("--samples-log",  default="nano_samples.txt")
    p.add_argument("--append-log",   action="store_true")

    # Poem generation
    p.add_argument("--poem-count",      type=int, default=4)
    p.add_argument("--lines-per-poem",  type=int, default=4, choices=[2,4])
    p.add_argument("--line-max-tokens", type=int, default=28)
    p.add_argument("--line-retries",    type=int, default=30)
    p.add_argument("--poem-candidates", type=int, default=12)

    return p.parse_args()


def write_log(path: str, records: List[str], append: bool) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, mode, encoding="utf-8") as f:
        f.write(f"=== Generation run @ {ts} ===\n\n")
        for rec in records:
            f.write(rec + "\n\n")
    print(f"Log written: {path}")


def main():
    args   = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    prompts      = args.prompt      or ["In the quiet night", "I miss her", "When she is gone"]
    temperatures = args.temperature or [0.74, 0.82]

    model       = None
    stoi        = None
    itos        = None
    known_words : Set[str] = set()
    records     : List[str] = []

    do_train = args.mode in {"train","train_and_generate","train_and_generate_poems"}
    do_text  = args.mode in {"generate","train_and_generate","train_and_generate_poems"}
    do_poems = args.mode in {"generate_poems","train_and_generate_poems"}

    if do_train:
        raw    = Path(args.text_path).read_text(encoding="utf-8-sig")
        corpus = (normalize_text(raw) + "\n") * args.repeat
        tokens = tokenize(corpus)

        if args.resume and Path(args.checkpoint).exists():
            model, stoi, itos, known_words, _ = load_checkpoint(args.checkpoint, device)
            print(f"Resuming from {args.checkpoint}")
            # IMPORTANT: keep checkpoint vocab when resuming so token ids stay aligned
            data = encode_tokens(tokens, stoi)
        else:
            stoi, itos = build_vocab(tokens, args.max_vocab, args.min_freq)
            data   = encode_tokens(tokens, stoi)
            known_words = {t.lower() for t in tokens if ALPHA_WORD_RE.fullmatch(t) and t in stoi}
            model = NanoTransformer(
                vocab_size = len(stoi),
                emb_dim    = args.emb_dim,
                num_heads  = args.num_heads,
                num_layers = args.num_layers,
                ffn_dim    = args.ffn_dim,
                max_seq    = args.seq_len,
                dropout    = args.dropout,
            ).to(device)

        num_params = sum(p.numel() for p in model.parameters())
        if num_params > args.max_params:
            raise ValueError(f"Model too large: {num_params:,} params (max {args.max_params:,})")

        train_data, val_data = split_train_val(data, args.val_split)
        print(f"Params: {num_params:,} | Vocab: {len(stoi)} | Tokens: {len(tokens):,}")
        print(f"Architecture: {args.num_layers} layers Ã— {args.num_heads} heads Ã— emb{args.emb_dim} Ã— ffn{args.ffn_dim}")

        losses = train_model(
            model=model, train_data=train_data, val_data=val_data,
            vocab_size=len(stoi), device=device,
            epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size, seq_len=args.seq_len,
            lr=args.lr, grad_clip=args.grad_clip,
            warmup_steps=args.warmup_steps, patience=args.patience,
        )

        config = dict(
            emb_dim=args.emb_dim, num_heads=args.num_heads,
            num_layers=args.num_layers, ffn_dim=args.ffn_dim,
            seq_len=args.seq_len, dropout=args.dropout,
        )
        save_checkpoint(args.checkpoint, model, stoi, itos, known_words, config, losses)

    if do_text or do_poems:
        if model is None:
            model, stoi, itos, known_words, _ = load_checkpoint(args.checkpoint, device)
            print(f"Loaded: {args.checkpoint}")

    if do_text:
        print("\n=== Generated Text ===")
        for prompt in prompts:
            for temp in temperatures:
                for s in range(1, args.num_samples + 1):
                    print(f"\nPrompt='{prompt}' | temp={temp:.2f} | sample={s}")
                    out = generate_text(
                        model=model, stoi=stoi, itos=itos, prompt=prompt, device=device,
                        num_tokens=args.num_tokens, temperature=temp,
                        top_k=args.top_k, repetition_penalty=args.rep_penalty,
                        repetition_window=args.rep_window, top_p=args.top_p,
                        no_repeat_ngram=args.no_repeat_n,
                    )
                    print(out)
                    records.append(f"[text] prompt={prompt!r} temp={temp:.2f} sample={s}\n{out}")

    if do_poems:
        print("\n=== Poem Samples ===")
        for prompt in prompts:
            for temp in temperatures:
                for s in range(1, args.num_samples + 1):
                    poems = generate_poems(
                        model=model, stoi=stoi, itos=itos, prompt=prompt,
                        known_words=known_words, device=device,
                        poem_count=args.poem_count, lines_per_poem=args.lines_per_poem,
                        line_max_tokens=args.line_max_tokens, line_retries=args.line_retries,
                        candidates=args.poem_candidates,
                        temperature=temp, top_k=args.top_k,
                        repetition_penalty=args.rep_penalty, repetition_window=args.rep_window,
                        top_p=args.top_p, no_repeat_ngram=args.no_repeat_n,
                    )
                    print(f"\nPoems | prompt='{prompt}' | temp={temp:.2f} | sample={s}")
                    blocks = []
                    for i, poem in enumerate(poems, 1):
                        print(f"[Poem {i}]")
                        for ln in poem: print(ln)
                        print()
                        blocks.append("\n".join(poem))
                    records.append(
                        f"[poem] prompt={prompt!r} temp={temp:.2f} sample={s}\n" +
                        "\n\n".join(blocks)
                    )

    if records:
        write_log(args.samples_log, records, args.append_log)


if __name__ == "__main__":
    main()
