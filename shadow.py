import argparse
import copy
from collections import Counter
from datetime import datetime
from pathlib import Path
import random
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"
NL = "<nl>"
SPECIAL_TOKENS = [PAD, UNK, BOS, EOS, NL]

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[.,;:!?()]")
ALPHA_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
PUNCT_NO_SPACE_BEFORE = {".", ",", ";", ":", "!", "?", ")"}
PUNCT_NO_SPACE_AFTER = {"("}

URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
BOOK_RE = re.compile(r"^book\s+[ivxlcdm0-9]+\b", re.IGNORECASE)
CHAPTER_RE = re.compile(r"^chapter\b", re.IGNORECASE)
ROMAN_RE = re.compile(r"^[ivxlcdm]+$", re.IGNORECASE)
ONLY_NUM_RE = re.compile(r"^\d+[\d\s.,:-]*$")
AUTHOR_LINE_RE = re.compile(r"^by\s+[A-Za-z][A-Za-z .'-]{2,}$")
MONTH_RE = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(17|18|19|20)\d{2}\b")
NOISE_KEYWORD_RE = re.compile(
    r"project gutenberg|proofreading|ebook|copyright|all rights reserved|transcriber|"
    r"illustration|table of contents|contents|comment on the poem|introduction|biographical|"
    r"preface|appendix|\bnotes?\b|publish(ed|ing)|editor|goringe",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A1++ word-level LSTM poem generator.")
    parser.add_argument(
        "--mode",
        choices=[
            "clean_corpus",
            "train",
            "generate",
            "generate_poems",
            "train_and_generate",
            "train_and_generate_poems",
        ],
        default="train_and_generate_poems",
    )

    # Corpus cleaning options
    parser.add_argument("--clean-input", type=str, default="poems.txt")
    parser.add_argument("--clean-output", type=str, default="poems_clean.txt")
    parser.add_argument("--clean-min-words", type=int, default=1)
    parser.add_argument("--clean-max-words", type=int, default=18)
    parser.add_argument("--rebuild-clean", action="store_true", help="Rebuild poems_clean.txt before training.")

    # Data options
    parser.add_argument("--text-path", type=str, default="poems_clean.txt")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--style-path", type=str, default="")
    parser.add_argument("--style-repeat", type=int, default=0)
    parser.add_argument("--max-vocab", type=int, default=7000, help="Includes special tokens.")
    parser.add_argument("--min-freq", type=int, default=2)

    # Model options
    parser.add_argument("--emb-dim", type=int, default=160)
    parser.add_argument("--hidden-dim", type=int, default=320)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--max-params", type=int, default=12_000_000)

    # Training options
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=420)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--val-split", type=float, default=0.03)
    parser.add_argument("--eval-steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1.8e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--lr-decay", type=float, default=0.65)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)

    # Checkpoint options
    parser.add_argument("--checkpoint", type=str, default="poem_word_lstm_a1pp.pt")
    parser.add_argument("--resume", action="store_true", help="Continue training from --checkpoint if it exists.")

    # Generation options
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--temperature", action="append", type=float, default=[])
    parser.add_argument("--num-tokens", type=int, default=220)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--repetition-penalty", type=float, default=1.14)
    parser.add_argument("--repetition-window", type=int, default=72)
    parser.add_argument("--no-repeat-ngram", type=int, default=3)
    parser.add_argument("--samples-log", type=str, default="poem_samples_a1pp.txt")
    parser.add_argument("--append-log", action="store_true")

    # Poem generation options
    parser.add_argument("--poem-count", type=int, default=4)
    parser.add_argument("--lines-per-poem", type=int, default=4, choices=[2, 4])
    parser.add_argument("--line-max-tokens", type=int, default=24)
    parser.add_argument("--line-retries", type=int, default=34)
    parser.add_argument("--poem-candidates", type=int, default=10)

    return parser.parse_args()


class WordLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 160,
        hidden_dim: int = 320,
        num_layers: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.emb_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, hidden=None):
        emb = self.embedding(x)
        emb = self.emb_dropout(emb)
        out, hidden = self.lstm(emb, hidden)
        out = self.norm(out)
        logits = self.fc(out)
        return logits, hidden


def load_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Text file not found: {p}")
    return p.read_text(encoding="utf-8-sig", errors="ignore")


def fix_mojibake(text: str) -> str:
    if not any(ch in text for ch in ("Ã", "â", "Â")):
        return text
    try:
        repaired = text.encode("latin1").decode("utf-8")
        if repaired.count("\ufffd") <= text.count("\ufffd"):
            return repaired
    except UnicodeError:
        pass
    return text


def normalize_text(text: str) -> str:
    replacements = {
        "\r\n": "\n",
        "\r": "\n",
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2014": " ",
        "\u2013": " ",
        "\u00a0": " ",
        "\t": " ",
        "â€™": "'",
        "â€˜": "'",
        "â€œ": '"',
        "â€\x9d": '"',
        "â€”": " ",
        "â€“": " ",
        "â€¦": "...",
        "Ã©": "e",
        "Ã¨": "e",
        "Ãª": "e",
        "Ã«": "e",
        "Ã¡": "a",
        "Ã ": "a",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"--+", " ", text)
    text = re.sub(r"[ ]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _looks_like_heading(line: str, words: Sequence[str]) -> bool:
    letters = sum(ch.isalpha() for ch in line)
    uppercase = sum(ch.isupper() for ch in line)
    if letters > 0 and uppercase / letters > 0.92 and len(words) <= 8:
        return True
    return False


def _should_drop_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if URL_RE.search(s):
        return True
    if "[" in s or "]" in s:
        return True
    if re.search(r"\d", s):
        return True
    if NOISE_KEYWORD_RE.search(s):
        return True
    if BOOK_RE.match(s) or CHAPTER_RE.match(s):
        return True
    if ROMAN_RE.match(s):
        return True
    if ONLY_NUM_RE.match(s):
        return True
    if AUTHOR_LINE_RE.match(s):
        return True
    if MONTH_RE.search(s) and YEAR_RE.search(s):
        return True

    words = ALPHA_WORD_RE.findall(s)
    if not words:
        return True
    if _looks_like_heading(s, words):
        return True
    return False


def clean_poem_corpus(text: str, min_words: int = 1, max_words: int = 18) -> Tuple[str, Dict[str, int]]:
    text = normalize_text(text)
    out_lines: List[str] = []
    stats = {
        "total_lines": 0,
        "kept_lines": 0,
        "dropped_noise": 0,
        "dropped_short_or_long": 0,
        "blank_lines": 0,
    }

    for raw_line in text.split("\n"):
        stats["total_lines"] += 1
        line = fix_mojibake(raw_line).strip()
        line = re.sub(r"_([^_]+)_", r"\1", line)
        line = re.sub(r"\s+", " ", line).strip()

        if not line:
            stats["blank_lines"] += 1
            if out_lines and out_lines[-1] != "":
                out_lines.append("")
            continue

        if _should_drop_line(line):
            stats["dropped_noise"] += 1
            continue

        words = ALPHA_WORD_RE.findall(line)
        if len(words) < min_words or len(words) > max_words:
            stats["dropped_short_or_long"] += 1
            continue

        out_lines.append(line)
        stats["kept_lines"] += 1

    while out_lines and out_lines[-1] == "":
        out_lines.pop()

    clean_text = "\n".join(out_lines).strip() + "\n"
    return clean_text, stats


def build_clean_corpus_file(input_path: str, output_path: str, min_words: int = 1, max_words: int = 18) -> Dict[str, int]:
    raw = load_text(input_path)
    clean_text, stats = clean_poem_corpus(raw, min_words=min_words, max_words=max_words)
    out = Path(output_path)
    out.write_text(clean_text, encoding="utf-8")
    token_count = len(tokenize(clean_text))
    stats["tokens"] = token_count
    stats["chars"] = len(clean_text)
    print(f"Clean corpus written: {out}")
    print(
        "Clean stats | "
        f"total={stats['total_lines']} kept={stats['kept_lines']} "
        f"dropped_noise={stats['dropped_noise']} dropped_len={stats['dropped_short_or_long']} "
        f"tokens={stats['tokens']} chars={stats['chars']}"
    )
    return stats


def build_corpus(main_text: str, repeat: int, style_text: str = "", style_repeat: int = 0) -> str:
    chunks: List[str] = []
    if repeat > 0:
        chunks.append((normalize_text(main_text) + "\n") * repeat)
    if style_text and style_repeat > 0:
        chunks.append((normalize_text(style_text) + "\n") * style_repeat)
    corpus = "".join(chunks).strip()
    if not corpus:
        raise ValueError("Corpus is empty. Check repeat/style-repeat and source files.")
    return corpus


def tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    for line in text.split("\n"):
        line_tokens = WORD_RE.findall(line)
        tokens.extend(tok.lower() for tok in line_tokens)
        tokens.append(NL)
    if tokens and tokens[-1] == NL:
        tokens.pop()
    return tokens


def build_vocab(tokens: Sequence[str], max_vocab: int, min_freq: int) -> Tuple[Dict[str, int], Dict[int, str]]:
    counter = Counter(tokens)
    keep = [tok for tok, freq in counter.most_common() if freq >= min_freq and tok not in SPECIAL_TOKENS]
    keep = keep[: max(0, max_vocab - len(SPECIAL_TOKENS))]
    vocab = SPECIAL_TOKENS + keep
    stoi = {tok: i for i, tok in enumerate(vocab)}
    itos = {i: tok for tok, i in stoi.items()}
    return stoi, itos


def encode_tokens(tokens: Sequence[str], stoi: Dict[str, int]) -> torch.Tensor:
    unk_id = stoi[UNK]
    return torch.tensor([stoi.get(tok, unk_id) for tok in tokens], dtype=torch.long)


def build_common_bigrams(tokens: Sequence[str], top_n: int = 12000) -> Set[str]:
    words = [tok for tok in tokens if ALPHA_WORD_RE.fullmatch(tok)]
    counter: Counter[Tuple[str, str]] = Counter()
    for a, b in zip(words, words[1:]):
        counter[(a, b)] += 1
    return {f"{a} {b}" for (a, b), _ in counter.most_common(max(0, top_n))}


def polish_generated_text(text: str) -> str:
    polished: List[str] = []
    for raw_line in text.split("\n"):
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            if polished and polished[-1] != "":
                polished.append("")
            continue
        line = re.sub(r"^[,.;:!?()'\"]+\s*", "", line)
        line = re.sub(r"\s+([,.;:!?])", r"\1", line)
        line = re.sub(r"([,.;:!?])\1{1,}", r"\1", line)
        line = re.sub(r"\bi\b", "I", line)
        if line and line[0].isalpha():
            line = line[0].upper() + line[1:]
        polished.append(line)

    while polished and polished[-1] == "":
        polished.pop()
    return "\n".join(polished)


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
    return polish_generated_text("".join(out).strip())


def get_batch(data_tensor: torch.Tensor, seq_len: int, batch_size: int, device: torch.device):
    if len(data_tensor) <= seq_len + 1:
        raise ValueError(f"Data too short for seq_len={seq_len}.")
    starts = torch.randint(0, len(data_tensor) - seq_len - 1, (batch_size,))
    x = torch.stack([data_tensor[i : i + seq_len] for i in starts])
    y = torch.stack([data_tensor[i + 1 : i + seq_len + 1] for i in starts])
    return x.to(device), y.to(device)


def split_train_val(data_tensor: torch.Tensor, val_split: float, seq_len: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if val_split <= 0.0:
        return data_tensor, None
    if len(data_tensor) < (seq_len + 1) * 3:
        return data_tensor, None

    split_idx = int(len(data_tensor) * (1.0 - val_split))
    split_idx = max(seq_len + 2, min(split_idx, len(data_tensor) - seq_len - 2))
    if split_idx >= len(data_tensor) - seq_len - 1:
        return data_tensor, None

    train_data = data_tensor[:split_idx]
    val_start = max(0, split_idx - seq_len - 1)
    val_data = data_tensor[val_start:]
    if len(val_data) <= seq_len + 1:
        return train_data, None
    return train_data, val_data


def evaluate_model(
    model: nn.Module,
    data_tensor: torch.Tensor,
    vocab_size: int,
    device: torch.device,
    seq_len: int,
    batch_size: int,
    eval_steps: int,
    criterion: nn.Module,
) -> float:
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for _ in range(max(1, eval_steps)):
            xb, yb = get_batch(data_tensor, seq_len=seq_len, batch_size=batch_size, device=device)
            logits, _ = model(xb)
            loss = criterion(logits.reshape(-1, vocab_size), yb.reshape(-1))
            losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses)


def train_model(
    model: nn.Module,
    train_tensor: torch.Tensor,
    val_tensor: Optional[torch.Tensor],
    vocab_size: int,
    device: torch.device,
    epochs: int,
    steps_per_epoch: int,
    eval_steps: int,
    batch_size: int,
    seq_len: int,
    lr: float,
    weight_decay: float,
    lr_decay: float,
    grad_clip: float,
    label_smoothing: float,
    early_stop_patience: int,
) -> Dict[str, object]:
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_decay,
        patience=2,
        min_lr=1e-5,
    )

    history: Dict[str, object] = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
        "best_epoch": 0,
        "best_val_loss": float("inf"),
    }
    best_state = copy.deepcopy(model.state_dict())
    bad_epochs = 0

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for _ in range(steps_per_epoch):
            xb, yb = get_batch(train_tensor, seq_len=seq_len, batch_size=batch_size, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(xb)
            loss = criterion(logits.reshape(-1, vocab_size), yb.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / max(1, steps_per_epoch)
        if val_tensor is not None:
            val_loss = evaluate_model(
                model=model,
                data_tensor=val_tensor,
                vocab_size=vocab_size,
                device=device,
                seq_len=seq_len,
                batch_size=batch_size,
                eval_steps=eval_steps,
                criterion=criterion,
            )
        else:
            val_loss = train_loss

        scheduler.step(val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])
        cast_train = history["train_loss"]
        cast_val = history["val_loss"]
        cast_lr = history["lr"]
        assert isinstance(cast_train, list) and isinstance(cast_val, list) and isinstance(cast_lr, list)
        cast_train.append(float(train_loss))
        cast_val.append(float(val_loss))
        cast_lr.append(current_lr)

        improved = val_loss < float(history["best_val_loss"]) - 1e-4
        if improved:
            history["best_val_loss"] = float(val_loss)
            history["best_epoch"] = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        print(
            f"Epoch {epoch:02d}/{epochs} | train={train_loss:.4f} | val={val_loss:.4f} | "
            f"lr={current_lr:.6f}{' | best' if improved else ''}"
        )

        if early_stop_patience > 0 and bad_epochs >= early_stop_patience:
            print(f"Early stopping at epoch {epoch} (no val improvement for {bad_epochs} epochs).")
            break

    model.load_state_dict(best_state)
    return history


def _apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.numel():
        return logits
    values, idx = torch.topk(logits, k=top_k)
    filtered = torch.full_like(logits, float("-inf"))
    filtered[idx] = values
    return filtered


def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)
    remove = cumulative > top_p
    if remove.numel() > 0:
        remove[0] = False
    sorted_logits[remove] = float("-inf")
    filtered = torch.full_like(logits, float("-inf"))
    filtered[sorted_idx] = sorted_logits
    return filtered


def _apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: List[int],
    repetition_penalty: float,
    repetition_window: int,
) -> torch.Tensor:
    if repetition_penalty <= 1.0 or not generated_ids:
        return logits
    out = logits.clone()
    recent = generated_ids[-repetition_window:] if repetition_window > 0 else generated_ids
    for token_id in set(recent):
        if out[token_id] < 0:
            out[token_id] *= repetition_penalty
        else:
            out[token_id] /= repetition_penalty
    return out


def _collect_no_repeat_ngram_bans(history: List[int], no_repeat_ngram: int) -> Set[int]:
    if no_repeat_ngram <= 1:
        return set()
    prefix_len = no_repeat_ngram - 1
    if len(history) < prefix_len:
        return set()

    target_prefix = tuple(history[-prefix_len:])
    banned: Set[int] = set()
    for i in range(0, len(history) - no_repeat_ngram + 1):
        if tuple(history[i : i + prefix_len]) == target_prefix:
            banned.add(history[i + prefix_len])
    return banned


def sample_next_id(
    logits: torch.Tensor,
    generated_ids: List[int],
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    repetition_window: int,
    banned_ids: Optional[Set[int]] = None,
) -> int:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    adjusted = logits / temperature
    adjusted = _apply_repetition_penalty(adjusted, generated_ids, repetition_penalty, repetition_window)
    adjusted = _apply_top_k(adjusted, top_k=top_k)
    adjusted = _apply_top_p(adjusted, top_p=top_p)

    if banned_ids:
        for tok_id in banned_ids:
            adjusted[tok_id] = float("-inf")

    if torch.isinf(adjusted).all():
        adjusted = logits.clone()
        if banned_ids:
            for tok_id in banned_ids:
                adjusted[tok_id] = float("-inf")

    if torch.isinf(adjusted).all():
        adjusted = torch.zeros_like(logits)

    probs = torch.softmax(adjusted, dim=-1)
    if torch.isnan(probs).any() or float(probs.sum().item()) <= 0.0:
        probs = torch.ones_like(probs) / probs.numel()
    return int(torch.multinomial(probs, num_samples=1).item())


def generate_token_ids(
    model: nn.Module,
    seed_ids: List[int],
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    repetition_window: int,
    no_repeat_ngram: int,
    stop_token_id: Optional[int] = None,
    banned_ids: Optional[Set[int]] = None,
    min_tokens_before_stop: int = 0,
) -> List[int]:
    model.eval()
    with torch.no_grad():
        if not seed_ids:
            raise ValueError("seed_ids cannot be empty")

        x = torch.tensor(seed_ids, dtype=torch.long, device=device).unsqueeze(0)
        hidden = None
        for t in range(x.size(1) - 1):
            _, hidden = model(x[:, t : t + 1], hidden)
        cur = x[:, -1:]

        generated: List[int] = []
        history = list(seed_ids)
        for _ in range(max_new_tokens):
            step_banned = set(banned_ids or set())
            if stop_token_id is not None and len(generated) < min_tokens_before_stop:
                step_banned.add(stop_token_id)
            if no_repeat_ngram > 1:
                step_banned.update(_collect_no_repeat_ngram_bans(history, no_repeat_ngram))

            logits, hidden = model(cur, hidden)
            next_id = sample_next_id(
                logits=logits[:, -1, :].squeeze(0),
                generated_ids=history,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                repetition_window=repetition_window,
                banned_ids=step_banned,
            )
            generated.append(next_id)
            history.append(next_id)
            cur = torch.tensor([[next_id]], dtype=torch.long, device=device)

            if stop_token_id is not None and next_id == stop_token_id:
                break
        return generated


def _word_list(line: str) -> List[str]:
    return [w.lower() for w in ALPHA_WORD_RE.findall(line)]


def _ending_word(line: str) -> str:
    words = _word_list(line)
    return words[-1] if words else ""


def _line_similarity(a: str, b: str) -> float:
    aw = set(_word_list(a))
    bw = set(_word_list(b))
    if not aw or not bw:
        return 0.0
    return len(aw & bw) / len(aw | bw)


def score_line(line: str, known_words: Set[str], common_bigrams: Optional[Set[str]] = None) -> float:
    if not line:
        return -1e9

    words = _word_list(line)
    if not words:
        return -1e9

    counts = Counter(words)
    known_ratio = sum(1 for w in words if w in known_words) / len(words)
    unique_ratio = len(counts) / len(words)
    alpha_ratio = sum(ch.isalpha() for ch in line) / max(1, len(line))
    punct_count = len(re.findall(r"[,.;:!?()]", line))
    punct_ratio = punct_count / max(1, len(line))
    repeat_words = sum(v - 1 for v in counts.values() if v > 1)
    ends_clean = line.strip().endswith((".", "!", "?", ";", ","))
    bigram_ratio = 0.0
    if common_bigrams and len(words) > 1:
        hits = 0
        for a, b in zip(words, words[1:]):
            if f"{a} {b}" in common_bigrams:
                hits += 1
        bigram_ratio = hits / (len(words) - 1)

    score = 0.0
    score += known_ratio * 4.0
    score += unique_ratio * 1.8
    score += alpha_ratio * 1.0
    score += max(0.0, 1.0 - abs(len(words) - 8) / 8.0) * 1.6
    score += bigram_ratio * 2.0
    score += 0.4 if ends_clean else 0.0
    score -= repeat_words * 0.8

    if not line[0].isalpha():
        score -= 3.0
    if len(words) < 5:
        score -= 1.2
    if len(words) > 14:
        score -= 1.0
    if known_ratio < 0.80:
        score -= 1.2
    if unique_ratio < 0.62:
        score -= 1.0
    if punct_ratio > 0.18:
        score -= 1.0
    if re.search(r"(.)\1{3,}", line):
        score -= 1.5
    if line.count("(") != line.count(")"):
        score -= 0.6
    if len(line) < 20:
        score -= 0.5
    return score


def score_line_with_context(
    line: str,
    prior_lines: Sequence[str],
    known_words: Set[str],
    common_bigrams: Optional[Set[str]] = None,
) -> float:
    score = score_line(line, known_words, common_bigrams=common_bigrams)
    if score < -1e8:
        return score

    end_word = _ending_word(line)
    prior_endings = {_ending_word(ln) for ln in prior_lines if ln}
    if end_word and end_word in prior_endings:
        score -= 0.4

    for prev in prior_lines:
        sim = _line_similarity(line, prev)
        if sim > 0.78:
            score -= 2.0
        elif sim > 0.58:
            score -= 1.1
    return score


def score_poem(lines: Sequence[str], known_words: Set[str], common_bigrams: Optional[Set[str]] = None) -> float:
    if not lines:
        return -1e9

    line_scores = [score_line(ln, known_words, common_bigrams=common_bigrams) for ln in lines]
    if any(s < -1e8 for s in line_scores):
        return -1e9

    mean_score = sum(line_scores) / len(line_scores)
    endings = [_ending_word(ln) for ln in lines if _ending_word(ln)]
    ending_diversity = (len(set(endings)) / len(endings)) if endings else 0.0
    line_diversity = len(set(lines)) / len(lines)

    sims: List[float] = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            sims.append(_line_similarity(lines[i], lines[j]))
    avg_sim = sum(sims) / len(sims) if sims else 0.0

    score = mean_score
    score += ending_diversity * 1.1
    score += line_diversity * 1.2
    score -= avg_sim * 1.6
    return score


def normalize_line(line: str) -> str:
    line = re.sub(r"\s+", " ", line).strip()
    line = re.sub(r"\bi\b", "I", line)
    if line and line[0].isalpha():
        line = line[0].upper() + line[1:]
    return line


def prompt_to_ids(prompt_tokens: Sequence[str], stoi: Dict[str, int]) -> List[int]:
    ids: List[int] = []
    for tok in prompt_tokens:
        if tok in stoi:
            ids.append(stoi[tok])
            continue
        for cand in (tok.lower(), tok.capitalize(), tok.upper()):
            if cand in stoi:
                ids.append(stoi[cand])
                break
    if not ids:
        ids = [stoi[BOS]]
    return ids


def generate_text(
    model: nn.Module,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    prompt: str,
    device: torch.device,
    num_tokens: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    repetition_window: int,
    top_p: float = 0.92,
    no_repeat_ngram: int = 3,
) -> str:
    prompt_tokens = tokenize(normalize_text(prompt))
    seed_ids = prompt_to_ids(prompt_tokens, stoi)
    banned = {stoi[PAD], stoi[UNK], stoi[BOS]}

    gen_ids = generate_token_ids(
        model=model,
        seed_ids=seed_ids,
        device=device,
        max_new_tokens=num_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        repetition_window=repetition_window,
        no_repeat_ngram=no_repeat_ngram,
        stop_token_id=None,
        banned_ids=banned,
    )
    out_tokens = [itos[i] for i in seed_ids + gen_ids]
    return detokenize(out_tokens)


def generate_poems(
    model: nn.Module,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    prompt: str,
    known_words: Set[str],
    device: torch.device,
    poem_count: int,
    lines_per_poem: int,
    line_max_tokens: int,
    line_retries: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    repetition_window: int,
    top_p: float = 0.92,
    no_repeat_ngram: int = 3,
    poem_candidates: int = 10,
    common_bigrams: Optional[Set[str]] = None,
) -> List[List[str]]:
    if poem_count <= 0:
        raise ValueError("poem_count must be > 0")
    if lines_per_poem not in {2, 4}:
        raise ValueError("lines_per_poem must be 2 or 4")
    banned = {stoi[PAD], stoi[UNK], stoi[BOS]}

    def extract_candidate_lines(text: str) -> List[str]:
        candidates: List[str] = []
        for raw in text.split("\n"):
            raw = raw.strip()
            if not raw:
                continue
            parts = re.split(r"(?<=[.;!?])\s+", raw)
            for part in parts:
                line = normalize_line(part)
                if not line or not line[0].isalpha():
                    continue
                words = _word_list(line)
                if len(words) < 5 or len(words) > 14:
                    continue
                candidates.append(line)
        return candidates

    poems: List[List[str]] = []
    for _ in range(poem_count):
        best_poem_lines: List[str] = []
        best_poem_score = -1e9

        for _ in range(max(1, poem_candidates)):
            seed_tokens = tokenize(normalize_text(prompt))
            seed_ids = prompt_to_ids(seed_tokens, stoi)
            gen_ids = generate_token_ids(
                model=model,
                seed_ids=seed_ids,
                device=device,
                max_new_tokens=max(line_max_tokens * lines_per_poem * 3, 80),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                repetition_window=repetition_window,
                no_repeat_ngram=no_repeat_ngram,
                stop_token_id=None,
                banned_ids=banned,
            )
            generated_text = detokenize([itos[i] for i in seed_ids + gen_ids])
            pool = extract_candidate_lines(generated_text)
            if not pool:
                continue

            # Pick best diverse lines greedily.
            ranked = sorted(
                pool,
                key=lambda ln: score_line(ln, known_words, common_bigrams=common_bigrams),
                reverse=True,
            )
            lines: List[str] = []
            for cand in ranked:
                if any(_line_similarity(cand, prev) > 0.65 for prev in lines):
                    continue
                lines.append(cand)
                if len(lines) >= lines_per_poem:
                    break

            if len(lines) < lines_per_poem:
                for cand in ranked:
                    if cand in lines:
                        continue
                    lines.append(cand)
                    if len(lines) >= lines_per_poem:
                        break

            if len(lines) < lines_per_poem:
                continue

            poem_score = score_poem(lines[:lines_per_poem], known_words, common_bigrams=common_bigrams)
            if poem_score > best_poem_score:
                best_poem_score = poem_score
                best_poem_lines = lines[:lines_per_poem]

        if not best_poem_lines:
            best_poem_lines = ["The quiet wind remembers what we are."] * lines_per_poem
        poems.append(best_poem_lines)
    return poems


def save_checkpoint(
    path: str,
    model: nn.Module,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    emb_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    known_words: Set[str],
    common_bigrams: Optional[Set[str]] = None,
    history: Optional[Dict[str, object]] = None,
    train_config: Optional[Dict[str, object]] = None,
) -> None:
    ckpt = {
        "model_state_dict": model.state_dict(),
        "stoi": stoi,
        "itos": itos,
        "emb_dim": emb_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "known_words": sorted(known_words),
        "common_bigrams": sorted(common_bigrams or set()),
        "history": history or {},
        "train_config": train_config or {},
    }
    torch.save(ckpt, path)
    print(f"Saved checkpoint: {path}")


def load_checkpoint(path: str, device: torch.device):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    ckpt = torch.load(p, map_location=device)
    stoi = ckpt["stoi"]
    itos_raw = ckpt["itos"]
    itos = {int(k): v for k, v in itos_raw.items()} if isinstance(next(iter(itos_raw.keys())), str) else itos_raw

    model = WordLSTM(
        vocab_size=len(stoi),
        emb_dim=int(ckpt.get("emb_dim", 160)),
        hidden_dim=int(ckpt.get("hidden_dim", 320)),
        num_layers=int(ckpt.get("num_layers", 1)),
        dropout=float(ckpt.get("dropout", 0.0)),
    ).to(device)
    state = ckpt["model_state_dict"]
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        model.load_state_dict(state, strict=False)
    known_words = set(w.lower() for w in ckpt.get("known_words", []))
    return model, stoi, itos, known_words, ckpt


def write_samples_log(path: str, records: List[str], append: bool = False) -> None:
    if not records:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(p, mode, encoding="utf-8") as f:
        f.write(f"=== Generation run @ {ts} ===\n\n")
        for rec in records:
            f.write(rec)
            f.write("\n\n")
    print(f"Wrote generation sweep log: {p}")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    if args.mode == "clean_corpus":
        build_clean_corpus_file(
            input_path=args.clean_input,
            output_path=args.clean_output,
            min_words=args.clean_min_words,
            max_words=args.clean_max_words,
        )
        return

    if args.rebuild_clean:
        build_clean_corpus_file(
            input_path=args.clean_input,
            output_path=args.clean_output,
            min_words=args.clean_min_words,
            max_words=args.clean_max_words,
        )

    prompts = args.prompt or ["In the quiet night", "When dawn returns", "I carry the rain"]
    temperatures = args.temperature or [0.74, 0.80]

    model: Optional[WordLSTM] = None
    stoi: Optional[Dict[str, int]] = None
    itos: Optional[Dict[int, str]] = None
    known_words: Set[str] = set()
    common_bigrams: Set[str] = set()
    sample_records: List[str] = []

    train_requested = args.mode in {"train", "train_and_generate", "train_and_generate_poems"}
    text_requested = args.mode in {"generate", "train_and_generate", "train_and_generate_poems"}
    poem_requested = args.mode in {"generate_poems", "train_and_generate_poems"}

    if train_requested:
        main_text = load_text(args.text_path)
        style_text = ""
        if args.style_path:
            sp = Path(args.style_path)
            if sp.exists():
                style_text = load_text(args.style_path)
            else:
                print(f"Style file not found, skipping: {sp}")

        corpus = build_corpus(main_text, args.repeat, style_text=style_text, style_repeat=args.style_repeat)
        tokens = tokenize(corpus)

        if args.resume and Path(args.checkpoint).exists():
            model, stoi, itos, known_words, ckpt = load_checkpoint(args.checkpoint, device=device)
            print(f"Resuming from checkpoint: {args.checkpoint}")
            if len(stoi) == 0:
                raise ValueError("Checkpoint vocabulary is empty.")
            data = encode_tokens(tokens, stoi)
            if not known_words:
                known_words = set(tok.lower() for tok in tokens if ALPHA_WORD_RE.fullmatch(tok) and tok in stoi)
            common_bigrams = set(ckpt.get("common_bigrams", []))
            if not common_bigrams:
                common_bigrams = build_common_bigrams(tokens, top_n=12000)
            history_prev = ckpt.get("history", {})
            if isinstance(history_prev, dict) and history_prev:
                best_val = history_prev.get("best_val_loss")
                if best_val is not None:
                    print(f"Previous best val loss: {float(best_val):.4f}")
        else:
            stoi, itos = build_vocab(tokens, max_vocab=args.max_vocab, min_freq=args.min_freq)
            data = encode_tokens(tokens, stoi)
            known_words = set(tok.lower() for tok in tokens if ALPHA_WORD_RE.fullmatch(tok) and tok in stoi)
            common_bigrams = build_common_bigrams(tokens, top_n=12000)
            model = WordLSTM(
                vocab_size=len(stoi),
                emb_dim=args.emb_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
            ).to(device)

        assert model is not None and stoi is not None and itos is not None
        num_params = sum(p.numel() for p in model.parameters())
        if num_params >= args.max_params:
            raise ValueError(f"Model too large: {num_params} params (max={args.max_params})")
        print(f"Parameters: {num_params}")
        print(f"Corpus tokens: {len(tokens)} | Vocab size: {len(stoi)}")

        train_data, val_data = split_train_val(data, val_split=args.val_split, seq_len=args.seq_len)
        if val_data is None:
            print("Validation split disabled (data too small or val-split=0).")
        else:
            print(f"Train tokens: {len(train_data)} | Val tokens: {len(val_data)}")

        history = train_model(
            model=model,
            train_tensor=train_data,
            val_tensor=val_data,
            vocab_size=len(stoi),
            device=device,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            eval_steps=args.eval_steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            lr=args.lr,
            weight_decay=args.weight_decay,
            lr_decay=args.lr_decay,
            grad_clip=args.grad_clip,
            label_smoothing=args.label_smoothing,
            early_stop_patience=args.early_stop_patience,
        )

        train_config = {
            "epochs": args.epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "val_split": args.val_split,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram": args.no_repeat_ngram,
        }
        save_checkpoint(
            path=args.checkpoint,
            model=model,
            stoi=stoi,
            itos=itos,
            emb_dim=args.emb_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            known_words=known_words,
            common_bigrams=common_bigrams,
            history=history,
            train_config=train_config,
        )

    if text_requested:
        if model is None or stoi is None or itos is None:
            model, stoi, itos, known_words, ckpt = load_checkpoint(args.checkpoint, device=device)
            common_bigrams = set(ckpt.get("common_bigrams", []))
            if not common_bigrams and Path(args.text_path).exists():
                common_bigrams = build_common_bigrams(tokenize(load_text(args.text_path)), top_n=12000)
            print(f"Loaded checkpoint: {args.checkpoint}")
        print("\n=== Generated Text ===")
        for prompt in prompts:
            for temp in temperatures:
                for sidx in range(1, args.num_samples + 1):
                    print(f"\nPrompt={prompt!r} | temp={temp:.2f} | sample={sidx}")
                    text = generate_text(
                        model=model,
                        stoi=stoi,
                        itos=itos,
                        prompt=prompt,
                        device=device,
                        num_tokens=args.num_tokens,
                        temperature=temp,
                        top_k=args.top_k,
                        repetition_penalty=args.repetition_penalty,
                        repetition_window=args.repetition_window,
                        top_p=args.top_p,
                        no_repeat_ngram=args.no_repeat_ngram,
                    )
                    print(text)
                    sample_records.append(
                        "\n".join(
                            [
                                (
                                    f"[text] prompt={prompt!r} temp={temp:.2f} sample={sidx} "
                                    f"top_k={args.top_k} top_p={args.top_p:.2f} "
                                    f"rep={args.repetition_penalty:.2f} ngram={args.no_repeat_ngram}"
                                ),
                                text,
                            ]
                        )
                    )

    if poem_requested:
        if model is None or stoi is None or itos is None:
            model, stoi, itos, known_words, ckpt = load_checkpoint(args.checkpoint, device=device)
            common_bigrams = set(ckpt.get("common_bigrams", []))
            if not common_bigrams and Path(args.text_path).exists():
                common_bigrams = build_common_bigrams(tokenize(load_text(args.text_path)), top_n=12000)
            print(f"Loaded checkpoint: {args.checkpoint}")

        print("\n=== Poem Samples ===")
        for prompt in prompts:
            for temp in temperatures:
                for sidx in range(1, args.num_samples + 1):
                    poems = generate_poems(
                        model=model,
                        stoi=stoi,
                        itos=itos,
                        prompt=prompt,
                        known_words=known_words,
                        device=device,
                        poem_count=args.poem_count,
                        lines_per_poem=args.lines_per_poem,
                        line_max_tokens=args.line_max_tokens,
                        line_retries=args.line_retries,
                        temperature=temp,
                        top_k=args.top_k,
                        repetition_penalty=args.repetition_penalty,
                        repetition_window=args.repetition_window,
                        top_p=args.top_p,
                        no_repeat_ngram=args.no_repeat_ngram,
                        poem_candidates=args.poem_candidates,
                        common_bigrams=common_bigrams,
                    )
                    print(f"\nPoems | prompt={prompt!r} | temp={temp:.2f} | sample={sidx}")
                    blocks: List[str] = []
                    for i, poem_lines in enumerate(poems, start=1):
                        print(f"[Poem {i}]")
                        for ln in poem_lines:
                            print(ln)
                        print("")
                        blocks.append("\n".join(poem_lines))
                    sample_records.append(
                        "\n".join(
                            [
                                (
                                    f"[poem] prompt={prompt!r} temp={temp:.2f} sample={sidx} "
                                    f"top_k={args.top_k} top_p={args.top_p:.2f} rep={args.repetition_penalty:.2f} "
                                    f"ngram={args.no_repeat_ngram} retries={args.line_retries} "
                                    f"candidates={args.poem_candidates}"
                                ),
                                "\n\n".join(blocks),
                            ]
                        )
                    )

    if text_requested or poem_requested:
        write_samples_log(path=args.samples_log, records=sample_records, append=args.append_log)


if __name__ == "__main__":
    main()
