import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
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

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[.,;:!?()\"-]")
PUNCT_NO_SPACE_BEFORE = {".", ",", ";", ":", "!", "?", ")", "]", "}"}
PUNCT_NO_SPACE_AFTER = {"(", "[", "{", '"'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Word-level LSTM poem generator (poem-only).")
    parser.add_argument(
        "--mode",
        choices=["train", "generate", "generate_poems", "train_and_generate", "train_and_generate_poems"],
        default="train_and_generate_poems",
    )

    # Data options
    parser.add_argument("--text-path", type=str, default="poems_clean.txt")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--style-path", type=str, default="")
    parser.add_argument("--style-repeat", type=int, default=0)
    parser.add_argument("--max-vocab", type=int, default=3000, help="Includes special tokens.")
    parser.add_argument("--min-freq", type=int, default=2)

    # Model options
    parser.add_argument("--emb-dim", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--max-params", type=int, default=5_000_000)

    # Training options
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--steps-per-epoch", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--seq-len", type=int, default=72)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    # Checkpoint options
    parser.add_argument("--checkpoint", type=str, default="poem_word_lstm.pt")
    parser.add_argument("--resume", action="store_true", help="Continue training from --checkpoint if it exists.")

    # Generation options
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--temperature", action="append", type=float, default=[])
    parser.add_argument("--num-tokens", type=int, default=220)
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--repetition-penalty", type=float, default=1.18)
    parser.add_argument("--repetition-window", type=int, default=64)
    parser.add_argument("--samples-log", type=str, default="poem_samples.txt")
    parser.add_argument("--append-log", action="store_true")

    # Poem generation options
    parser.add_argument("--poem-count", type=int, default=4)
    parser.add_argument("--lines-per-poem", type=int, default=4, choices=[2, 4])
    parser.add_argument("--line-max-tokens", type=int, default=24)
    parser.add_argument("--line-retries", type=int, default=40)

    return parser.parse_args()


class WordLSTM(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 96, hidden_dim: int = 192):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, hidden=None):
        emb = self.embedding(x)
        out, hidden = self.lstm(emb, hidden)
        logits = self.fc(out)
        return logits, hidden


def load_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Text file not found: {p}")
    return p.read_text(encoding="utf-8-sig")


def normalize_text(text: str) -> str:
    replacements = {
        "\r\n": "\n",
        "\r": "\n",
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2014": "-",
        "\u2013": "-",
        "\t": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[ ]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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
    return "".join(out).strip()


def get_batch(data_tensor: torch.Tensor, seq_len: int, batch_size: int, device: torch.device):
    if len(data_tensor) <= seq_len + 1:
        raise ValueError(f"Data too short for seq_len={seq_len}.")
    starts = torch.randint(0, len(data_tensor) - seq_len - 1, (batch_size,))
    x = torch.stack([data_tensor[i : i + seq_len] for i in starts])
    y = torch.stack([data_tensor[i + 1 : i + seq_len + 1] for i in starts])
    return x.to(device), y.to(device)


def train_model(
    model: nn.Module,
    data_tensor: torch.Tensor,
    vocab_size: int,
    device: torch.device,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    seq_len: int,
    lr: float,
    grad_clip: float,
) -> List[float]:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses: List[float] = []
    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for _ in range(steps_per_epoch):
            xb, yb = get_batch(data_tensor, seq_len=seq_len, batch_size=batch_size, device=device)
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits.reshape(-1, vocab_size), yb.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / steps_per_epoch
        losses.append(avg_loss)
        print(f"Epoch {epoch:02d}/{epochs} | Loss: {avg_loss:.4f}")
    return losses


def _apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.numel():
        return logits
    values, idx = torch.topk(logits, k=top_k)
    filtered = torch.full_like(logits, float("-inf"))
    filtered[idx] = values
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


def sample_next_id(
    logits: torch.Tensor,
    generated_ids: List[int],
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    repetition_window: int,
    banned_ids: Optional[Set[int]] = None,
) -> int:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    adjusted = logits / temperature
    adjusted = _apply_repetition_penalty(adjusted, generated_ids, repetition_penalty, repetition_window)
    adjusted = _apply_top_k(adjusted, top_k=top_k)
    if banned_ids:
        for tok_id in banned_ids:
            adjusted[tok_id] = float("-inf")
    probs = torch.softmax(adjusted, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def generate_token_ids(
    model: nn.Module,
    seed_ids: List[int],
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    repetition_window: int,
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
            logits, hidden = model(cur, hidden)
            next_id = sample_next_id(
                logits=logits[:, -1, :].squeeze(0),
                generated_ids=history,
                temperature=temperature,
                top_k=top_k,
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


def score_line(line: str, known_words: Set[str]) -> float:
    if not line:
        return -1e9
    words = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", line.lower())
    if not words:
        return -1e9
    known_ratio = sum(1 for w in words if w in known_words) / len(words)
    alpha_ratio = sum(ch.isalpha() for ch in line) / max(1, len(line))
    repeats = len(re.findall(r"(.)\1{3,}", line))
    score = 0.0
    score += known_ratio * 3.0
    score += alpha_ratio * 1.5
    score += max(0.0, 1.0 - abs(len(words) - 8) / 8.0)
    score -= repeats * 1.2
    if len(words) < 6:
        score -= 2.0
    if len(words) > 18:
        score -= 1.0
    if known_ratio < 0.78:
        score -= 1.2
    if line[-1] in ".!?;,":
        score += 0.2
    if len(line) < 18:
        score -= 0.8
    return score


def normalize_line(line: str) -> str:
    line = re.sub(r"\s+", " ", line).strip()
    if line and line[0].isalpha():
        line = line[0].upper() + line[1:]
    return line


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
) -> str:
    prompt_tokens = tokenize(normalize_text(prompt))
    if not prompt_tokens:
        seed_ids = [stoi[BOS]]
    else:
        seed_ids = [stoi.get(tok, stoi[UNK]) for tok in prompt_tokens]
    banned = {stoi[PAD], stoi[UNK], stoi[BOS]}
    gen_ids = generate_token_ids(
        model=model,
        seed_ids=seed_ids,
        device=device,
        max_new_tokens=num_tokens,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        repetition_window=repetition_window,
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
) -> List[List[str]]:
    if poem_count <= 0:
        raise ValueError("poem_count must be > 0")
    if lines_per_poem not in {2, 4}:
        raise ValueError("lines_per_poem must be 2 or 4")

    nl_id = stoi[NL]
    banned = {stoi[PAD], stoi[UNK], stoi[BOS]}

    poems: List[List[str]] = []
    for _ in range(poem_count):
        lines: List[str] = []
        context = prompt.strip()
        for _line_idx in range(lines_per_poem):
            seed_tokens = tokenize(normalize_text(context))
            if not seed_tokens:
                seed_ids = [stoi[BOS]]
            else:
                seed_ids = [stoi.get(tok, stoi[UNK]) for tok in seed_tokens]

            best_line = ""
            best_score = -1e9
            for _ in range(max(1, line_retries)):
                new_ids = generate_token_ids(
                    model=model,
                    seed_ids=seed_ids,
                    device=device,
                    max_new_tokens=line_max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    repetition_window=repetition_window,
                    stop_token_id=nl_id,
                    banned_ids=banned,
                    min_tokens_before_stop=8,
                )
                candidate_tokens = [itos[i] for i in new_ids]
                line = normalize_line(detokenize(candidate_tokens).split("\n", 1)[0])
                score = score_line(line, known_words)
                if score > best_score:
                    best_score = score
                    best_line = line
                if score >= 3.6:
                    break

            if not best_line:
                best_line = "..."
            lines.append(best_line)
            context = "\n".join(lines)
        poems.append(lines)
    return poems


def save_checkpoint(
    path: str,
    model: nn.Module,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    emb_dim: int,
    hidden_dim: int,
    known_words: Set[str],
    losses: Optional[List[float]] = None,
) -> None:
    ckpt = {
        "model_state_dict": model.state_dict(),
        "stoi": stoi,
        "itos": itos,
        "emb_dim": emb_dim,
        "hidden_dim": hidden_dim,
        "known_words": sorted(known_words),
        "losses": losses or [],
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
        emb_dim=int(ckpt.get("emb_dim", 96)),
        hidden_dim=int(ckpt.get("hidden_dim", 192)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
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
    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    prompts = args.prompt or ["Moonlight on the river", "In the quiet night", "At dawn, I remember"]
    temperatures = args.temperature or [0.72, 0.82]

    model = None
    stoi = None
    itos = None
    known_words: Set[str] = set()
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
            model, stoi, itos, known_words, _ = load_checkpoint(args.checkpoint, device=device)
            data = encode_tokens(tokens, stoi)
            print(f"Resuming from checkpoint: {args.checkpoint}")
        else:
            stoi, itos = build_vocab(tokens, max_vocab=args.max_vocab, min_freq=args.min_freq)
            data = encode_tokens(tokens, stoi)
            known_words = set(tok for tok in tokens if tok.isalpha() and tok in stoi)
            model = WordLSTM(vocab_size=len(stoi), emb_dim=args.emb_dim, hidden_dim=args.hidden_dim).to(device)

        num_params = sum(p.numel() for p in model.parameters())
        if num_params >= args.max_params:
            raise ValueError(f"Model too large: {num_params} params (max={args.max_params})")
        print(f"Parameters: {num_params}")
        print(f"Corpus tokens: {len(tokens)} | Vocab size: {len(stoi)}")

        losses = train_model(
            model=model,
            data_tensor=data,
            vocab_size=len(stoi),
            device=device,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            lr=args.lr,
            grad_clip=args.grad_clip,
        )

        save_checkpoint(
            path=args.checkpoint,
            model=model,
            stoi=stoi,
            itos=itos,
            emb_dim=args.emb_dim,
            hidden_dim=args.hidden_dim,
            known_words=known_words,
            losses=losses,
        )

    if text_requested:
        if model is None or stoi is None or itos is None:
            model, stoi, itos, known_words, _ = load_checkpoint(args.checkpoint, device=device)
            print(f"Loaded checkpoint: {args.checkpoint}")
        print("\n=== Generated Text ===")
        for prompt in prompts:
            for temp in temperatures:
                for sidx in range(1, args.num_samples + 1):
                    print(f"\nPrompt='{prompt}' | temp={temp:.2f} | sample={sidx}")
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
                    )
                    print(text)
                    sample_records.append(
                        "\n".join(
                            [
                                f"[text] prompt={prompt!r} temp={temp:.2f} sample={sidx}",
                                text,
                            ]
                        )
                    )

    if poem_requested:
        if model is None or stoi is None or itos is None:
            model, stoi, itos, known_words, _ = load_checkpoint(args.checkpoint, device=device)
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
                    )
                    print(f"\nPoems | prompt='{prompt}' | temp={temp:.2f} | sample={sidx}")
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
                                f"[poem] prompt={prompt!r} temp={temp:.2f} sample={sidx}",
                                "\n\n".join(blocks),
                            ]
                        )
                    )

    if text_requested or poem_requested:
        write_samples_log(path=args.samples_log, records=sample_records, append=args.append_log)


if __name__ == "__main__":
    main()
