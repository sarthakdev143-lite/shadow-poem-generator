import json
import os
import re
import ssl
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path


ROOT = Path(".")
INDIAN_DIR = ROOT / "corpus_indian_sources"
WESTERN_DIR = ROOT / "corpus_western_extra"
NOTES_PATH = ROOT / "corpus_expansion_v3_notes.txt"
DOWNLOAD_LOG_PATH = ROOT / "corpus_download_manifest_v3.json"
BREAKDOWN_PATH = ROOT / "corpus_source_breakdown_v3.json"


for key in [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "GIT_HTTP_PROXY",
    "GIT_HTTPS_PROXY",
]:
    os.environ.pop(key, None)


CTX = ssl._create_unverified_context()


REQUESTED_GROUP_A = {
    7164: "tagore_gitanjali.txt",
    6955: "tagore_fruit_gathering.txt",
    6165: "tagore_the_gardener.txt",
    6161: "tagore_stray_birds.txt",
    6162: "tagore_crescent_moon.txt",
    15819: "tagore_lovers_gift.txt",
    27473: "kabir_songs.txt",
    44071: "naidu_golden_threshold.txt",
    44960: "naidu_bird_of_time.txt",
    7070: "toru_dutt_ancient_ballads.txt",
}


# Substitutions discovered via Gutendex lookups.
GROUP_A_SUBSTITUTIONS = {
    "tagore_gitanjali.txt": {"id": 7164, "reason": "original id valid"},
    "tagore_fruit_gathering.txt": {"id": 6522, "from": 6955, "reason": "requested id is wrong/non-poetry"},
    "tagore_the_gardener.txt": {"id": 6686, "from": 6165, "reason": "requested id is wrong/non-poetry"},
    "tagore_stray_birds.txt": {"id": 6524, "from": 6161, "reason": "requested id is wrong/non-poetry"},
    "naidu_golden_threshold.txt": {"id": 680, "from": 44071, "reason": "requested id is wrong/non-poetry"},
    "toru_dutt_ancient_ballads.txt": {"id": 23245, "from": 7070, "reason": "requested id is wrong/non-poetry"},
}


# Group B (query-driven, poetry collections found)
GROUP_B_SELECTED = {
    "manmohan_ghose_primavera.txt": {"id": 19170, "query": "Manmohan Ghose poems"},
}


# Group C (extra western lyric)
GROUP_C_SELECTED = {
    "rossetti_poems_19188.txt": {"id": 19188, "query": "Christina Rossetti poems"},
    "rossetti_goblin_market_16950.txt": {"id": 16950, "query": "Rossetti Goblin Market other poems"},
    "burns_complete_18500.txt": {"id": 18500, "query": "Burns poems songs"},
    "burns_poems_songs_1279.txt": {"id": 1279, "query": "Burns poems songs"},
    "burns_songs_lyrics_75462.txt": {"id": 75462, "query": "Burns poems songs"},
    "clare_poems_52601.txt": {"id": 52601, "query": "John Clare poems"},
    "clare_selected_v1_22443.txt": {"id": 22443, "query": "John Clare poems"},
    "clare_poems_ms_8672.txt": {"id": 8672, "query": "John Clare poems"},
    "clare_life_remains_9156.txt": {"id": 9156, "query": "John Clare poems"},
    "ebb_sonnets_2002.txt": {"id": 2002, "query": "Sonnets from the Portuguese"},
    "hardy_wessex_3167.txt": {"id": 3167, "query": "Hardy poems"},
    "hardy_past_present_3168.txt": {"id": 3168, "query": "Hardy poems"},
    "hardy_wessex_9452.txt": {"id": 9452, "query": "Hardy poems"},
    "hardy_past_present_9430.txt": {"id": 9430, "query": "Hardy poems"},
}


UNRESOLVED_REQUESTED = [
    "tagore_crescent_moon.txt",
    "tagore_lovers_gift.txt",
    "kabir_songs.txt",
    "naidu_bird_of_time.txt",
    "sri_aurobindo_poetry",
    "michael_madhusudan_dutt_poetry",
    "harindranath_chattopadhyay_poetry",
]


START_MARKERS = [
    "*** START OF THE PROJECT GUTENBERG EBOOK",
    "*** START OF THIS PROJECT GUTENBERG EBOOK",
    "*** START OF THE PROJECT GUTENBERG",
]
END_MARKERS = [
    "*** END OF THE PROJECT GUTENBERG EBOOK",
    "*** END OF THIS PROJECT GUTENBERG EBOOK",
    "*** END OF THE PROJECT GUTENBERG",
]


WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
ALPHA_RE = re.compile(r"[A-Za-z]")
ROMAN_RE = re.compile(r"^[IVXLCDM]+$", re.IGNORECASE)
NOISE_RE = re.compile(
    r"project gutenberg|ebook|copyright|proofread|preface|introduction|appendix|editor|edited by|"
    r"table of contents|produced by|all rights reserved|biographical|commentary|transcriber|notes|"
    r"translator|bengali|rabindranath|calcutta|goringe|\bindex\b|\bvolume\b|lines written under",
    re.IGNORECASE,
)
DIALOGUE_RE = re.compile(r"\b(said|answered|replied|cried|spake|quoth)\b", re.IGNORECASE)
KABIR_NOTE_RE = re.compile(r"^(note:|see:)", re.IGNORECASE)

STOP_END = {"with", "to", "for", "of", "and", "or", "that"}


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, context=CTX, timeout=60) as r:
        return json.load(r)


def pick_text_url(formats: dict) -> str | None:
    for key in ("text/plain; charset=utf-8", "text/plain; charset=us-ascii", "text/plain"):
        if formats.get(key):
            return formats[key]
    for k, v in formats.items():
        if k.startswith("text/plain") and v:
            return v
    return None


def download_book(book_id: int, out_path: Path) -> dict:
    meta = fetch_json(f"https://gutendex.com/books/{book_id}")
    text_url = pick_text_url(meta.get("formats", {}))
    if not text_url:
        raise RuntimeError(f"No plain-text format available for {book_id}")
    with urllib.request.urlopen(text_url, context=CTX, timeout=120) as r:
        data = r.read()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    return {
        "id": book_id,
        "title": meta.get("title", ""),
        "authors": [a.get("name", "") for a in meta.get("authors", [])],
        "url": text_url,
        "bytes": len(data),
    }


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2014", " ").replace("\u2013", " ")
    text = text.replace("\ufeff", "")
    text = re.sub(r"\t+", " ", text)
    return text


def extract_body(text: str) -> str:
    up = text.upper()
    start = 0
    for marker in START_MARKERS:
        i = up.find(marker)
        if i != -1:
            nl = text.find("\n", i)
            start = nl + 1 if nl != -1 else i
            break
    end = len(text)
    up2 = text.upper()
    for marker in END_MARKERS:
        i = up2.find(marker)
        if i != -1:
            end = min(end, i)
    return text[start:end]


def first_alpha_lower(line: str) -> bool:
    for ch in line:
        if ch.isalpha():
            return ch.islower()
    return False


def is_heading(lines: list[str], idx: int) -> bool:
    s = lines[idx].strip()
    if not s:
        return False
    words = WORD_RE.findall(s)
    if not words or len(words) > 10:
        return False
    prev_blank = idx == 0 or not lines[idx - 1].strip()
    next_blank = idx == len(lines) - 1 or not lines[idx + 1].strip()
    if not (prev_blank or next_blank):
        return False
    letters = "".join(ch for ch in s if ch.isalpha())
    if not letters:
        return False
    upper_ratio = sum(1 for ch in letters if ch.isupper()) / len(letters)
    title_ratio = sum(1 for w in s.split() if w[:1].isupper()) / max(1, len(s.split()))
    if s.isdigit():
        return True
    if ROMAN_RE.fullmatch(s):
        return True
    if upper_ratio > 0.75:
        return True
    if title_ratio > 0.85 and len(words) <= 8:
        return True
    return False


def split_sections(body: str) -> list[tuple[str, list[str]]]:
    lines = body.split("\n")
    sections: list[tuple[str, list[str]]] = []
    title = "__UNTITLED__"
    cur: list[str] = []
    for i, raw in enumerate(lines):
        s = re.sub(r"\s+", " ", raw).strip()
        if is_heading(lines, i):
            if cur:
                sections.append((title, cur))
            title = s
            cur = []
            continue
        cur.append(s)
    if cur:
        sections.append((title, cur))
    return sections


def universal_line_ok(line: str, max_words: int) -> bool:
    s = re.sub(r"\s+", " ", line).strip()
    if not s:
        return False
    if any(ch in s for ch in ("(", ")", "[", "]", "_")):
        return False
    if '"' in s:
        return False
    if s.startswith('"') or s.startswith("'"):
        return False
    if DIALOGUE_RE.search(s):
        return False
    if re.search(r"\d", s):
        return False
    if first_alpha_lower(s):
        return False
    if NOISE_RE.search(s):
        return False
    words = WORD_RE.findall(s)
    if not words:
        return False
    if len(words) > max_words:
        return False
    if words[-1].lower() in STOP_END:
        return False
    return True


def process_tagore(text: str, max_words: int = 22) -> tuple[str, int]:
    body = extract_body(normalize_text(text))
    out: list[str] = []
    dropped = 0
    started = False
    for raw in body.split("\n"):
        line = re.sub(r"\s+", " ", raw).strip()
        if not line:
            if out and out[-1] != "":
                out.append("")
            continue
        if ROMAN_RE.fullmatch(line):
            started = True
            continue
        # For Gitanjali, drop intro before first Roman numeral section.
        if "yeats" in line.lower():
            dropped += 1
            continue
        if not started and len(WORD_RE.findall(line)) > 16:
            dropped += 1
            continue
        if not universal_line_ok(line, max_words=max_words):
            dropped += 1
            continue
        out.append(line)
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out).strip() + "\n", dropped


def process_kabir(text: str) -> tuple[str, int]:
    body = extract_body(normalize_text(text))
    out: list[str] = []
    dropped = 0
    for raw in body.split("\n"):
        line = re.sub(r"\s+", " ", raw).strip()
        if not line:
            if out and out[-1] != "":
                out.append("")
            continue
        if KABIR_NOTE_RE.match(line):
            dropped += 1
            continue
        if not universal_line_ok(line, max_words=16):
            dropped += 1
            continue
        words = WORD_RE.findall(line)
        if len(words) < 4:
            dropped += 1
            continue
        out.append(line)
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out).strip() + "\n", dropped


def process_simple(text: str, max_words: int) -> tuple[str, int]:
    body = extract_body(normalize_text(text))
    out: list[str] = []
    dropped = 0
    lines = body.split("\n")
    for i, raw in enumerate(lines):
        line = re.sub(r"\s+", " ", raw).strip()
        if not line:
            if out and out[-1] != "":
                out.append("")
            continue
        if is_heading(lines, i):
            dropped += 1
            continue
        if not universal_line_ok(line, max_words=max_words):
            dropped += 1
            continue
        out.append(line)
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out).strip() + "\n", dropped


def process_sectioned(
    text: str,
    *,
    max_words: int,
    max_section_lines: int | None = None,
    min_section_lines: int = 2,
    skip_title_terms: tuple[str, ...] = (),
) -> tuple[str, int]:
    body = extract_body(normalize_text(text))
    sections = split_sections(body)
    out: list[str] = []
    dropped = 0
    for title, lines in sections:
        t = title.lower()
        if any(term in t for term in skip_title_terms):
            dropped += len(lines)
            continue
        nonblank = [ln for ln in lines if ln.strip()]
        if len(nonblank) < min_section_lines:
            dropped += len(lines)
            continue
        if max_section_lines is not None and len(nonblank) > max_section_lines:
            dropped += len(lines)
            continue
        sec_out: list[str] = []
        for raw in lines:
            line = re.sub(r"\s+", " ", raw).strip()
            if not line:
                if sec_out and sec_out[-1] != "":
                    sec_out.append("")
                continue
            if not universal_line_ok(line, max_words=max_words):
                dropped += 1
                continue
            sec_out.append(line)
        while sec_out and sec_out[-1] == "":
            sec_out.pop()
        if sec_out:
            out.extend(sec_out)
            out.append("")
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out).strip() + "\n", dropped


def main() -> None:
    INDIAN_DIR.mkdir(exist_ok=True)
    WESTERN_DIR.mkdir(exist_ok=True)

    notes: list[str] = []
    manifest: list[dict] = []

    notes.append("STEP 1 - Downloads")
    notes.append("Group A requested IDs vs substitutions:")
    for bid, name in REQUESTED_GROUP_A.items():
        notes.append(f"- requested {bid} -> {name}")
    notes.append("")
    notes.append("Applied substitutions:")
    for fname, info in GROUP_A_SUBSTITUTIONS.items():
        if "from" in info:
            notes.append(f"- {fname}: {info['from']} -> {info['id']} ({info['reason']})")
        else:
            notes.append(f"- {fname}: {info['id']} ({info['reason']})")
    notes.append("")

    # Group A downloads
    group_a_files: dict[str, Path] = {}
    for fname, info in GROUP_A_SUBSTITUTIONS.items():
        out = INDIAN_DIR / fname
        try:
            meta = download_book(info["id"], out)
            manifest.append(
                {
                    "group": "A",
                    "file": str(out),
                    "status": "downloaded",
                    "requested_from": info.get("from", info["id"]),
                    **meta,
                }
            )
            group_a_files[fname] = out
        except Exception as e:
            manifest.append(
                {
                    "group": "A",
                    "file": str(out),
                    "status": "failed",
                    "requested_from": info.get("from", info["id"]),
                    "id": info["id"],
                    "error": str(e),
                }
            )

    # Unresolved Group A/B requested books
    notes.append("Unresolved requested Indian sources (no suitable Gutendex poetry record found):")
    for item in UNRESOLVED_REQUESTED:
        notes.append(f"- {item}")
    notes.append("")

    # Group B downloads
    notes.append("Group B query-based selections:")
    group_b_files: dict[str, Path] = {}
    for fname, info in GROUP_B_SELECTED.items():
        out = INDIAN_DIR / fname
        try:
            meta = download_book(info["id"], out)
            manifest.append({"group": "B", "file": str(out), "status": "downloaded", **meta, "query": info["query"]})
            group_b_files[fname] = out
            notes.append(f"- {fname} <- id {info['id']} (query: {info['query']})")
        except Exception as e:
            manifest.append(
                {"group": "B", "file": str(out), "status": "failed", "id": info["id"], "query": info["query"], "error": str(e)}
            )
            notes.append(f"- FAILED {fname} id {info['id']}: {e}")
    notes.append("")

    # Group C downloads
    notes.append("Group C western selections:")
    group_c_files: dict[str, Path] = {}
    for fname, info in GROUP_C_SELECTED.items():
        out = WESTERN_DIR / fname
        try:
            meta = download_book(info["id"], out)
            manifest.append({"group": "C", "file": str(out), "status": "downloaded", **meta, "query": info["query"]})
            group_c_files[fname] = out
            notes.append(f"- {fname} <- id {info['id']} (query: {info['query']})")
        except Exception as e:
            manifest.append(
                {"group": "C", "file": str(out), "status": "failed", "id": info["id"], "query": info["query"], "error": str(e)}
            )
            notes.append(f"- FAILED {fname} id {info['id']}: {e}")
    notes.append("")

    DOWNLOAD_LOG_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Step 2 - poems_indian_raw.txt
    notes.append("STEP 2 - Build poems_indian_raw.txt")
    indian_parts: list[str] = []
    source_chars: dict[str, int] = defaultdict(int)

    indian_processors = [
        ("Tagore", group_a_files.get("tagore_gitanjali.txt"), "tagore"),
        ("Tagore", group_a_files.get("tagore_fruit_gathering.txt"), "tagore"),
        ("Tagore", group_a_files.get("tagore_the_gardener.txt"), "tagore"),
        ("Tagore", group_a_files.get("tagore_stray_birds.txt"), "tagore"),
        ("Naidu", group_a_files.get("naidu_golden_threshold.txt"), "simple16"),
        ("Toru Dutt", group_a_files.get("toru_dutt_ancient_ballads.txt"), "simple16"),
        ("Manmohan Ghose", group_b_files.get("manmohan_ghose_primavera.txt"), "sectioned16"),
    ]

    for poet, path, mode in indian_processors:
        if not path or not path.exists():
            continue
        txt = path.read_text(encoding="utf-8", errors="ignore")
        if mode == "tagore":
            cleaned, _ = process_tagore(txt, max_words=22)
        elif mode == "simple16":
            cleaned, _ = process_simple(txt, max_words=16)
        elif mode == "sectioned16":
            cleaned, _ = process_sectioned(txt, max_words=16, max_section_lines=120, skip_title_terms=("act", "scene", "play", "drama"))
        else:
            continue
        cleaned = cleaned.strip()
        if cleaned:
            indian_parts.append(cleaned)
            source_chars[poet] += len(cleaned)

    indian_raw = "\n\n".join(indian_parts).strip() + "\n"
    (ROOT / "poems_indian_raw.txt").write_text(indian_raw, encoding="utf-8")
    notes.append(f"- poems_indian_raw.txt chars: {len(indian_raw):,}")
    notes.append("")

    # Step 3 - poems_western_extra_raw.txt
    notes.append("STEP 3 - Build poems_western_extra_raw.txt")
    western_parts: list[str] = []

    western_processors = [
        ("Rossetti", group_c_files.get("rossetti_poems_19188.txt"), "rossetti"),
        ("Rossetti", group_c_files.get("rossetti_goblin_market_16950.txt"), "rossetti"),
        ("Burns", group_c_files.get("burns_complete_18500.txt"), "burns"),
        ("Burns", group_c_files.get("burns_poems_songs_1279.txt"), "burns"),
        ("Burns", group_c_files.get("burns_songs_lyrics_75462.txt"), "burns"),
        ("John Clare", group_c_files.get("clare_poems_52601.txt"), "clare"),
        ("John Clare", group_c_files.get("clare_selected_v1_22443.txt"), "clare"),
        ("John Clare", group_c_files.get("clare_poems_ms_8672.txt"), "clare"),
        ("John Clare", group_c_files.get("clare_life_remains_9156.txt"), "clare"),
        ("E. B. Browning", group_c_files.get("ebb_sonnets_2002.txt"), "ebb"),
        ("Hardy", group_c_files.get("hardy_wessex_3167.txt"), "hardy"),
        ("Hardy", group_c_files.get("hardy_past_present_3168.txt"), "hardy"),
        ("Hardy", group_c_files.get("hardy_wessex_9452.txt"), "hardy"),
        ("Hardy", group_c_files.get("hardy_past_present_9430.txt"), "hardy"),
    ]

    for poet, path, mode in western_processors:
        if not path or not path.exists():
            continue
        txt = path.read_text(encoding="utf-8", errors="ignore")
        if mode == "rossetti":
            cleaned, _ = process_sectioned(
                txt,
                max_words=14,
                max_section_lines=60,
                min_section_lines=2,
                skip_title_terms=("goblin market", "act ", "scene ", "play", "drama"),
            )
        elif mode == "burns":
            cleaned, _ = process_sectioned(
                txt,
                max_words=14,
                max_section_lines=120,
                min_section_lines=2,
                skip_title_terms=("appendix", "index", "notes", "correspondence", "letter", "epistle"),
            )
        elif mode == "clare":
            cleaned, _ = process_sectioned(
                txt,
                max_words=14,
                max_section_lines=120,
                min_section_lines=2,
                skip_title_terms=("notes", "preface", "appendix"),
            )
        elif mode == "ebb":
            cleaned, _ = process_sectioned(
                txt,
                max_words=14,
                max_section_lines=20,
                min_section_lines=2,
                skip_title_terms=("notes", "preface"),
            )
        elif mode == "hardy":
            cleaned, _ = process_sectioned(
                txt,
                max_words=14,
                max_section_lines=90,
                min_section_lines=2,
                skip_title_terms=("notes", "preface", "appendix"),
            )
        else:
            continue

        cleaned = cleaned.strip()
        if cleaned:
            western_parts.append(cleaned)
            source_chars[poet] += len(cleaned)

    western_raw = "\n\n".join(western_parts).strip() + "\n"
    (ROOT / "poems_western_extra_raw.txt").write_text(western_raw, encoding="utf-8")
    notes.append(f"- poems_western_extra_raw.txt chars: {len(western_raw):,}")
    notes.append("")

    # Step 4 combine
    existing = (ROOT / "poems_pure_lyric_clean.txt").read_text(encoding="utf-8")
    combined = existing.strip() + "\n\n" + indian_raw.strip() + "\n\n" + western_raw.strip() + "\n"
    (ROOT / "poems_combined_v3.txt").write_text(combined, encoding="utf-8")
    notes.append("STEP 4 - Combine")
    notes.append(f"- Existing : {len(existing):,} chars")
    notes.append(f"- Indian   : {len(indian_raw):,} chars")
    notes.append(f"- Western+ : {len(western_raw):,} chars")
    notes.append(f"- Combined : {len(combined):,} chars")
    notes.append("")

    NOTES_PATH.write_text("\n".join(notes).strip() + "\n", encoding="utf-8")
    BREAKDOWN_PATH.write_text(json.dumps(source_chars, indent=2), encoding="utf-8")

    print(f"Wrote {NOTES_PATH}")
    print(f"Wrote {DOWNLOAD_LOG_PATH}")
    print(f"Wrote {BREAKDOWN_PATH}")
    print(f"Wrote poems_indian_raw.txt ({len(indian_raw):,} chars)")
    print(f"Wrote poems_western_extra_raw.txt ({len(western_raw):,} chars)")
    print(f"Wrote poems_combined_v3.txt ({len(combined):,} chars)")


if __name__ == "__main__":
    main()
