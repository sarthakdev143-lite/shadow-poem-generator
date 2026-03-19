# CORPUS EXPANSION — INDIAN POETS (EXPANDED)
# Read fully before executing. Follow steps in exact order.

## GOAL
Expand poems_pure_lyric_clean.txt from ~1.82M chars to 4M+ chars
by adding Indian English-language poetry.

Target: 700k-900k tokens total after expansion.
This fixes Colab overfitting (was 420k tokens, memorized in 4 epochs).

Also fix: filter out "Goringe" and any other unknown/corrupt words found
in colab_test_output.txt — these leaked from corpus.

---

## SOURCES — GROUP A (Project Gutenberg IDs)

Download all to: corpus_indian_sources/
Use gutendex API exactly as before.

```python
books = {
    # TAGORE
    7164:  'tagore_gitanjali.txt',
    6955:  'tagore_fruit_gathering.txt',
    6165:  'tagore_the_gardener.txt',
    6161:  'tagore_stray_birds.txt',
    6162:  'tagore_crescent_moon.txt',
    15819: 'tagore_lovers_gift.txt',
    
    # KABIR
    27473: 'kabir_songs.txt',
    
    # SAROJINI NAIDU
    44071: 'naidu_golden_threshold.txt',
    44960: 'naidu_bird_of_time.txt',

    # TORU DUTT
    7070:  'toru_dutt_ancient_ballads.txt',
    
    # SRI AUROBINDO
    # Search gutendex for "Aurobindo" if ID not found
    
    # MICHAEL MADHUSUDAN DUTT
    # Search gutendex for "Madhusudan"
}
```

For any ID that 404s or returns wrong book:
- Search gutendex.com/books/?search=author_name
- Use the correct ID
- Log substitutions in a note

---

## SOURCES — GROUP B (Direct search)

For these poets, search gutendex first:
```python
queries = [
    'Sarojini Naidu',
    'Toru Dutt',
    'Sri Aurobindo poems',
    'Manmohan Ghose poems',
    'Harindranath Chattopadhyay',
    'Michael Madhusudan Dutt',
]
```

Download ANY poetry collections found. Skip prose works.
Save to corpus_indian_sources/ with descriptive names.

---

## SOURCES — GROUP C (Extra Western lyric to boost size)

Also add these to reach 4M+ chars target:

```python
extra_western = {
    # Christina Rossetti — short lyrics only
    # Search: "Rossetti Goblin Market other poems"
    # ONLY take poems under 40 lines, skip Goblin Market itself
    
    # Robert Burns — songs and short poems
    # Search: "Burns poems songs"
    
    # John Clare — nature lyrics
    # Search: "John Clare poems"
    
    # Elizabeth Barrett Browning — Sonnets from the Portuguese
    # Search: "Sonnets from the Portuguese"
    
    # Thomas Hardy — short poems
    # Search: "Hardy poems"
}
```

Save to corpus_western_extra/ folder.

---

## CLEANING RULES

### Tagore (CRITICAL — prose poetry):
- max_words = 22 (his lines are naturally longer)
- KEEP numbered sections (I, II, III) — each is one poem
- DROP the section number lines themselves (standalone "I" or "XIV")
- DROP Yeats's introduction (first several paragraphs before poem I)
- DROP lines containing: "translator", "bengali", "rabindranath", "calcutta"
- His best lines look like:
  "Where the mind is without fear and the head is held high"
  "Let me not pray to be sheltered from dangers"
  These are LONG but deeply lyrical — KEEP them

### Kabir:
- Short songs, 4-16 words per line
- DROP translator notes between songs
- DROP lines starting with "Note:" or "See:"

### Sarojini Naidu:
- Pure lyric, treat same as Shelley
- max_words = 16
- KEEP Indian imagery: lotus, jasmine, peacock, sitar, monsoon, bazaar
- DROP poem title lines (they're headings, not verse)

### Toru Dutt:
- Victorian + Indian blend, very lyrical
- max_words = 16
- KEEP references to Indian mythology (fine for training)
- DROP footnotes and editorial notes

### Sri Aurobindo / Manmohan Ghose / others:
- Treat same as Western lyric poets
- max_words = 16
- DROP any prose commentary

### Western extras (Burns, Clare, Hardy, Rossetti):
- Same rules as existing Western corpus
- Rossetti: DROP Goblin Market (too narrative/long)
  KEEP: shorter devotional and lyric poems
- Burns: KEEP songs and short poems
  DROP: long narrative poems
- max_words = 14 for Western extras

### Universal rules (apply to ALL sources):
- DROP lines containing: "(", ")", "[", "]", "_"
- DROP lines starting with quotes: " or '
- DROP dialogue words: said, answered, replied, cried, spake, quoth
- DROP lines with digits
- DROP lines where first alpha char is lowercase
- DROP lines ending with: with, to, for, of, and, or, that
- DROP corrupt/unknown words — add to blocklist:
  "goringe" (found in colab output — corpus contamination)

Add "goringe" explicitly to the noise_re pattern:
```python
noise_re = re.compile(
    r'project gutenberg|ebook|copyright|...|goringe',
    re.IGNORECASE
)
```

---

## BUILD STEPS

### Step 1 — Download all sources
Save to corpus_indian_sources/ and corpus_western_extra/
Log any failures.

### Step 2 — Build poems_indian_raw.txt
Apply Indian-specific cleaning rules per source.
Use process_sectioned() for Tagore/Kabir/sectioned works.
Use process_simple() for Naidu/Toru Dutt.

### Step 3 — Build poems_western_extra_raw.txt
Apply Western cleaning rules to Burns, Clare, Hardy, Rossetti extras.

### Step 4 — Combine everything:
```python
from pathlib import Path

existing  = Path('poems_pure_lyric_clean.txt').read_text(encoding='utf-8')
indian    = Path('poems_indian_raw.txt').read_text(encoding='utf-8')
western   = Path('poems_western_extra_raw.txt').read_text(encoding='utf-8')

combined  = existing.strip() + '\n\n' + indian.strip() + '\n\n' + western.strip() + '\n'
Path('poems_combined_v3.txt').write_text(combined, encoding='utf-8')

print(f'Existing : {len(existing):,} chars')
print(f'Indian   : {len(indian):,} chars')
print(f'Western+ : {len(western):,} chars')
print(f'Combined : {len(combined):,} chars')
```

### Step 5 — Final clean pass:
```powershell
.\venv\Scripts\python.exe shadow.py --mode clean_corpus \
    --clean-input poems_combined_v3.txt \
    --clean-output poems_combined_v3_clean.txt \
    --clean-min-words 4 \
    --clean-max-words 22
```

### Step 6 — Inspect 40 random lines:
```python
import random
from pathlib import Path
lines = [ln.strip() for ln in 
         Path('poems_combined_v3_clean.txt').read_text().splitlines() 
         if ln.strip()]
random.seed(42)
for i, ln in enumerate(random.sample(lines, 40), 1):
    print(f"{i:02d}. {ln}")
```

What you want to see:
- Western lyric lines ✅
- Tagore devotional lines ✅
- Naidu/Indian imagery lines ✅
- SHORT lines (4-22 words) ✅
- NO dialogue, NO editorial, NO "goringe" ✅

### Step 7 — Report stats:
Print:
- Total chars in poems_combined_v3_clean.txt
- Total non-blank lines
- Estimated tokens = chars / 5.5
- Source breakdown (how many chars per poet)

Target: 3.5M-5M chars total.

---

## AFTER CORPUS READY — STOP

DO NOT retrain locally.
DO NOT run shadow_transformer.py training.
Just report final stats and 40-line sample.
Colab retrain will be done manually with new corpus.

---

## ABSOLUTE RULES
- Do NOT delete poems_pure_lyric_clean.txt
- Do NOT filter out Indian vocabulary (lotus, monsoon, jasmine etc.)
- Do NOT retrain locally
- "goringe" must be added to noise filter
- Report download failures — do not silently skip
- If total combined chars < 3M, report and list what failed
