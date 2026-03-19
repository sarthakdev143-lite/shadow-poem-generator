# SHADOW — RESCUE CYCLE BRIEFING
# Read this fully. Execute steps in exact order. Do not skip ahead.
# Do not start the next step until the current one is complete and logged.

---

## CONTEXT
The nano transformer trained for ~16 hours total but did not clearly beat LSTM in the
quality gate. Root cause analysis says this is a DATA problem, not an architecture problem.
The comparison was also unfair (retries=6 vs the proper retries=30).

This rescue cycle fixes both issues in the shortest possible time.

---

## STEP 1 — FAIR DECODE GATE (30 mins, do this first)
Before any retraining, run a fair comparison with proper decode settings.
The previous quality gate used retries=6/candidates=6 which is too low for the transformer.

Run this:
```powershell
.\venv\Scripts\python.exe shadow_transformer.py --mode generate_poems --checkpoint shadow_nano.pt --prompt "In the quiet night" --prompt "I miss her" --prompt "Death be not proud" --poem-count 3 --lines-per-poem 4 --line-retries 30 --candidates 12 --temperature 0.74 --samples-log fair_decode_nano.txt
```

Then run LSTM with same settings:
```powershell
.\venv\Scripts\python.exe shadow.py --mode generate_poems --checkpoint poem_word_lstm_a1.pt --prompt "In the quiet night" --prompt "I miss her" --prompt "Death be not proud" --poem-count 3 --lines-per-poem 4 --line-retries 30 --poem-candidates 12 --temperature 0.74 --samples-log fair_decode_lstm.txt
```

Score both outputs using the same rubric as before (coherence, repetition, tone, grammar).
Save verdict to: fair_decode_verdict.txt

IF nano clearly wins (2+ points ahead on average) → SKIP to STEP 4 (server.py integration).
IF still tied or LSTM wins → continue to STEP 2.

---

## STEP 2 — BUILD PURE LYRIC CORPUS
This is the most important step. The transformer memorizes narrative context deeply.
Contamination hurts it FAR more than it hurts the LSTM.

### KEEP these sources (pure lyrical poetry only):
- Shakespeare's Sonnets (all 154) — pure lyric ✅
- Blake's Songs of Innocence and Experience — pure lyric ✅
- Dickinson's Complete Poems — pure lyric ✅
- Keats: Odes ONLY (Ode to a Nightingale, Ode on a Grecian Urn, To Autumn, etc.)
  REMOVE: Endymion, Lamia, Isabella, The Eve of St Agnes (all narrative)
- Shelley: Short lyrics ONLY (Love's Philosophy, Ozymandias, Music When Soft Voices Die, etc.)
  REMOVE: Queen Mab, Prometheus Unbound, The Revolt of Islam (all epic/narrative)
- Yeats: Short poems ONLY (The Lake Isle of Innisfree, When You Are Old, etc.)
  REMOVE: longer narrative or play-format poems
- REMOVE Tennyson entirely (Arthurian narrative dominates)
- REMOVE Rossetti entirely (too narrative/dialogue-heavy)

### Filtering rules (stricter than before):
- Drop ANY line containing: "said", "answered", "replied", "cried", "spake", "quoth"
  (these indicate dialogue/narrative, not lyric)
- Drop ANY line containing character names: "lancelot", "arthur", "guinevere", "merlin",
  "endymion", "prometheus", "satan", any proper name that isn't a place or deity
- Drop lines with more than 16 words (too narrative/prose-like)
- Drop lines with numbers or years
- Drop lines starting with "and then", "but when", "so that", "for when" 
  (narrative connectors, not lyric)
- Keep the existing clean_poem_corpus() filters from shadow.py

### Target size: 1.8M - 2.5M chars of PURE lyric poetry
### Save to: poems_pure_lyric.txt

Run through the cleaner:
```powershell
.\venv\Scripts\python.exe shadow.py --mode clean_corpus --clean-input poems_pure_lyric.txt --clean-output poems_pure_lyric_clean.txt --clean-min-words 4 --clean-max-words 14
```

Then manually inspect 30 random lines from poems_pure_lyric_clean.txt.
Every single line should look like a line of lyric poetry — no storytelling, no dialogue.
If you see narrative lines, tighten the filters and repeat.

---

## STEP 3 — FINE-TUNE (not full retrain)
Do NOT retrain from scratch. Fine-tune the existing shadow_nano.pt checkpoint.
This takes 2-4 hours instead of 8 hours.

```powershell
.\venv\Scripts\python.exe shadow_transformer.py --mode train --resume --checkpoint shadow_nano.pt --text-path poems_pure_lyric_clean.txt --epochs 12 --steps-per-epoch 200 --batch-size 32 --lr 1e-4 --grad-clip 1.0 --warmup-steps 100 --patience 6 --val-split 0.03 --samples-log finetune_log.txt
```

Key settings explained:
- --resume: loads existing weights instead of starting from scratch
- --lr 1e-4: lower learning rate for fine-tuning (don't overwrite what it learned)
- --epochs 12: short run, just refining
- --steps-per-epoch 200: faster per epoch
- --patience 6: early stop if not improving

Save checkpoint to: shadow_nano_ft.pt (rename after training completes)
```powershell
Copy-Item shadow_nano.pt shadow_nano_ft.pt
```

---

## STEP 4 — FINAL QUALITY GATE
Run full decode comparison between fine-tuned nano and LSTM:

```powershell
.\venv\Scripts\python.exe shadow_transformer.py --mode generate_poems --checkpoint shadow_nano_ft.pt --prompt "In the quiet night" --prompt "I miss her" --prompt "Death be not proud" --prompt "When she is gone" --poem-count 3 --lines-per-poem 4 --line-retries 30 --candidates 12 --temperature 0.74 --samples-log final_nano_samples.txt
```

Score using rubric. If nano wins → proceed to STEP 5.
If still losing → report with logs. Do NOT retrain again. The architecture has done its job.

---

## STEP 5 — INTEGRATE server.py
Replace the entire load_models() function in server.py with this:

```python
def load_models():
    global MODEL, MODEL_TYPE, STOI, ITOS, KNOWN_WORDS

    # Try nano transformer first
    for ckpt in ["shadow_nano_ft.pt", "shadow_nano.pt"]:
        if Path(ckpt).exists():
            try:
                from shadow_transformer import load_checkpoint, generate_text, generate_poems
                MODEL, STOI, ITOS, KNOWN_WORDS, _ = load_checkpoint(ckpt, DEVICE)
                MODEL.eval()
                MODEL_TYPE = "nano"
                # Store references to the functions
                import shadow_transformer as _st
                globals()["_generate_text"] = _st.generate_text
                globals()["_generate_poems"] = _st.generate_poems
                print(f"✅ Nano Transformer loaded from {ckpt}")
                return
            except Exception as e:
                print(f"Nano load failed: {e}")

    # Fall back to LSTM
    for ckpt in ["poem_word_lstm_a1.pt", "poem_word_lstm.pt"]:
        if Path(ckpt).exists():
            try:
                from shadow import load_checkpoint, generate_text, generate_poems
                MODEL, STOI, ITOS, KNOWN_WORDS, _ = load_checkpoint(ckpt, DEVICE)
                MODEL.eval()
                MODEL_TYPE = "lstm"
                import shadow as _sh
                globals()["_generate_text"] = _sh.generate_text
                globals()["_generate_poems"] = _sh.generate_poems
                print(f"✅ LSTM loaded from {ckpt}")
                return
            except Exception as e:
                print(f"LSTM load failed: {e}")

    print("❌ No model found!")
    sys.exit(1)
```

Also update the /generate and /generate_poems endpoints to use _generate_text and
_generate_poems globals instead of inline imports.

Remove ALL GPT-2 code from server.py entirely.

Remove the transformers library from requirements.txt.

---

## STEP 6 — VALIDATE LOCALLY
```powershell
.\venv\Scripts\python.exe server.py
```

Check:
1. Terminal says "✅ Nano Transformer loaded from shadow_nano_ft.pt"
2. Open http://localhost:8000 — UI loads
3. Generate a poem — works correctly
4. Check http://localhost:8000/model_info returns {"model_type": "nano"}
5. Rename shadow_nano_ft.pt temporarily, restart server, verify LSTM fallback loads
6. Rename it back

---

## STEP 7 — PUSH TO GITHUB AND HF SPACES

GitHub (NO .pt files):
```powershell
git add shadow_transformer.py server.py index.html app.py requirements.txt README.md poems_pure_lyric_clean.txt
git add fair_decode_verdict.txt quality_gate_report.md
git commit -m "Add nano transformer, pure lyric corpus, updated server"
git push origin main
```

HF Spaces (WITH model checkpoint):
```powershell
git checkout -b hf-deploy
git add -f shadow_nano_ft.pt
git add shadow_transformer.py server.py index.html requirements.txt README.md
git commit -m "Deploy nano transformer to HF Spaces"
git push hf hf-deploy:main --force
git checkout main
git branch -d hf-deploy
```

Update README.md to mention:
- Now powered by nano transformer (from scratch, ~2.2M params)
- Trained on pure lyric poetry corpus
- Built entirely without borrowed weights

---

## ABSOLUTE RULES
- Do NOT increase model size or change architecture
- Do NOT do a third full retrain — fine-tune only
- Do NOT push .pt files to GitHub origin
- Do NOT skip the fair decode gate in Step 1
- Do NOT start Step 3 without inspecting corpus manually first
- If anything is unclear, STOP and report — do not guess

---
Last updated: 2026-02-28 03:10 AM
Status: Rescue cycle initiated after quality gate failure
Root cause: Unfair comparison + narrative corpus contamination
