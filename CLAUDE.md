# CLAUDE.md

Operating brief for Claude Code on this repository.
Full specification: `docs/PROJECT.md`. Read it before any architectural work.

---

## What this project is

A service that generates **Roman Urdu** subtitles for uploaded video, handling
Urdu–English code-switching. Pipeline: video → audio → fine-tuned Qwen3-ASR →
spelling normalization → forced alignment → SRT/VTT.

**The output script is Latin, not Perso-Arabic.** This is deliberate and load-bearing.

---

## Non-negotiable constraints

Violating any of these breaks the product. If a task seems to require it, stop and ask.

1. **Never route the text path through Urdu (Nastaliq) or Devanagari script.**
   The two-stage `audio → Urdu script → Roman` approach was tried and failed — it
   mangles English words. Urdu script is permitted *only* as a timing bridge
   (`PROJECT.md` §4.4, option 2), never for producing customer-facing text.

2. **English words stay in English orthography.** `meeting` not `mitting`,
   `problem` not `prablam`. Any code path that phonetically re-spells an English
   word is a bug.

3. **One word → one spelling, always.** All Roman Urdu output passes through the
   canonical normalizer (`src/labeling/normalize.py`) driven by
   `docs/SPELLING-SPEC.md`. Never emit unnormalized model output to a user.

4. **Never report raw WER as a headline metric.** Roman Urdu has no standard
   orthography, so WER punishes correct-but-variant spellings. Use CER and SN-WER.
   Raw WER may appear in tables only alongside CER.

5. **The eval set is never trained on.** `data/eval/` is sacred. No exceptions,
   not even "just for a quick sanity check."

6. **Do not use LoRA without an A/B against full fine-tuning.** Two independent
   studies found vanilla FFT substantially outperforming LoRA on this problem class.
   Full fine-tuning is the default.

7. **Urdu is not an officially supported Qwen3-ASR language** (Hindi is; the
   forced aligner supports neither). Do not force `language="Urdu"` — use the
   language-agnostic decoding prefix. Do not trust built-in language ID.

---

## Architecture at a glance

```
ingest → preprocess → ASR → normalize → align → subtitle → serve
```

| Stage | Module | Key tool |
|---|---|---|
| Ingest | `src/ingest/` | ffmpeg → 16 kHz mono 16-bit PCM |
| Preprocess | `src/preprocess/` | Demucs, pyannote 3.1; drop <2s, split >35s |
| ASR | `src/inference/` | fine-tuned Qwen3-ASR (0.6B dev / 1.7B prod) |
| Normalize | `src/labeling/` | canonical spelling map |
| Align | `src/inference/` | Qwen3-ForcedAligner (≤5 min chunks) |
| Subtitle | `src/subtitle/` | line breaking, reading speed, SRT/VTT |
| Serve | `src/api/` | vLLM + FastAPI |

---

## Repository layout

```
CLAUDE.md              # this file
docs/
  PROJECT.md           # full spec — scope, architecture, roadmap
  SPELLING-SPEC.md     # canonical Roman Urdu orthography (frozen before Phase 2)
  DECISIONS.md         # ADR log — append, never rewrite history
data/
  raw/                 # corpora (gitignored, large)
  labels/              # romanized + normalized transcripts
  eval/                # held-out benchmark — NEVER train on this
src/
  ingest/ preprocess/ labeling/ training/ inference/ subtitle/ api/
scripts/               # one-off CLIs
tests/
notebooks/             # Colab experiments — not production code
```

---

## Commands

```bash
# environment
conda create -n rux python=3.12 -y && conda activate rux
pip install -U qwen-asr[vllm]
pip install -U flash-attn --no-build-isolation   # MAX_JOBS=4 if <96GB RAM

# quality gates — run before every commit
ruff check src/ tests/
ruff format src/ tests/
mypy src/
pytest -q

# pipeline
python -m src.ingest.extract --input video.mp4 --out data/work/
python -m src.labeling.romanize --in data/raw/ --out data/labels/
python -m src.training.finetune --config configs/phase1.yaml
python -m src.eval.score --pred out.txt --ref data/eval/ref.txt   # CER + SN-WER

# serving
qwen-asr-serve Qwen/Qwen3-ASR-1.7B --gpu-memory-utilization 0.8 --port 8000
```

Keep this section accurate. If you add or rename a command, update it in the same commit.

---

## Code conventions

- **Python 3.12**, type hints on every public function
- **ruff** for lint + format; **mypy** for types; **pytest** for tests
- Config in YAML under `configs/`, never hardcoded in scripts
- Every training run logs to the experiment tracker with a config hash
- Model paths and hyperparameters come from config, never inline literals
- Pure functions in `src/labeling/` — normalization must be deterministic and testable
- Docstrings explain *why*, not *what*

---

## Testing expectations

- Every normalizer rule gets a unit test with a real example pair
- Golden-file tests for SRT/VTT output — subtitle formatting regressions are silent
- Alignment offset reassembly needs a test with audio longer than 5 minutes
- Any bug fix starts with a failing test that reproduces it

---

## Working style

- **Read `docs/PROJECT.md` before proposing architecture changes.** Most decisions
  there have reasons that aren't obvious from the code.
- **Measure before optimizing.** The error breakdown (`PROJECT.md` §6.3) drives
  priority. Don't guess whether a problem is acoustic or orthographic — classify it.
- **Small commits, one concern each.** Conventional commit messages.
- **Record real decisions in `docs/DECISIONS.md`** — context, options considered,
  choice, consequence. Append only.
- **Flag when you disagree.** If a requested change conflicts with a constraint
  above or with the spec, say so before implementing rather than after.
- Don't add dependencies without noting the license in `docs/DECISIONS.md`.
  This is a commercial product; Apache-2.0 / MIT preferred, copyleft needs a decision.

---

## Current status

**Phase 0 — verify and baseline.** Nothing is trained yet. Blocking items:

1. Confirm UrduSpeech is downloadable and its license permits commercial use
2. Build the 30–60 min eval set from real target content
3. Baseline stock Qwen3-ASR-1.7B on it
4. Test whether the forced aligner handles Roman Urdu with `language="English"`
5. Produce the acoustic-vs-orthographic error breakdown

Items 1 and 4 gate everything downstream. Do them first.
