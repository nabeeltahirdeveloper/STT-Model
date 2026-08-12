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

Dependencies and the Python version are both managed by uv. There is no
virtualenv to activate: `uv run` resolves the environment per invocation.

```bash
# environment — uv installs Python 3.12 itself (.python-version)
uv sync --all-extras --all-groups        # everything, from uv.lock
uv sync                                  # runtime deps only, fastest loop
./run.sh setup                           # first time: also scaffolds + git hooks

# quality gates — run before every commit
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run mypy src/
uv run pytest -q
./run.sh check                           # all three, in order

# pipeline
uv run python -m src.ingest.extract --input video.mp4 --out data/work/
uv run python -m src.labeling.romanize --input data/raw/ --out data/labels/
uv run python -m src.training.finetune --config configs/phase1.yaml
uv run python -m src.eval.score --pred out.txt --ref data/eval/ref.txt  # CER + SN-WER

# serving
uv run qwen-asr-serve Qwen/Qwen3-ASR-1.7B --gpu-memory-utilization 0.8 --port 8000
```

flash-attn is the `cuda` extra, marker-gated to Linux. It cannot build under
uv's default build isolation (its setup.py imports torch), so `./run.sh setup`
syncs twice — torch first, then flash-attn against it. Set `MAX_JOBS=4` if the
machine has under 96 GB RAM.

Keep this section accurate. If you add or rename a command, update it in the same commit.

---

## Code conventions

- **Dependencies are managed with uv.** Use `uv add` / `uv add --dev`, never
  `pip install` or `uv pip install`. Commit `uv.lock` with any dependency change.
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

**Phase 0 is complete (ADR-008, ADR-010). Nothing is trained yet.**

| | |
|---|---|
| Eval set | `data/eval/reference.txt` — 269 utterances, 35.4 min, 12 categories, human-corrected |
| Baseline CER | **27.9%** after romanization (75.8% raw — stock output is Devanagari). ADR-011 respelling moved this from 26.7%; the model did not change |
| English preserved | **41.9%**, against a > 90% target |
| Error split | 58.1% acoustic · 39.8% orthographic · 2.1% code-switch |
| Timing | §4.4 **strategy 1** — direct alignment works, 46 ms median. No Urdu-script bridge |

**Phase 1 is complete.** `SPELLING-SPEC.md` is frozen at 1.0.0 (ADR-015) and
`data/labels/labels.jsonl` holds 29,749 labels covering 90 h.

| | |
|---|---|
| Romanization | dictionary lookup, not neural (ADR-014). 83% of Roman-Urdu-Parl is misaligned; the dictionary is built from the rest |
| Label quality | ~3.5% of words are the wrong word, ~2% unknown. Inherited from the corpus — training cannot fix labels |
| First training run | LoRA on MPS shifted output from 5.7% to 48.3% Latin. The labels teach Roman (ADR-016) |

**Phase 2 — training.** Full fine-tuning on a rented GPU is the production path
and `src/training/finetune.py` is still a stub. `scripts/train.py` is the local
LoRA experiment and is **not** a substitute: constraint 6's A/B cannot run on
16 GB, so its numbers validate the pipeline, not the method.

Two things that gate quality more than model size:

- **38% of errors are orthographic.** They are fixed in the romanizer and
  lexicon, not with more training data. Measure before assuming otherwise.
- **The eval set is benchmark audio, not target content** (R6, ADR-008). Treat
  26.7% as optimistic and build a real-content set before quoting it externally.
