# Claude Code — Bootstrap Prompt

Paste the block below into Claude Code (VS Code extension) as your **first message**
in an empty project folder.

**Before you paste:** drop `CLAUDE.md` at the repo root and
`PROJECT.md` + `SPELLING-SPEC.md` into `docs/`. Claude Code reads `CLAUDE.md`
automatically, so the constraints are already in context when it starts.

---

## The prompt

````text
Bootstrap this repository. Read `CLAUDE.md` and `docs/PROJECT.md` first — they
contain non-negotiable constraints. Do not deviate from them; if a constraint
seems wrong, tell me before you build anything.

This is Phase 0 scaffolding. We are NOT training a model yet. The goal is a
clean, tested, runnable skeleton where every module has a real interface and a
failing-or-passing test, so that Phase 0 verification work can start immediately.

## What to build

### 1. Project configuration

Dependencies are managed with **uv**. Never `pip install` or `uv pip install`;
use `uv add` / `uv add --dev`, and commit `uv.lock`.

- `pyproject.toml` — hatchling backend, project name `roman-urdu-captions`,
  `requires-python = "==3.12.*"`. Dependencies:
  - `[project].dependencies`: `torch`, `qwen-asr`, `transformers`, `pydantic`,
    `pyyaml`, `typer`, `fastapi`, `uvicorn`, `python-multipart`
  - `[project.optional-dependencies].align`: `torchaudio`, `pyannote.audio`, `demucs`
  - `[project.optional-dependencies].cuda`: `flash-attn; sys_platform == 'linux'`
  - `[dependency-groups].dev` (PEP 735, **not** an extra — this is what
    `uv add --dev` writes to): `ruff`, `mypy`, `pytest`, `pytest-cov`,
    `pre-commit`, `types-pyyaml`
- `[tool.uv]` — the two things that otherwise break:

  ```toml
  [tool.uv]
  # flash-attn's setup.py imports torch to read the CUDA version, so it cannot
  # build in an isolated environment. Setup syncs twice: once without the cuda
  # extra so torch exists, then again so flash-attn builds against it.
  no-build-isolation-package = ["flash-attn"]

  # On a CUDA machine, pin torch to the matching index (match `nvidia-smi`):
  # [tool.uv.sources]
  # torch = [{ index = "pytorch-cu124" }]
  # torchaudio = [{ index = "pytorch-cu124" }]
  #
  # [[tool.uv.index]]
  # name = "pytorch-cu124"
  # url = "https://download.pytorch.org/whl/cu124"
  # explicit = true
  ```

- `.python-version` containing `3.12` — uv provisions the interpreter from this.
  Make sure `.gitignore` does **not** ignore it, or `uv.lock`.
- Configure ruff (line length 100, select E/F/I/N/UP/B/SIM) and mypy
  (`strict = true` for `src`, relaxed for `tests`) inside `pyproject.toml`.
- `.gitignore` — Python defaults plus `data/raw/`, `data/work/`, `.venv/`,
  `*.wav`, `*.mp4`, `out/`, model checkpoints.
- `.pre-commit-config.yaml` — ruff, ruff-format, mypy, trailing-whitespace.
- `.editorconfig`.
- Run `uv lock` and commit `uv.lock`.

### 2. Package skeleton
Create these with real type-hinted signatures, docstrings explaining WHY, and
`NotImplementedError` bodies where the logic is Phase 1+ work:

```
src/
  __init__.py
  config.py              # pydantic settings loaded from configs/*.yaml
  types.py               # Segment, Token, TimedToken, Caption dataclasses
  ingest/
    extract.py           # extract_audio(video: Path) -> Path  (ffmpeg, 16kHz mono s16)
  preprocess/
    separate.py          # remove_music(audio: Path) -> Path   (demucs)
    diarize.py           # diarize(audio: Path) -> list[Segment]
    segment.py           # enforce_bounds(segs, min_s=2.0, max_s=35.0)
  labeling/
    romanize.py          # urdu_to_roman(text: str) -> str
    normalize.py         # normalize(tokens: list[str]) -> list[str]   ← PURE, DETERMINISTIC
    lexicon.py           # Lexicon loader: canonical.tsv, english.txt, ambiguous.tsv
  inference/
    asr.py               # transcribe(audio, model_id) -> list[Segment]
    align.py             # align(audio, text, strategy) -> list[TimedToken]
  subtitle/
    layout.py            # line breaking, reading-speed limits per SPELLING-SPEC §6
    writer.py            # to_srt(captions) -> str ; to_vtt(captions) -> str
  eval/
    metrics.py           # cer(), sn_wer(), normalized_wer(), english_preservation()
    score.py             # CLI entrypoint
  api/
    app.py               # FastAPI: POST /transcribe, GET /jobs/{id}
    pipeline.py          # orchestrates ingest -> ... -> subtitle
```

### 3. The normalizer is the priority
`src/labeling/normalize.py` is the only module I want FULLY implemented now,
because everything else depends on its contract. Implement it per
`docs/SPELLING-SPEC.md` §9:

- Pure function, no I/O inside the transform
- Lexicons loaded once at construction
- Lookup order: english → acronym → canonical → variant-map → rule fallback
- Every unknown token logged to a growable set for later lexicon expansion
- English words pass through UNTOUCHED (SPELLING-SPEC §5 — this is a P0 rule)

Seed `data/lexicon/canonical.tsv` from the table in SPELLING-SPEC §8.
Seed `data/lexicon/english.txt` with the loanwords in §5.2 plus a few hundred
common English words.

### 4. Tests
- `tests/test_spelling_spec.py` — one test per row of SPELLING-SPEC §11
- `tests/test_normalize.py` — determinism (same input × 1000 → identical output),
  English passthrough, unknown-token logging
- `tests/test_subtitle.py` — golden-file SRT/VTT output
- `tests/test_align_offsets.py` — chunk reassembly for audio > 5 minutes
  (aligner caps at 5 min; offsets must be correct). Mark `xfail` for now.
- `tests/conftest.py` — fixtures for sample segments and a tiny fake audio file

Tests for unimplemented modules should exist and be marked
`@pytest.mark.skip(reason="Phase 1")` — I want the test names visible as a TODO list.

### 5. Configs
- `configs/phase1.yaml` — training config for the romanized read-speech phase
- `configs/phase2.yaml` — code-switched phase
- Both following Srota's hyperparameters from PROJECT.md §3.2:
  full fine-tune (NOT LoRA), AdamW, lr 2e-5 linear, warmup_ratio 0.02,
  effective batch 32, bf16, 2 epochs, language-agnostic decoding prefix.
- `configs/inference.yaml` — model ids, chunk size, aligner strategy

### 6. Scripts
- `scripts/build_lexicon.py` — read Roman-Urdu-Parl, emit frequency-ranked TSV
- `scripts/verify_data.py` — Phase 0 gate: check UrduSpeech downloaded, integrity,
  print hours per subset, warn if license file missing
- `scripts/test_aligner.py` — run Qwen3-ForcedAligner on Roman Urdu text with
  `language="English"` and report alignment error. This resolves PROJECT.md §4.4
  and is the highest-priority experiment in the project.

### 7. Runner + docs
- Make `run.sh` executable; verify `run.bat` dispatch matches it command-for-command
- `README.md` — quickstart, architecture diagram, links to docs/
- `docs/DECISIONS.md` — ADR log seeded with the four decisions in SPELLING-SPEC §10
- `.github/workflows/ci.yml` — lint, typecheck, test on push

## Rules for this task

- **Do not** implement romanization logic yet — that is Phase 1 and needs the
  frequency lexicon first. Stub it.
- **Do not** add a dependency without noting its license in `docs/DECISIONS.md`.
  Apache-2.0 / MIT preferred; copyleft needs my sign-off.
- **Do not** write anything that routes text through Urdu or Devanagari script.
- Every file you create gets a docstring saying what it is for.
- Commit in logical chunks with conventional commit messages, not one giant commit.
- When done, run `./run.sh check` and show me the output.

## Finish by telling me

1. Anything in CLAUDE.md or PROJECT.md you think is wrong or underspecified
2. The exact command to run the aligner experiment (`scripts/test_aligner.py`)
3. What you'd want decided before Phase 1 starts
````

---

## After it finishes

Run these in order — both are Phase 0 gates from `PROJECT.md` §8:

```bash
./run.sh doctor                           # confirm uv, ffmpeg, GPU, lexicons
uv run python -m scripts.verify_data      # gate on R1: is UrduSpeech actually usable?
uv run python -m scripts.test_aligner     # gate on §4.4: does alignment work on Roman Urdu?
```

If `verify_data` fails, stop and re-plan — the whole roadmap assumes that corpus.

---

## Follow-up prompts (in order)

Use these as separate Claude Code sessions, one per unit of work:

**2 — Lexicon**
> Implement `scripts/build_lexicon.py`. Read Roman-Urdu-Parl, tokenize, count, emit
> the top 5000 Roman Urdu tokens ranked by frequency to `docs/lexicon-frequency.tsv`.
> Then produce a report comparing those frequencies against every ⚠️ row in
> `docs/SPELLING-SPEC.md` §8, and recommend a resolution for each.

**3 — Normalizer hardening**
> Using `docs/lexicon-frequency.tsv`, expand `data/lexicon/canonical.tsv` and the
> variant map. Add a test per new rule. Report unknown-token rate on a sample.

**4 — Ingest + preprocess**
> Implement `src/ingest/extract.py` and `src/preprocess/*`. ffmpeg to 16kHz mono
> s16 PCM, Demucs separation, pyannote diarization, segment bounds per PROJECT.md
> §4.1. Integration test on a 60-second sample.

**5 — Baseline evaluation**
> Implement `src/eval/metrics.py` and `score.py`. Run stock Qwen3-ASR-1.7B against
> `data/eval/`, then produce the error breakdown from PROJECT.md §6.3 —
> classify every error as acoustic / orthographic / code-switch / timing and give
> me the percentages.

Prompt 5 is the one that actually answers the open question the whole project
hinges on. Get there fast.
