# Architecture Decision Record

Append-only. Never rewrite history — supersede instead.

Format: context → options → decision → consequence.

---

## ADR-001 — Reject the two-stage script pipeline

**Date:** 2026-08-06 · **Status:** Accepted

**Context.** The obvious pipeline is `audio → Urdu (Nastaliq) → Roman Urdu`.
It was built and tested. It failed.

**Options.**
1. Two-stage via Perso-Arabic intermediate
2. Two-stage via Devanagari intermediate (better-resourced romanization tooling)
3. End-to-end `audio → Roman Urdu`

**Decision.** Option 3.

**Reasoning.** English words round-trip through Perso-Arabic and come back
mangled — `meeting` → میٹنگ → `mitting`. Measured externally: Whisper-Large-v3
goes 0.289 → 0.532 WER when code-switching is present, specifically because it
transliterates or translates English into Urdu script. Option 2 has the same
defect plus script-selection instability.

In end-to-end Roman, English and Urdu share one alphabet, so there is no script
boundary to destroy the word.

**Consequence.** No public (audio → Roman Urdu) training corpus exists; labels
must be manufactured. Urdu script survives only as a possible *timing* bridge
(PROJECT.md §4.4 option 2), never for text.

---

## ADR-002 — Qwen3-ASR as the base model

**Date:** 2026-08-06 · **Status:** Accepted

**Context.** Need a base checkpoint to fine-tune for Roman Urdu output.

**Options.** Whisper large-v3-turbo · Qwen3-ASR-0.6B/1.7B · Urdu-specific Whisper
fine-tunes (`kingabzpro/whisper-large-v3-urdu`, `ihanif/whisper-medium-urdu`)

**Decision.** Qwen3-ASR — `0.6B` for iteration, `1.7B` for production.

**Reasoning.** Apache-2.0 with explicit commercial use. Proven on the sister
problem: Srota, a Hinglish fine-tune from Qwen3-ASR-0.6B, took conversational WER
24.73% → 15.85% for ~$6.50 of compute, built to fix these exact failure modes.
Ships a matched forced aligner. Whisper has a *documented* failure on code-switched
Urdu. Urdu-script fine-tunes are actively wrong here — their decoders are pushed
toward Nastaliq output.

**Consequence.** ⚠️ Urdu is not among Qwen3-ASR's 30 supported languages (Hindi
is). The forced aligner supports 11 languages, neither Urdu nor Hindi. Mitigations
in PROJECT.md §4.3–§4.4. Use language-agnostic decoding; do not trust built-in LID.

---

## ADR-003 — Full fine-tuning, not LoRA

**Date:** 2026-08-06 · **Status:** Accepted

**Context.** LoRA is the reflexive default for adapting large models cheaply.

**Decision.** Full-parameter fine-tuning is the default. LoRA requires an A/B
against FFT before use.

**Reasoning.** Two independent results point the same way. A Pashto study found
vanilla full fine-tuning beat LoRA (rank 64) by 33.36 percentage points. Srota
used full fine-tuning of ~780M params explicitly with no LoRA and no frozen
layers. At 0.6B scale the compute argument for LoRA is weak — the reference run
cost about $6.50.

**Consequence.** Higher VRAM per run. Negligible cost impact at this scale.

---

## ADR-004 — Spelling consistency is a product artifact

**Date:** 2026-08-06 · **Status:** Accepted

**Context.** Roman Urdu has no standard orthography. Multiple spellings of the
same word are all "correct."

**Decision.** Maintain a canonical spelling specification
(`docs/SPELLING-SPEC.md`), frozen before Phase 2 training, applied to training
labels rather than at inference.

**Reasoning.** In an end-to-end model, label inconsistency is baked into the
weights permanently. Applying normalization at inference cannot fix a model that
learned ambiguity. Two systems with equal word accuracy are judged very
differently on consistency alone, so this is where quality perception is won.

The lexicon does not need to be hand-built: Roman-Urdu-Parl provides 6.37M
sentence pairs and a 42,927-word frequency-ranked Roman vocabulary that already
captures real spelling distribution.

**Consequence.** Spec changes after freeze require a major version bump and a
full retrain. Freeze deliberately, then leave it alone.

---

## ADR-005 — CER and SN-WER over raw WER

**Date:** 2026-08-06 · **Status:** Accepted

**Context.** WER is the standard ASR metric.

**Decision.** CER is primary; SN-WER (script-normalized WER) is secondary. Raw
WER may appear in tables only alongside CER, never as a headline.

**Reasoning.** WER scores `nahin` vs `nahi` as a full error though both are
correct, systematically understating quality for a language with no fixed
orthography. Script-normalized scoring has been shown to reduce inflated Urdu
error rates by 6.4–9.0%.

**Consequence.** Numbers are not directly comparable to published Urdu ASR WER
figures. Always state which metric is being quoted.

---

## ADR-006 — English orthography outranks the variant map

**Date:** 2026-08-06 · **Status:** Accepted

**Context.** `docs/SPELLING-SPEC.md` §8 lists `the` and `they` as rejected
variants of تھے (canonical `thay`), and `main` as a rejected variant of میں
(canonical `mein`). All three are also ordinary English words. A variant map that
contains them will rewrite English text: `the best` becomes `thay best`.

**Options.**
1. Follow §8 literally and accept that some English words get respelled
2. Drop the colliding rows from the variant map
3. Resolve every collision by context at runtime

**Decision.** Option 2 for `the` and `they` — they are absent from
`data/lexicon/canonical.tsv`. Option 3 for `main`, which is declared in
`data/lexicon/ambiguous.tsv` with the Urdu reading as default and a short list of
following nouns (`road`, `gate`, `street`, …) that force the English reading.

**Reasoning.** Respelling an English word is a P0 bug (SPELLING-SPEC §5.1) and
is the specific failure that rejected the two-stage architecture (ADR-001).
A missed variant costs one inconsistent Roman spelling; a wrong English
respelling costs the thing the product is for. Context resolution is only worth
its false-positive risk where the Urdu reading is genuinely frequent, which is
true for `main` (میں = "I") and not for `the`/`they`.

**Consequence.** `data/lexicon/canonical.tsv` deliberately diverges from
SPELLING-SPEC §8 in two rows; `tests/test_lexicon.py` pins the divergence so it
cannot be silently reverted. `Lexicon.load` now refuses to start if a token
appears in both `english.txt` and the Roman Urdu lexicon without a row in
`ambiguous.tsv` — collisions are a load-time error, not a runtime surprise.

Note also that §11 row 7 (`main TV dekh raha tha` → `Main TV dekh raha tha`)
contradicts §8, which makes `mein` canonical. The test for it is `xfail(strict)`
pending the frequency data required by §2.

---

## ADR-007 — uv for dependency management

**Date:** 2026-08-06 · **Status:** Accepted

> Numbered 007, not 006: ADR-006 is already taken by the English-orthography
> decision above, and this log is append-only with stable identifiers.

**Context.** The project was scaffolded with `python -m venv` plus
`pip install -e ".[dev]"`, which produces no lockfile. Two contributors
installing on the same day get different versions of torch, transformers and
pyannote, and neither can reproduce the other's result. For a project whose
central output is a *trained model*, an unreproducible environment means an
unreproducible experiment — the config hash logged with every run (CLAUDE.md >
Code conventions) is only as meaningful as the environment it ran in.

**Options.** pip + venv (status quo) · Poetry · PDM · uv

**Decision.** uv. Dev tooling moves to a PEP 735 `[dependency-groups]` block;
`uv.lock` is committed; `.python-version` pins the interpreter.

**Reasoning.**

- **Lockfile reproducibility.** `uv.lock` is universal — one file resolves
  correctly for macOS laptops and Linux training boxes, so the Mac used for
  Phase 0 normalizer work and the H100 box used for Phase 2 agree on versions.
  Poetry and PDM also lock; pip alone does not.
- **Resolver speed.** Full resolution of 197 packages takes seconds rather than
  minutes. This matters more than it sounds: a slow resolve is a resolve people
  skip, and skipped resolves are how lockfiles drift out of date.
- **uv manages Python itself.** `uv python install 3.12` removes the assumption
  that the right interpreter is already on PATH. This was not hypothetical — the
  machine this migration ran on had only Python 3.14 installed, and the previous
  `run.sh` would have silently proceeded on it after a warning. Poetry and PDM
  manage packages but not interpreters.
- **The upstream stack already recommends it.** The Qwen3-ASR documentation uses
  uv as the environment manager for the vLLM install path, which is the exact
  serving path in PROJECT.md §4.2. Matching upstream's tooling means their
  install instructions work here unmodified.

**Consequence.**

- Contributors now need uv installed. It is a single-command install on every
  platform and replaces both the Python and the pip install, so this is a net
  reduction in setup steps, but it *is* a new hard prerequisite — `./run.sh
  doctor` and `./run.sh setup` both fail fast without it.
- **flash-attn needs a build-isolation escape hatch.** Its `setup.py` imports
  torch to read the CUDA version, so it cannot build in uv's default isolated
  build environment. It is declared as the `cuda` extra, marker-gated to Linux,
  with `no-build-isolation-package = ["flash-attn"]`, and `./run.sh setup` syncs
  twice — once excluding the extra so torch is present, then again so flash-attn
  builds against it. Getting this wrong produces a confusing `ModuleNotFoundError:
  torch` from inside a build backend.
- Torch is resolved from default PyPI. A training box must add the matching
  `[[tool.uv.index]]` for its CUDA version and re-run `uv lock`; the block is
  commented in `pyproject.toml` next to the setting.
- `uv.lock` must be committed with every dependency change, and reviewed like
  code. A lockfile that is edited but not reviewed is worse than no lockfile.

---

## Spelling spec decisions

Mirrors `docs/SPELLING-SPEC.md` §10 — amend in both places.

| Date | Rule | Decision | Reason |
|---|---|---|---|
| 2026-08-06 | §3.4 | Retroflex → plain `t`/`d`/`r` | ITRANS capitals unreadable in subtitles |
| 2026-08-06 | §3.2 | Keep `q` separate from `k` | `qismat` dominant in real usage |
| 2026-08-06 | §4.3 | Word-final long i → `i` not `ee` | Shorter; matches common typing |
| 2026-08-06 | §5.1 | English keeps English spelling | Core architectural requirement (ADR-001) |

---

## Dependency licenses

Every new dependency gets a row. Apache-2.0 / MIT preferred; copyleft needs sign-off.

| Package | License | Commercial OK | Notes |
|---|---|---|---|
| qwen-asr / Qwen3-ASR | Apache-2.0 | ✅ | Weights and code |
| Qwen3-ForcedAligner-0.6B | Apache-2.0 | ✅ | |
| IndicXlit | MIT | ✅ | |
| vLLM | Apache-2.0 | ✅ | |
| FastAPI | MIT | ✅ | |
| pyannote.audio | MIT | ✅ | Models gated on HF — accept terms |
| Demucs | MIT | ✅ | |
| ffmpeg | LGPL/GPL | ⚠️ | Depends on build flags — use LGPL build |
| **UrduSpeech corpus** | **TBD** | **❓** | **Blocking — see PROJECT.md R1** |
| Roman-Urdu-Parl | Research | ❓ | Verify commercial terms |
| Dakshina | CC BY-SA 4.0 | ⚠️ | Share-alike — check derived-work implications |

Added at Phase 0 scaffolding (2026-08-06):

| Package | License | Commercial OK | Notes |
|---|---|---|---|
| torch | BSD-3-Clause | ✅ | |
| torchaudio | BSD-2-Clause | ✅ | `[align]` extra |
| transformers | Apache-2.0 | ✅ | |
| pydantic | MIT | ✅ | Config validation |
| PyYAML | MIT | ✅ | |
| typer | MIT | ✅ | CLI entrypoints |
| uvicorn | BSD-3-Clause | ✅ | |
| python-multipart | Apache-2.0 | ✅ | Upload handling |
| hatchling | MIT | ✅ | Build backend |
| ruff | MIT | ✅ | dev only |
| mypy | MIT | ✅ | dev only |
| pytest / pytest-cov | MIT | ✅ | dev only |
| pre-commit | MIT | ✅ | dev only |
| types-pyyaml | Apache-2.0 | ✅ | dev only |
| uv | MIT / Apache-2.0 (dual) | ✅ | Dependency + interpreter management (ADR-007). A tool, not a linked dependency |
| flash-attn | BSD-3-Clause | ✅ | `cuda` extra, Linux-only; needs `no-build-isolation` |

No copyleft dependency has been added. `jiwer` was **not** added: CER and SN-WER
are short pure functions and an extra dependency for edit distance is not worth
the licence surface (`src/eval/metrics.py`).
