# Roman Urdu Auto-Captioning

Automatic **Roman Urdu** subtitles for video, built for how Pakistanis actually
speak — Urdu and English mixed in the same sentence.

```
video → audio → fine-tuned Qwen3-ASR → spelling normalizer → forced alignment → .srt / .vtt
```

Output is **Latin script**, not Perso-Arabic. That is a deliberate architectural
decision — see [`docs/PROJECT.md`](docs/PROJECT.md) §3.

---

## Status

**Phase 0 — verify and baseline.** Nothing is trained yet. Two gates block
everything downstream:

| Gate | Question | Command |
|---|---|---|
| R1 | Is the UrduSpeech corpus actually downloadable and commercially usable? | `python -m scripts.verify_data` |
| §4.4 | Does forced alignment work on Roman Urdu text? | `python -m scripts.test_aligner` |

---

## Quickstart

**Linux / macOS**

```bash
git clone <repo> && cd roman-urdu-captions
chmod +x run.sh
./run.sh setup
./run.sh doctor
./run.sh check
```

**Windows**

```bat
git clone <repo> && cd roman-urdu-captions
run.bat setup
run.bat doctor
run.bat check
```

Transcribe something:

```bash
./run.sh transcribe path/to/video.mp4 --out out/
```

Run `./run.sh help` for the full command list.

---

## Requirements

| | |
|---|---|
| Python | 3.12 |
| ffmpeg | required (audio extraction) |
| GPU | NVIDIA, ≥16 GB VRAM for inference, ≥40 GB for training |
| Disk | ~150 GB (corpora + checkpoints) |

Training is not viable on CPU. Inference on `Qwen3-ASR-0.6B` will run on modest
GPUs; `1.7B` wants more headroom.

**Phase 0 needs none of that.** Nothing in `src/` imports torch, `qwen-asr` or
`transformers` at module scope, so the spelling normalizer, the lexicons, the
tests and `./run.sh check` all run on a laptop. If a Phase 0 gate starts needing
a model dependency, a heavy import has leaked to module scope — fix the import.

---

## Why this problem is hard

Two things, and both shape every decision in the codebase:

**1. Roman Urdu has no standard spelling.** نہیں is validly `nahi`, `nahin`,
`nai`, or `nhi`. This means word error rate is a misleading metric — it punishes
correct-but-variant spellings — and it means *consistency* is the main driver of
perceived quality. Hence [`docs/SPELLING-SPEC.md`](docs/SPELLING-SPEC.md), which
is a product artifact, not documentation.

**2. Code-switching is the normal case, not an edge case.** Measured on a
Pakistani Urdu benchmark, Whisper-Large-v3 goes from 0.289 WER on monolingual
speech to 0.532 on code-switched speech, because it transliterates English words
into Urdu script instead of keeping them literal. That failure is why this project
does *not* route text through Perso-Arabic script.

---

## Architecture

| Stage | Module | Tooling |
|---|---|---|
| Ingest | `src/ingest/` | ffmpeg → 16 kHz mono 16-bit PCM |
| Preprocess | `src/preprocess/` | Demucs, pyannote 3.1; drop <2s, split >35s |
| ASR | `src/inference/asr.py` | fine-tuned Qwen3-ASR (0.6B dev / 1.7B prod) |
| Normalize | `src/labeling/normalize.py` | canonical spelling map |
| Align | `src/inference/align.py` | Qwen3-ForcedAligner, ≤5 min chunks |
| Subtitle | `src/subtitle/` | line breaking, reading speed, SRT/VTT |
| Serve | `src/api/` | vLLM + FastAPI |

Full detail in [`docs/PROJECT.md`](docs/PROJECT.md) §4.

---

## Documentation

| Doc | What it covers |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | Agent operating brief — constraints, conventions, commands |
| [`docs/PROJECT.md`](docs/PROJECT.md) | Full spec — scope, problem, solution, architecture, risks, roadmap |
| [`docs/SPELLING-SPEC.md`](docs/SPELLING-SPEC.md) | Canonical Roman Urdu orthography |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | ADR log |
| [`SETUP-PROMPT.md`](SETUP-PROMPT.md) | Claude Code bootstrap prompts |

---

## Contributing rules that actually matter

1. **Never route customer-facing text through Urdu or Devanagari script.**
2. **English words keep English spelling.** `meeting`, not `mitting`.
3. **Never train on `data/eval/`.** No exceptions.
4. **Never report raw WER as a headline metric.** Use CER and SN-WER.
5. **Run `./run.sh check` before every commit.**

The reasoning behind each is in [`CLAUDE.md`](CLAUDE.md).

---

## Licensing note

This is a commercial product. Core dependencies are Apache-2.0 or MIT
(Qwen3-ASR, IndicXlit, vLLM, FastAPI). Some optional components have different
terms — check `docs/DECISIONS.md` before adding anything, and note that dataset
licenses are tracked separately from code licenses.

Corpus licensing is an open risk (see PROJECT.md R1/R4). Resolve it before
taking revenue.
