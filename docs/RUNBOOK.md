# Runbook

How to set up this project and run every step, in the order they happen.

`README.md` says what the project is. `PROJECT.md` says why it is built this
way. **This file says what to type.**

Every command assumes you are in the repository root. There is no virtualenv to
activate — `uv run` resolves the environment per invocation.

---

## 1. Environment

Install [uv](https://docs.astral.sh/uv/). It is the only thing you install by
hand; it provisions Python 3.12 as well as the packages.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh          # macOS / Linux
powershell -c "irm https://astral.sh/uv/install.ps1|iex" # Windows
```

First time on a machine:

```bash
./run.sh setup      # provisions Python, syncs deps, scaffolds dirs, installs git hooks
./run.sh doctor     # checks ffmpeg, uv, disk, GPU
```

Afterwards:

```bash
uv sync --all-extras --all-groups   # everything, from uv.lock
uv sync                             # runtime deps only — fastest loop
```

`ffmpeg` must be on PATH (`brew install ffmpeg` / `apt install ffmpeg`).

### Before every commit

```bash
./run.sh check      # ruff check, ruff format, mypy, pytest — in that order
```

Git hooks run these too, and will **rewrite files and abort the commit** if
formatting is off. That is not a failure: re-stage and commit again.

### What runs where

| Task | Runs on this Mac? |
|---|---|
| Lexicon, normalizer, tests, scoring | Yes, instantly |
| Romanization (label generation) | Yes, slow — hours |
| ASR inference | Yes — 1.86× realtime on the 1.7B |
| Forced alignment | Yes |
| **Full fine-tuning** | **No.** ~12.5 GB of optimizer state before activations, against 16 GB shared with macOS. Rent 2× H100 (~50 min, ~$6.50) |

---

## 2. Data

Large downloads are listed with their size so you can decide when to run them.

```bash
# Eval + label transcripts — small, needed for everything
uv run --with huggingface_hub hf download ASLP-lab/UrduSpeech \
  --repo-type dataset --include "corpus/**/*_final_transcription.jsonl" \
  --local-dir data/raw/urduspeech                                    # 31 MB

# Spelling frequency corpus — needed to rebuild the lexicon
uv run --with huggingface_hub hf download Mavkif/Roman-Urdu-Parl-split \
  --repo-type dataset --include "original_data/roman-urdu.txt" \
  --local-dir data/raw/roman-urdu-parl                               # 455 MB

# ASR model — needed for baselines and inference
uv run --with huggingface_hub hf download Qwen/Qwen3-ASR-1.7B        # 4.4 GB

# Audio for training — only needed on the training box, not here
# corpus/US-CS is 56 GB. Do not download it to a 78 GB laptop.
```

Downloads resume. Re-run the same command after an interruption.

---

## 3. The spelling lexicon

The lexicon decides how every Roman Urdu word is spelled, everywhere. It is
frozen before training because changing it afterwards means regenerating labels
and retraining (`SPELLING-SPEC.md` §12, risk R8).

```bash
# Count the corpus and check every canonical spelling against real usage
uv run python -m scripts.build_lexicon              # ~4 min, writes docs/lexicon-frequency.tsv
```

Reading the output: `AGREES` means our spelling matches usage; `CORPUS
DISAGREES` means it loses and §2 says the corpus wins; `NEAR-TIE` within 15%
means the §3–§4 letter rules decide; `UNSEEN` means the corpus never attests it
and a human must choose.

> A near-tie between two *rare* spellings usually means the real answer is
> missing from the lexicon entirely, not that the choice is balanced. That is
> how `kaisay` (52,463 uses) was found hiding behind a tie between `kaise` (8)
> and `kese` (7).

The files it judges, all hand-editable:

| File | Holds |
|---|---|
| `data/lexicon/canonical.tsv` | `urdu <TAB> canonical <TAB> rejected,variants` |
| `data/lexicon/english.txt` | English words that pass through untouched |
| `data/lexicon/acronyms.txt` | Always uppercase. Outranks `english.txt` |
| `data/lexicon/ambiguous.tsv` | Tokens that are both English and Urdu, decided by context |

**Never put a common English word in a variant list.** Mapping `they`, `such`,
`no` or `keh` onto an Urdu spelling silently respells English — this has been
found four separate times (ADR-012, ADR-013). The loader rejects undeclared
collisions, so run this after any lexicon edit:

```bash
uv run pytest tests/test_lexicon.py -q
```

---

## 4. The eval set

**`data/eval/` is never trained on.** Not once, not for a quick check.

It already exists — 269 utterances, 35.4 min, 12 categories, human-corrected.
Rebuild it only if you mean to.

```bash
# 1. Machine drafts from the corpus transcripts (slow — the romanizer)
uv run python -m scripts.build_eval_from_corpus --minutes-per-category 3

# 2. A human corrects every line in data/eval/drafts-corpus/*.txt
#    Only the plain lines. The `#` line above each is the verified Urdu.
#    Append `??` to a line where the corpus transcript itself looks wrong.

# 3. Promote drafts to the reference. Refuses on empty lines, non-Latin script,
#    or any category more than half identical to the machine draft.
uv run python -m scripts.build_eval_reference
```

---

## 5. Baseline and scoring

```bash
# Check speed, supported languages and output script before committing an hour
uv run python -m scripts.probe_asr

# Transcribe the whole eval set — ~66 min, writes raw + romanized
uv run python -m scripts.run_baseline

# Score
uv run python -m src.eval.score --pred out/baseline-romanized.txt
```

Reading the scorecard:

- **CER** is primary. Target < 12%.
- **SN-WER** ignores case and punctuation but *not* spelling. Target < 25%.
- **normalized-WER** additionally applies the lexicon. The gap between it and
  SN-WER is the work the normalizer is doing.
- **raw WER** appears beside CER and never alone — it charges a full error for a
  correct-but-variant spelling (ADR-005).
- **§6.3 breakdown** splits errors into acoustic (needs data), orthographic
  (needs lexicon work) and code-switch (needs label quality). This is the number
  that decides where effort goes.

> Scores are not comparable across a spelling-spec change. When ADR-011 replaced
> 19 spellings, CER moved 26.7% → 27.9% without the model changing at all.
> `normalized-wer` is the spec-invariant one.

---

## 6. Training labels

```bash
# Small batch first — ~2 h, enough to eyeball the output
uv run python -m scripts.build_labels --split US-CS --limit 2000

# The real run — ~28 h for 29,749 code-switched utterances
uv run python -m scripts.build_labels --split US-CS
```

Resumable: Ctrl-C, re-run the same command, it skips what is done. Use `screen`
or `tmux` so closing the terminal does not kill it.

After a spelling-spec change, **do not re-romanize**:

```bash
uv run python -m scripts.build_labels --renormalize-only    # seconds
```

Romanization is slow and spec-independent; normalization is fast and
spec-dependent. They are stored separately for exactly this reason.

### Reviewing labels before the freeze

```bash
uv run python -m scripts.build_label_sample --count 500
```

Writes `data/labels/sample-500.txt`, grouped by rule rather than shuffled — 
checking 80 English-preservation cases in a row catches a systematic breach that
500 mixed lines would hide. Mark a bad line with `X` at the start.

---

## 7. Timing / forced alignment

```bash
# Pick clips and pre-fill the word list from the reference
uv run python -m scripts.prepare_alignment_clips

# Draft the onsets with an independent aligner (MMS), for a human to correct
uv run python -m scripts.draft_onsets

# Correct them in Audacity: File > Import > Labels, drag what is wrong,
# File > Export > Export Labels. Then convert:
uv run python -m scripts.prepare_alignment_clips --from-audacity data/eval/alignment

# Score
uv run python -m scripts.test_aligner --clips data/eval/alignment
```

> Onset files must contain *real* timings. Evenly-spaced placeholders score the
> aligner against a uniform-spacing assumption and produce a confident,
> meaningless number. Check that word durations vary before trusting a result.

---

## 8. Training

Not on this machine — see §1. On a rented box:

```bash
uv sync --all-extras                      # includes the cuda extra on Linux
uv run python -m src.training.finetune --config configs/phase1.yaml
```

Order: prove the recipe on `Qwen3-ASR-0.6B` first, then scale to `1.7B`. Full
fine-tuning is the default; LoRA needs an A/B against it first (constraint 6).

---

## Troubleshooting

**`uv add` fails with "unsatisfiable"** — something pins a conflicting version.
`qwen-asr` pins `transformers==4.57.6` and ships its own model code, so do not
upgrade transformers to reach Qwen3-ASR.

**`LexiconError: tokens appear in both english.txt and the Roman Urdu lexicon`** —
a word is claimed by both. Either drop it from `english.txt` or add a context
rule to `ambiguous.tsv`. This guard is doing its job; do not disable it.

**Pre-commit rewrites files and aborts** — expected. Re-stage and commit again.

**A long job died with the terminal** — every long-running script here is
resumable. Re-run the same command.

**Scores changed and the model did not** — check whether the lexicon changed.
See §5.

---

## Where the documentation lives

| Doc | Answers |
|---|---|
| `README.md` | What is this? |
| `docs/RUNBOOK.md` | What do I type? *(this file)* |
| `docs/PROJECT.md` | Why is it built this way? |
| `docs/SPELLING-SPEC.md` | How is a word spelled? |
| `docs/DECISIONS.md` | Why was that decided, and what was tried and failed? |
| `CLAUDE.md` | Constraints and conventions for agents |
