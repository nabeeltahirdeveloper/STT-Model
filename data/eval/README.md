# Held-out evaluation set

**Nothing in this directory is ever trained on.** No exceptions, not even for a
quick sanity check (CLAUDE.md constraint 5). `src.config.DataConfig` rejects any
training manifest whose path is under `data/eval/`, and that guard exists because
a leaked eval set silently invalidates every number the project reports.

Empty right now. Phase 0 gate: it must be built before any training code is
written (PROJECT.md §6.1).

## What goes here

| File | What it is |
|---|---|
| `manifest.jsonl` | `{"audio": ..., "text": ...}` per utterance |
| `reference.txt` | Plain reference transcript, one utterance per line |
| `alignment/` | `<name>.wav` + `<name>.tsv` (`token <TAB> start_seconds`) for `scripts/test_aligner.py` |

## How to build it

- 30–60 minutes of **your actual target content**, not benchmark audio. Real
  content has music, overlap and noise; benchmarks do not, and PROJECT.md R6
  says that gap is where projects like this get surprised.
- Hand-transcribed in the canonical Roman convention (`docs/SPELLING-SPEC.md`).
- Plus the 9-hour UrduSpeech benchmark, romanized identically, once R1 clears.

The target content type is still undecided (PROJECT.md open question 1), and it
changes what belongs here. Decide that before spending hours transcribing.
