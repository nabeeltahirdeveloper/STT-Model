"""Generate a sample of Phase 1 training labels for spec-compliance review.

    uv run python -m scripts.build_label_sample --count 500

SPELLING-SPEC §12's last substantive freeze gate is "500 random labels
hand-reviewed for spec compliance". This produces them.

**Eval clips are excluded, by audio path, and the script refuses to run if the
exclusion list is missing.** Reviewing labels that overlap the eval set would
put eval audio in front of the process that produces training data, which is the
first step toward the leak CLAUDE.md constraint 5 forbids. The check is cheap
and the failure it prevents is silent and unrecoverable.

Labels are produced the way training labels will be: corpus transcript →
OpenCut romanization → canonical normalizer. Reviewing anything else would test
a pipeline we are not going to ship.

The output is grouped by *rule* rather than by clip, because the question here
is not "is this line right" but "does the spec hold" — and a reviewer checking
80 English-preservation cases in a row is far more likely to notice a
systematic breach than one checking 500 mixed lines.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from src.labeling.lexicon import Lexicon
from src.labeling.normalize import Normalizer
from src.labeling.transliterate import load as load_translit
from src.labeling.transliterate import romanize

BENCHMARK = Path("data/raw/urduspeech/benchmark/US-benchmark-CS")
EVAL_MANIFEST = Path("data/eval/manifest.jsonl")
OUT = Path("data/labels/sample-500.txt")

_LATIN = re.compile(r"[A-Za-z][A-Za-z']+")
_HEADER = """\
# Phase 1 label sample — {count} lines, for SPELLING-SPEC §12 review
#
# Every line: does it obey the spec? Mark a bad line by putting X at the start.
#
#   [urdu]  the corpus transcript — what was said
#   [label] what the pipeline produced — what the model would be trained on
#
# The eval set is excluded from this sample; nothing here overlaps it.
#
# WHAT TO LOOK FOR, in priority order
#   1. English respelled          `meeting` -> `mitting`  (worst; §5.1)
#   2. Wrong word                 the label says something else entirely
#   3. Inconsistent spelling      the same word spelled two ways in this file
#   4. Non-ASCII, or Urdu/Devanagari characters left in
"""


def load_rows(root: Path, exclude: set[str]) -> list[dict[str, str]]:
    """Every benchmark transcript row whose audio is not in the eval set."""
    rows: list[dict[str, str]] = []
    for jsonl in sorted(root.rglob("clean_transcription.jsonl")):
        audio_dir = jsonl.parent / "audio"
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            audio = str(audio_dir / str(row.get("Audio_Clip", "")))
            if audio in exclude:
                continue
            text = str(row.get("ground_truth") or row.get("Transcription") or "").strip()
            if len(text) > 12:
                rows.append({"audio": audio, "urdu": text})
    return rows


def main(count: int = 500, out: str = str(OUT), seed: int = 20260810) -> None:
    """Romanize and normalize a random non-eval sample, ready for review."""
    if not EVAL_MANIFEST.exists():
        raise SystemExit(
            f"{EVAL_MANIFEST} not found. Refusing to sample without the eval "
            f"exclusion list — overlapping it risks the leak constraint 5 forbids."
        )
    exclude = {
        str(json.loads(line)["audio"])
        for line in EVAL_MANIFEST.read_text(encoding="utf-8").splitlines()
    }

    rows = load_rows(BENCHMARK, exclude)
    print(f"{len(rows):,} candidate rows ({len(exclude)} eval clips excluded)")
    random.Random(seed).shuffle(rows)
    chosen = rows[:count]

    print(f"romanizing {len(chosen)} ...", flush=True)
    normalizer = Normalizer(Lexicon.load())
    table = load_translit()
    labels = [normalizer.normalize_text(romanize(r["urdu"], table).text) for r in chosen]

    # Group by the rule most worth checking on each line, so a reviewer sees
    # like with like. A line can only appear once; the first bucket it matches
    # wins, ordered by how expensive the failure is.
    buckets: dict[str, list[tuple[str, str]]] = {
        "A. lines containing English (check §5.1 first)": [],
        "B. lines with numbers or codes": [],
        "C. everything else": [],
    }
    for row, label in zip(chosen, labels, strict=True):
        pair = (row["urdu"], label)
        if _LATIN.search(row["urdu"]):
            buckets["A. lines containing English (check §5.1 first)"].append(pair)
        elif re.search(r"\d", row["urdu"]):
            buckets["B. lines with numbers or codes"].append(pair)
        else:
            buckets["C. everything else"].append(pair)

    destination = Path(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [_HEADER.format(count=len(chosen))]
    for title, pairs in buckets.items():
        lines.append(f"\n# ===== {title} — {len(pairs)} lines =====\n")
        for urdu, label in pairs:
            lines += [f"[urdu]  {urdu}", f"[label] {label}", ""]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for title, pairs in buckets.items():
        print(f"  {title}: {len(pairs)}")
    print(f"\n-> {destination}")


if __name__ == "__main__":  # pragma: no cover
    import typer

    typer.run(main)
