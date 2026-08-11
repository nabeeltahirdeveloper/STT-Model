"""Scoring CLI.

    python -m src.eval.score --pred out/predictions.txt --ref data/eval/reference.txt

Prints the §1.4 scorecard and the §6.3 error breakdown. The CLI shape is fixed
by `run.sh eval` and the two must not drift.

Two deliberate refusals:

*Raw WER never appears alone.* ADR-005 allows it in a table beside CER and
nowhere else, because Roman Urdu has no standard orthography and WER charges a
full error for a correct-but-variant spelling. The table below always prints CER
first and marks raw WER as diagnostic.

*The corpus score is not the mean of line scores.* See `metrics.aggregate`.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eval.metrics import (
    Score,
    aggregate,
    cer,
    classify_errors,
    english_preservation,
    normalized_wer,
    sn_wer,
    spelling_consistency,
    wer,
)
from src.labeling.lexicon import Lexicon
from src.labeling.normalize import Normalizer

DEFAULT_METRICS = "cer,sn-wer,normalized-wer,english-preservation"

# PROJECT.md §1.4. `None` means the spec sets no target, not that anything goes.
TARGETS: dict[str, tuple[float, str]] = {
    "cer": (0.12, "<"),
    "sn-wer": (0.25, "<"),
    "english-preservation": (0.90, ">"),
    "spelling-consistency": (0.98, ">"),
}


class ScoreError(RuntimeError):
    """Refused to score. Nothing was reported."""


def _read(path: Path) -> list[str]:
    if not path.exists():
        raise ScoreError(f"not found: {path}")
    return path.read_text(encoding="utf-8").splitlines()


def _verdict(name: str, value: float) -> str:
    target = TARGETS.get(name)
    if target is None:
        return ""
    threshold, direction = target
    ok = value < threshold if direction == "<" else value > threshold
    return f"  {'PASS' if ok else 'FAIL'}  (target {direction} {threshold:.0%})"


def score(references: list[str], hypotheses: list[str], normalizer: Normalizer) -> dict[str, Score]:
    """Every metric, aggregated over the corpus."""
    pairs = list(zip(references, hypotheses, strict=True))
    return {
        "cer": aggregate([cer(r, h) for r, h in pairs]),
        "sn-wer": aggregate([sn_wer(r, h) for r, h in pairs]),
        "normalized-wer": aggregate([normalized_wer(r, h, normalizer) for r, h in pairs]),
        "wer-raw": aggregate([wer(r, h) for r, h in pairs]),
        "english-preservation": aggregate(
            [english_preservation(r, h, normalizer) for r, h in pairs]
        ),
        "spelling-consistency": spelling_consistency(hypotheses, normalizer),
    }


def main(
    pred: str = "out/predictions.txt",
    ref: str = "data/eval/reference.txt",
    metrics: str = DEFAULT_METRICS,
    manifest: str = "data/eval/manifest.jsonl",
) -> None:
    """Score predictions against the held-out reference and print a table.

    Raises:
        ScoreError: if either file is missing, or they disagree on line count.
    """
    references, hypotheses = _read(Path(ref)), _read(Path(pred))
    if len(references) != len(hypotheses):
        raise ScoreError(
            f"{len(hypotheses)} predictions against {len(references)} references. "
            f"They are matched by line number, so a mismatch means every score "
            f"after the first missing line is measured against the wrong utterance."
        )

    normalizer = Normalizer(Lexicon.load())
    scores = score(references, hypotheses, normalizer)
    requested = [name.strip() for name in metrics.split(",") if name.strip()]

    print(f"\n{len(references)} utterances\n")
    print(f"{'metric':<24}{'rate':>9}{'errors':>10}{'of':>10}")
    print("-" * 70)
    for name in requested:
        if name not in scores:
            raise ScoreError(f"unknown metric {name!r}; known: {', '.join(scores)}")
        value = scores[name]
        # Preservation and consistency are reported as the rate that PASSES.
        shown = (
            1 - value.rate
            if name in ("english-preservation", "spelling-consistency")
            else value.rate
        )
        print(f"{name:<24}{shown:>8.1%}{value.errors:>10}{value.total:>10}{_verdict(name, shown)}")

    raw = scores["wer-raw"]
    print(f"{'wer-raw (diagnostic)':<24}{raw.rate:>8.1%}{raw.errors:>10}{raw.total:>10}")
    print("-" * 70)
    print("Raw WER is shown beside CER only, never alone: it charges a full error")
    print("for a correct-but-variant spelling (ADR-005).")

    breakdown = classify_errors(references, hypotheses, normalizer)
    print("\n§6.3 error breakdown")
    print(f"  acoustic      {breakdown.acoustic:>7.1%}   more/better training data")
    print(f"  orthographic  {breakdown.orthographic:>7.1%}   normalizer and lexicon work")
    print(f"  code-switch   {breakdown.code_switch:>7.1%}   label quality")
    print(f"  timing        {breakdown.timing:>7.1%}   aligner (needs §4.4; 0 until then)")

    path = Path(manifest)
    if path.exists():
        _per_category(path, references, hypotheses)


def _per_category(manifest: Path, references: list[str], hypotheses: list[str]) -> None:
    """CER by category. A single number hides that drama is twice as hard as news."""
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    if len(rows) != len(references):
        return  # manifest is for a different set; the main table is still valid

    grouped: dict[str, list[Score]] = {}
    for row, reference, hypothesis in zip(rows, references, hypotheses, strict=True):
        grouped.setdefault(str(row["category"]), []).append(cer(reference, hypothesis))

    print("\nCER by category")
    for category, scores in sorted(grouped.items(), key=lambda kv: -aggregate(kv[1]).rate):
        total = aggregate(scores)
        print(f"  {category:<18}{total.rate:>7.1%}   ({len(scores)} utterances)")


if __name__ == "__main__":  # pragma: no cover
    import typer

    typer.run(main)
