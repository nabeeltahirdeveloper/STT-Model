"""Carve a small held-out dev set for measuring CER *during* training.

    uv run python -m scripts.build_dev_set --clips 150

Every failure in this project's training so far was found at the end of a
session, because the only real metric was computed at the end. Script mix, loss,
p(eos) and character ratios are all proxies, and each of them read healthy for a
model that was in fact getting worse. CER is the goal; this is what lets the
training loop measure the goal while there is still time to act on it.

**Why a separate set rather than `data/eval/`.** The eval set is frozen and is
never scored against mid-run (CLAUDE.md constraint 5): a number watched during
training is a number that shapes decisions, and a benchmark used that way stops
being held out. This set exists to be watched. `data/eval/` stays for the one
measurement that goes in a report.

Selection mirrors `select_training_subset`: round-robin across categories, so a
dev CER is not dominated by whichever genre happens to sort first. The result is
written with `dev: true` so that `select_training_subset --exclude` can drop
exactly these rows from training, and a test asserts the two never overlap.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

LABELS = Path("data/labels/labels.jsonl")
OUT = Path("data/labels/dev-set.jsonl")


def select(
    rows: list[dict[str, object]],
    clips: int,
    min_seconds: float,
    max_seconds: float,
    max_unknown: float,
) -> list[dict[str, object]]:
    """Round-robin across categories, clips nearest the median duration first.

    Not shortest-first, which the first version did to keep the pass cheap. CER
    is errors divided by reference characters, so on a 12-character label two
    stray words read as 133% and the metric swings uselessly: a real run
    reported 49.9% then 170.6% while loss, p(eos) and the sample transcripts
    were all steady. The eval set averages 130 characters per clip, so a dev set
    of 16-character clips was not measuring the same thing at all.

    Selecting around the median keeps the pass affordable while making the
    number comparable to the gate it is meant to predict.
    """
    eligible = []
    for row in rows:
        duration = float(row.get("duration_s") or 0)
        words = max(len(str(row.get("urdu", "")).split()), 1)
        if not (min_seconds <= duration <= max_seconds):
            continue
        if len(row.get("unknown", [])) / words > max_unknown:  # type: ignore[arg-type]
            continue
        if not str(row.get("label") or "").strip():
            continue
        eligible.append(row)

    by_category: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in eligible:
        by_category[str(row.get("category") or "?")].append(row)
    # Nearest the median label length, not shortest: see the docstring.
    lengths = sorted(len(str(row.get("label", ""))) for row in eligible)
    median = lengths[len(lengths) // 2] if lengths else 0
    for bucket in by_category.values():
        bucket.sort(key=lambda r: abs(len(str(r.get("label", ""))) - median))

    chosen: list[dict[str, object]] = []
    categories = sorted(by_category)
    index = 0
    while len(chosen) < clips and any(by_category[c] for c in categories):
        bucket = by_category[categories[index % len(categories)]]
        if bucket:
            chosen.append(bucket.pop(0))
        index += 1
    return chosen


def main(
    clips: int = 150,
    labels: str = str(LABELS),
    out: str = str(OUT),
    min_seconds: float = 1.0,
    max_seconds: float = 20.0,
    max_unknown: float = 0.02,
) -> None:
    """Write the dev manifest and say how to exclude it from training."""
    rows = [
        json.loads(line)
        for line in Path(labels).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    chosen = select(rows, clips, min_seconds, max_seconds, max_unknown)
    if not chosen:
        raise SystemExit(f"no eligible rows in {labels}")

    destination = Path(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "\n".join(json.dumps({**row, "dev": True}, ensure_ascii=False) for row in chosen) + "\n",
        encoding="utf-8",
    )

    seconds = sum(float(r.get("duration_s") or 0) for r in chosen)
    by_category: dict[str, int] = defaultdict(int)
    for row in chosen:
        by_category[str(row.get("category") or "?")] += 1
    print(f"{len(rows):,} labels -> {len(chosen):,} dev clips ({seconds / 60:.1f} min)")
    for category, count in sorted(by_category.items(), key=lambda kv: -kv[1]):
        print(f"    {category:22} {count:>4}")
    print(f"\n-> {destination}")
    print(
        "\nExclude these from training, or the dev CER measures recall rather\n"
        "than generalisation:\n"
        f"  uv run python -m scripts.select_training_subset --exclude {destination}"
    )


if __name__ == "__main__":  # pragma: no cover
    import typer

    typer.run(main)
