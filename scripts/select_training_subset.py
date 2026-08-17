"""Choose a training subset and emit the download command for its audio.

    uv run python -m scripts.select_training_subset --hours 20

The full US-CS audio is 56 GB against 29 GB free, so local training needs a
subset. This picks one and writes the manifest; the audio download is printed
as a command to run rather than run here, because it is large.

Selection is deliberate rather than a head slice:

- **Unknown-word rate at or below `--max-unknown`.** A label containing Urdu
  script teaches the model to emit Urdu script. 2% is the agreed threshold: a
  stray unknown word in a long utterance costs less than dropping the acoustic
  variety of the whole line.
- **Duration capped.** Training runs at batch size 1 on 16 GB of shared memory,
  and activation memory scales with audio length. Long clips are what turns a
  fitting run into an OOM at step 900.
- **Spread across categories.** Drama, comedy, news and interviews differ
  acoustically; 20 hours of one of them teaches less than 20 hours of all.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

LABELS = Path("data/labels/labels.jsonl")
OUT = Path("data/labels/train-subset.jsonl")


def main(
    hours: float = 20.0,
    max_unknown: float = 0.02,
    max_seconds: float = 20.0,
    min_seconds: float = 1.0,
    labels: str = str(LABELS),
    out: str = str(OUT),
    exclude: str = "",
) -> None:
    """Write the subset manifest and print the command to fetch its audio.

    Args:
        exclude: a manifest whose clips must not appear here. Used to keep the
            dev set out of training -- a dev CER measured on memorised audio
            reports recall, not generalisation, and would read far better than
            the model deserves.
    """
    rows = [json.loads(line) for line in Path(labels).read_text(encoding="utf-8").splitlines()]

    if exclude:
        excluded = {
            json.loads(line)["audio"]
            for line in Path(exclude).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        before = len(rows)
        rows = [row for row in rows if row["audio"] not in excluded]
        print(f"excluded {before - len(rows):,} clips held out in {exclude}")

    eligible = []
    for row in rows:
        duration = float(row.get("duration_s") or 0)
        words = max(len(str(row["urdu"]).split()), 1)
        if not (min_seconds <= duration <= max_seconds):
            continue
        if len(row.get("unknown", [])) / words > max_unknown:
            continue
        eligible.append(row)

    by_category: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in eligible:
        by_category[str(row.get("category") or "?")].append(row)
    for bucket in by_category.values():
        bucket.sort(key=lambda r: float(r.get("duration_s") or 0))

    # Round-robin across categories so the subset is not one genre.
    budget = hours * 3600
    chosen: list[dict[str, object]] = []
    while budget > 0 and any(by_category.values()):
        for category in sorted(by_category):
            bucket = by_category[category]
            if not bucket or budget <= 0:
                continue
            row = bucket.pop(0)
            chosen.append(row)
            budget -= float(row.get("duration_s") or 0)

    destination = Path(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in chosen:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = sum(float(r.get("duration_s") or 0) for r in chosen) / 3600
    print(f"{len(rows):,} labels -> {len(eligible):,} eligible -> {len(chosen):,} chosen")
    print(f"  {total:.1f} h audio, clips {min_seconds:.0f}-{max_seconds:.0f}s")
    per_category = defaultdict(float)
    for row in chosen:
        per_category[str(row.get("category"))] += float(row.get("duration_s") or 0) / 3600
    for category, hrs in sorted(per_category.items(), key=lambda kv: -kv[1]):
        print(f"    {category:<20}{hrs:>6.1f} h")
    print(f"\n-> {destination}")

    approx_gb = total * 0.62  # measured: US-CS audio is ~0.62 GB per hour
    print(f"\nIts audio is not on disk yet (~{approx_gb:.0f} GB). Fetch exactly")
    print("these clips -- a directory download would pull the whole 56 GB split:\n")
    print(f"  uv run python -m scripts.fetch_training_audio --manifest {destination}")
    print(f"\nThen: uv run python -m scripts.train --manifest {destination}")


if __name__ == "__main__":  # pragma: no cover
    import typer

    typer.run(main)
