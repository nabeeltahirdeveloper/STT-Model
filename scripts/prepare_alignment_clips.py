"""Prepare clips for the forced-alignment experiment (PROJECT.md §4.4).

    uv run python -m scripts.prepare_alignment_clips
    uv run python -m scripts.prepare_alignment_clips --from-audacity ~/labels

Picks a spread of short clips from the eval set, copies each `<name>.wav` into
`data/eval/alignment/`, and writes a `<name>.tsv` skeleton with the tokens
already filled in from the corrected reference -- so the only thing left to do
by hand is the one thing a human has to do, which is say when each word starts.

Why the tokens are pre-filled: `scripts/test_aligner.py` requires the reference
token count to equal the aligner's, and treats a mismatch as a harder failure
than bad timing. Retyping the words by hand would make token mismatches a
constant nuisance that has nothing to do with the question being asked.

Times are written as `TODO` rather than `0.0`. An unfilled file must fail loudly
when scored; a file full of zeros would score as a catastrophic alignment error
and look like a real result.

**Marking onsets by hand is slow. Use Audacity instead.** Open the wav, play it,
press Ctrl+B at each word onset to drop a label, then File > Export > Export
Labels. Re-run this script with `--from-audacity <dir>` and it converts those
label files into the `.tsv` format, matching them to clips by filename. Audacity
writes `start<TAB>end<TAB>text`; only the start is used.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.labeling.normalize import tokenize

MANIFEST = Path("data/eval/manifest.jsonl")
ALIGNMENT = Path("data/eval/alignment")

_HEADER = """\
# {name}
# {duration:.1f}s · {category} · {count} tokens
#
# Replace every TODO with the time in seconds when that word STARTS.
# Decimals are fine and expected: 1.24, not 1.
#
# Do not add, remove or reorder lines. The aligner is scored token-by-token and
# a count mismatch is reported as a different failure than a timing error.
#
# Faster route: mark onsets in Audacity (Ctrl+B at each word), export labels,
# then run:  uv run python -m scripts.prepare_alignment_clips --from-audacity <dir>
#
# Reference text:
# {text}
"""


def _pick(rows: list[dict[str, object]], count: int, max_seconds: float) -> list[dict[str, object]]:
    """A spread of short clips, at most one or two per category.

    Short because every second of clip is a second of marking. Spread because a
    timing result from one genre says nothing about the others -- news is read
    speech and drama has overlap, and the aligner may well behave differently.
    """
    usable = [
        row
        for row in rows
        if 2.0 <= float(row["duration_s"]) <= max_seconds  # type: ignore[arg-type]
        and len(tokenize(str(row["text"]))) >= 4
    ]
    by_category: dict[str, list[dict[str, object]]] = {}
    for row in sorted(usable, key=lambda r: float(r["duration_s"])):  # type: ignore[arg-type]
        by_category.setdefault(str(row["category"]), []).append(row)

    picked: list[dict[str, object]] = []
    while len(picked) < count and any(by_category.values()):
        for category in sorted(by_category):
            if by_category[category] and len(picked) < count:
                picked.append(by_category[category].pop(0))
    return picked


def _skeleton(row: dict[str, object], destination: Path) -> int:
    """Copy the wav and write the tsv skeleton. Returns the token count."""
    source = Path(str(row["audio"]))
    name = source.stem
    shutil.copy2(source, destination / f"{name}.wav")

    tokens = [token for token in tokenize(str(row["text"])) if any(c.isalnum() for c in token)]
    body = "\n".join(f"{token}\tTODO" for token in tokens)
    (destination / f"{name}.tsv").write_text(
        _HEADER.format(
            name=name,
            duration=float(row["duration_s"]),  # type: ignore[arg-type]
            category=row["category"],
            count=len(tokens),
            text=row["text"],
        )
        + body
        + "\n",
        encoding="utf-8",
    )
    return len(tokens)


def _convert(labels: Path, destination: Path) -> None:
    """Turn Audacity label exports into `.tsv` onsets, matched by filename."""
    converted = 0
    for label_file in sorted(labels.glob("*.txt")):
        target = destination / f"{label_file.stem}.tsv"
        if not target.exists():
            print(f"  skipped {label_file.name}: no matching clip in {destination}")
            continue
        starts: list[float] = []
        for line in label_file.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                starts.append(float(parts[0]))

        existing = target.read_text(encoding="utf-8").splitlines()
        head = [line for line in existing if line.startswith("#")]
        tokens = [line.split("\t")[0] for line in existing if line and not line.startswith("#")]
        if len(starts) != len(tokens):
            print(
                f"  MISMATCH {label_file.name}: {len(starts)} labels against "
                f"{len(tokens)} tokens — not written"
            )
            continue
        target.write_text(
            "\n".join(head)
            + "\n"
            + "\n".join(f"{t}\t{s:.3f}" for t, s in zip(tokens, starts, strict=True))
            + "\n",
            encoding="utf-8",
        )
        converted += 1
        print(f"  {label_file.name} -> {target.name} ({len(starts)} onsets)")
    print(f"\nconverted {converted} clips")


def main(
    count: int = 20,
    max_seconds: float = 12.0,
    out_dir: str = str(ALIGNMENT),
    from_audacity: str = "",
) -> None:
    """Generate clip + skeleton pairs, or fill them from Audacity labels."""
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)

    if from_audacity:
        _convert(Path(from_audacity), destination)
        return

    rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines()]
    picked = _pick(rows, count, max_seconds)
    if not picked:
        raise SystemExit("no clips matched the duration filter")

    total_tokens = 0
    total_seconds = 0.0
    for row in picked:
        total_tokens += _skeleton(row, destination)
        total_seconds += float(row["duration_s"])  # type: ignore[arg-type]
        print(f"  {Path(str(row['audio'])).stem:<34} {row['category']}")

    print(f"\n{len(picked)} clips · {total_seconds:.0f}s audio · {total_tokens} onsets to mark")
    print(f"-> {destination}/")


if __name__ == "__main__":  # pragma: no cover
    import typer

    typer.run(main)
