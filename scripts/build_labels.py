"""Romanize the UrduSpeech corpus into Phase 1 training labels.

    uv run python -m scripts.build_labels --split US-CS
    uv run python -m scripts.build_labels --split US-CS --limit 5000
    uv run python -m scripts.build_labels                     # everything

**Resumable.** Romanizing 66,709 utterances takes hours, and a job that loses
its work when a terminal closes will not be run to completion. Output is JSONL,
appended in batches, and a re-run skips every `audio_id` already present.

**Romanization and normalization are separated on purpose.** Romanizing is slow
(a neural model, minutes per hundred utterances) and does not depend on the
spelling spec. Normalizing is fast and depends on it entirely. So this writes
the romanized form *and* the normalized form, and `--renormalize` recomputes
only the second from the stored first. A spec change after the freeze therefore
costs seconds rather than another overnight run -- which matters, because
SPELLING-SPEC §12 warns that post-freeze amendments are expensive and this is
the part that would have made them so.

Eval clips are excluded by `audio_id`; the script refuses to run without the
eval manifest (CLAUDE.md constraint 5).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from scripts.romanize_via_opencut import romanize
from src.labeling.lexicon import Lexicon
from src.labeling.normalize import Normalizer

CORPUS = Path("data/raw/urduspeech/corpus")
EVAL_MANIFEST = Path("data/eval/manifest.jsonl")
OUT = Path("data/labels/labels.jsonl")


def _eval_ids() -> set[str]:
    if not EVAL_MANIFEST.exists():
        raise SystemExit(
            f"{EVAL_MANIFEST} not found. Refusing to build labels without the "
            f"eval exclusion list (CLAUDE.md constraint 5)."
        )
    ids: set[str] = set()
    for line in EVAL_MANIFEST.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        ids.add(Path(str(row["audio"])).stem)
    return ids


def load_corpus(root: Path, split: str, min_confidence: float) -> list[dict[str, object]]:
    """Every transcript row above the confidence floor, tagged with its split."""
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*_final_transcription.jsonl")):
        # Relative to the corpus root, so the split name does not depend on how
        # deep the corpus happens to sit in the tree.
        relative = path.relative_to(root).parts
        row_split = relative[0] if relative else "?"
        if split and row_split != split:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                confidence = float(row.get("Confidence_score") or 0)
            except (TypeError, ValueError):
                confidence = 0.0
            if confidence < min_confidence:
                continue
            text = str(row.get("ground_truth") or row.get("Transcription") or "").strip()
            if not text:
                continue
            rows.append(
                {
                    "audio_id": str(row.get("audio_id") or row.get("Audio_Clip") or ""),
                    "split": row_split,
                    "category": row.get("Audio_category"),
                    "duration_s": row.get("Duration_seconds"),
                    "confidence": confidence,
                    "urdu": text,
                }
            )
    return rows


def _done_ids(out: Path) -> set[str]:
    if not out.exists():
        return set()
    return {
        str(json.loads(line)["audio_id"])
        for line in out.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def renormalize(out: Path) -> None:
    """Recompute `label` from the stored romanization. Seconds, not hours."""
    normalizer = Normalizer(Lexicon.load())
    rows = [
        json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    changed = 0
    for row in rows:
        new = normalizer.normalize_text(str(row["roman"]))
        if new != row.get("label"):
            row["label"] = new
            changed += 1
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"renormalized {len(rows):,} labels, {changed:,} changed")


def main(
    split: str = "",
    limit: int = 0,
    batch: int = 500,
    min_confidence: float = 0.0,
    out: str = str(OUT),
    corpus: str = str(CORPUS),
    renormalize_only: bool = False,
) -> None:
    """Romanize and normalize corpus transcripts into training labels."""
    destination = Path(out)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if renormalize_only:
        renormalize(destination)
        return

    excluded = _eval_ids()
    rows = load_corpus(Path(corpus), split, min_confidence)
    already = _done_ids(destination)
    pending = [
        row
        for row in rows
        if str(row["audio_id"]) not in already and str(row["audio_id"]) not in excluded
    ]
    if limit:
        pending = pending[:limit]

    print(f"{len(rows):,} rows in scope · {len(already):,} already done · {len(pending):,} to do")
    if not pending:
        print("nothing to do")
        return

    normalizer = Normalizer(Lexicon.load())
    started = time.time()
    with destination.open("a", encoding="utf-8") as handle:
        for start in range(0, len(pending), batch):
            chunk = pending[start : start + batch]
            print(
                f"batch {start // batch + 1}/{-(-len(pending) // batch)} "
                f"({len(chunk)} utterances) — the romanizer prints its own "
                f"progress below; a batch takes a few minutes",
                flush=True,
            )
            romanized = romanize([str(row["urdu"]) for row in chunk])
            for row, roman in zip(chunk, romanized, strict=True):
                row["roman"] = roman
                row["label"] = normalizer.normalize_text(roman)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            done = start + len(chunk)
            rate = (time.time() - started) / done
            print(
                f"  {done:,}/{len(pending):,} done · "
                f"{rate * (len(pending) - done) / 60:.0f} min left",
                flush=True,
            )

    print(f"\n-> {destination}")


if __name__ == "__main__":  # pragma: no cover
    import typer

    typer.run(main)
