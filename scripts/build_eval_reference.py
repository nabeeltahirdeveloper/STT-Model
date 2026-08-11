"""Promote corrected drafts into the held-out evaluation reference.

    uv run python -m scripts.build_eval_reference

Reads `data/eval/drafts-corpus/*.txt` and writes the two files
`data/eval/README.md` specifies: `reference.txt` (one utterance per line) and
`manifest.jsonl` (`{"audio": ..., "text": ...}` per utterance).

This is the step where machine output becomes ground truth, so it is the step
that has to be suspicious. CLAUDE.md constraint 5 makes the eval set sacred, and
the cheapest way to violate it is not malice but momentum -- a file that was
never opened looks exactly like a file that was reviewed and needed no changes.
The guards below exist to tell those two apart, and they refuse to write rather
than warn, because a warning at the end of a long run is a warning nobody reads.

A line identical to its machine draft is *not* by itself an error: 26 of the 269
lines were short enough that the romanizer got them right first time, and
"Ready? Chalo" has one correct spelling. What would be an error is a whole file
of them, so the guard is proportional, not absolute.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

DRAFTS = Path("data/eval/drafts-corpus")
OUT = Path("data/eval")

# Above this share of machine-identical lines, a file reads as unreviewed rather
# than as reviewed-and-already-correct. Chosen from the observed rate: the real
# figure across the corrected set is 10%, so half the file is a wide margin.
_UNREVIEWED_SHARE = 0.5

# A reviewer appends `??` where the corpus transcript itself looked wrong. It is
# metadata about the line, not part of what was said, so it never reaches the
# reference text -- but it is worth keeping as a field.
_FLAG = re.compile(r"\s*\?\?\s*$")

_NON_LATIN = re.compile(r"[؀-ۿݐ-ݿऀ-ॿ]")


class ReferenceError(RuntimeError):
    """A guard refused to promote the drafts. The eval set is unchanged."""


@dataclass(slots=True)
class Utterance:
    audio: str
    category: str
    duration_s: float
    text: str
    corpus_transcript: str
    corpus_disputed: bool
    unchanged_from_draft: bool


def load(drafts: Path, manifest: list[dict[str, object]]) -> list[Utterance]:
    """Parse the per-category files back into utterances, keyed by audio path."""
    meta = {str(row["audio"]): row for row in manifest}
    out: list[Utterance] = []
    for path in sorted(drafts.glob("*.txt")):
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if not line.startswith("["):
                continue
            audio = line[1:-1]
            row = meta.get(audio)
            if row is None:
                raise ReferenceError(f"{path.name}: no manifest row for {audio}")
            raw = lines[index + 2]
            text = _FLAG.sub("", raw).strip()
            out.append(
                Utterance(
                    audio=audio,
                    category=str(row["category"]),
                    duration_s=float(row["duration_s"]),  # type: ignore[arg-type]
                    text=text,
                    corpus_transcript=str(row["corpus_transcript"]),
                    corpus_disputed=bool(_FLAG.search(raw)),
                    unchanged_from_draft=text == str(row["draft"]).strip(),
                )
            )
    return out


def check(utterances: list[Utterance]) -> None:
    """Refuse to write if anything looks unreviewed or malformed."""
    if not utterances:
        raise ReferenceError("no utterances parsed")

    empty = [u.audio for u in utterances if not u.text]
    if empty:
        raise ReferenceError(f"{len(empty)} empty reference lines, first: {empty[0]}")

    # Constraint 1/2: the output script is Latin. Any Perso-Arabic or Devanagari
    # character means a line was never romanized, not that it was reviewed.
    left = [u.audio for u in utterances if _NON_LATIN.search(u.text)]
    if left:
        raise ReferenceError(f"{len(left)} lines still contain non-Latin script, first: {left[0]}")

    by_category: dict[str, list[Utterance]] = {}
    for utterance in utterances:
        by_category.setdefault(utterance.category, []).append(utterance)
    for category, rows in sorted(by_category.items()):
        share = sum(u.unchanged_from_draft for u in rows) / len(rows)
        if share > _UNREVIEWED_SHARE:
            raise ReferenceError(
                f"{category}: {share:.0%} of lines are byte-identical to the machine "
                f"draft, which reads as unreviewed. The eval set may not contain "
                f"unverified machine output (CLAUDE.md constraint 5)."
            )


def main(drafts: str = str(DRAFTS), out_dir: str = str(OUT)) -> None:
    """Validate the corrected drafts and write reference.txt + manifest.jsonl."""
    source = Path(drafts)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    utterances = load(source, manifest)
    check(utterances)

    # Sorted by audio path so the reference and the manifest can never drift out
    # of order relative to each other, whatever order the files were read in.
    utterances.sort(key=lambda u: u.audio)

    destination = Path(out_dir)
    (destination / "reference.txt").write_text(
        "\n".join(u.text for u in utterances) + "\n", encoding="utf-8"
    )
    with (destination / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for u in utterances:
            handle.write(
                json.dumps(
                    {
                        "audio": u.audio,
                        "text": u.text,
                        "category": u.category,
                        "duration_s": u.duration_s,
                        "corpus_transcript": u.corpus_transcript,
                        "corpus_disputed": u.corpus_disputed,
                        "unchanged_from_draft": u.unchanged_from_draft,
                        "reference_status": "human-corrected",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    minutes = sum(u.duration_s for u in utterances) / 60
    unchanged = sum(u.unchanged_from_draft for u in utterances)
    disputed = sum(u.corpus_disputed for u in utterances)
    categories = len({u.category for u in utterances})
    print(f"{len(utterances)} utterances · {minutes:.1f} min · {categories} categories")
    print(f"  identical to machine draft : {unchanged} ({unchanged / len(utterances):.0%})")
    print(f"  corpus disputed (??)       : {disputed}")
    print(f"  -> {destination}/reference.txt")
    print(f"  -> {destination}/manifest.jsonl")


if __name__ == "__main__":  # pragma: no cover
    import typer

    typer.run(main)
