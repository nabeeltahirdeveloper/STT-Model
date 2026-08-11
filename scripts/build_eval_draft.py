"""Build correctable Roman Urdu drafts from ASR output, for a human to fix.

    uv run python -m scripts.build_eval_draft

Hand-transcribing code-switched podcast audio from scratch is hours of typing;
correcting a decent draft is roughly three times faster. That is the only
purpose of this script.

The draft is NEVER the eval set. A machine-generated reference measures how
well one model imitates another, which is worse than useless -- it would look
like a real number. Every line needs a human ear on it before
`data/eval/reference.txt` exists (PROJECT.md §6.1, CLAUDE.md constraint 5).

Romanization is OpenCut's job (`scripts/romanize_via_opencut`); canonical
spelling is ours (`src/labeling/normalize`). This file only segments, wires the
two together, and writes something a human can edit in any text editor.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.romanize_via_opencut import romanize
from src.labeling.lexicon import Lexicon
from src.labeling.normalize import Normalizer

BENCHMARK = Path("/Users/mubeen-dev/Documents/quantum/opencut/benchmarks/urdu-caption-20260730")

# Sentence enders across all three scripts, including the Devanagari danda.
_SEG_BREAK = re.compile(r"[.?!؟۔।]$")
# Longer lines are harder to correct and harder to align later.
_MAX_WORDS = 14

_HEADER = """\
# {clip} — {duration:.0f}s — {count} lines
#
# HOW TO CORRECT
#   Edit the plain lines only. Lines starting with # are the ASR source, kept so
#   you can see what the machine heard. Leave them alone.
#   Play the audio alongside: {media}
#
# THE RULES THAT MATTER (docs/SPELLING-SPEC.md)
#   English stays English:  meeting, percent, separate — never mitting.
#   One word, one spelling, everywhere in this file.
#   Plain ASCII. No accents. No capitals mid-word.
#   Capital at sentence start and for names (Karachi, Ahmed).
#   Drop umm/uhh. Keep acha, matlab, yaani, bas.
#   Numbers: 0–10 as words (teen), 11+ as digits (500 rupay).
#
# This draft IS wrong in places — that is expected. Known habits of the machine:
#   it hallucinates on short segments (بندہ came out as "shayar"),
#   and it mangles English it heard in Urdu script ("spratt" = separate).
"""


@dataclass(slots=True)
class Segment:
    start_ms: int
    end_ms: int
    source: str


def segment_words(words: list[dict[str, Any]]) -> list[Segment]:
    """Group provider words into correctable lines, keeping their timings."""
    segments: list[Segment] = []
    current: list[dict[str, Any]] = []

    def flush() -> None:
        if current:
            segments.append(
                Segment(
                    start_ms=int(current[0]["start"]),
                    end_ms=int(current[-1]["end"]),
                    source=" ".join(str(word.get("text", "")) for word in current),
                )
            )
            current.clear()

    for word in words:
        current.append(word)
        if _SEG_BREAK.search(str(word.get("text", ""))) or len(current) >= _MAX_WORDS:
            flush()
    flush()
    return segments


def main(
    benchmark: str = str(BENCHMARK),
    out_dir: str = "data/eval/drafts",
    provider: str = "assemblyai",
    exclude: str = "punjabi,pa-",
) -> None:
    """Generate one editable draft per clip, plus a manifest."""
    root = Path(benchmark)
    dataset = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    skip_tokens = [token.strip().lower() for token in exclude.split(",") if token.strip()]

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    normalizer = Normalizer(Lexicon.load())

    manifest: list[dict[str, Any]] = []
    skipped: list[str] = []

    for case in dataset["cases"]:
        clip = case["id"]
        haystack = f"{clip} {case.get('category', '')}".lower()
        if any(token in haystack for token in skip_tokens):
            skipped.append(f"{clip} — {case.get('category')}")
            continue

        raw = root / "raw" / f"{clip}-{provider}.json"
        if not raw.exists():
            skipped.append(f"{clip} — no {provider} output")
            continue
        words = json.loads(raw.read_text(encoding="utf-8")).get("words", [])
        if not words:
            skipped.append(f"{clip} — empty transcript")
            continue

        segments = segment_words(words)
        drafts = [
            normalizer.normalize_text(text)
            for text in romanize([segment.source for segment in segments])
        ]

        lines: list[str] = []
        entries: list[dict[str, Any]] = []
        for segment, draft in zip(segments, drafts, strict=True):
            lines += [
                f"[{segment.start_ms / 1000:7.2f} -> {segment.end_ms / 1000:7.2f}]",
                f"#   {segment.source}",
                draft,
                "",
            ]
            entries.append(
                {
                    "start_ms": segment.start_ms,
                    "end_ms": segment.end_ms,
                    "source": segment.source,
                    "draft": draft,
                }
            )

        media = str(root / case["media"])
        (destination / f"{clip}.txt").write_text(
            _HEADER.format(
                clip=clip,
                duration=case["durationMs"] / 1000,
                count=len(entries),
                media=media,
            )
            + "\n"
            + "\n".join(lines)
            + "\n",
            encoding="utf-8",
        )
        manifest.append(
            {
                "id": clip,
                "media": media,
                "category": case.get("category"),
                "duration_ms": case["durationMs"],
                "provider": provider,
                "reference_status": "draft-uncorrected",
                "segments": entries,
            }
        )
        print(f"  {clip:<28} {len(entries):>3} lines")

    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    minutes = sum(item["duration_ms"] for item in manifest) / 60_000
    total = sum(len(item["segments"]) for item in manifest)
    print(f"\n{len(manifest)} clips · {minutes:.1f} min · {total} lines -> {destination}/")
    for item in skipped:
        print(f"  skipped: {item}")
    if normalizer.unknown_tokens:
        print(f"\n{len(normalizer.unknown_tokens)} words not in the lexicon yet — expected.")


if __name__ == "__main__":  # pragma: no cover
    import typer

    typer.run(main)
