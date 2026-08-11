"""Build correctable Roman Urdu drafts from the UrduSpeech benchmark.

    uv run python -m scripts.build_eval_from_corpus --minutes-per-category 3

Different input to `build_eval_draft.py`, and a better one. That script starts
from ASR output, so a human has to fix both what the machine *heard* and how it
was spelled. This one starts from UrduSpeech's human-verified transcripts
(PROJECT.md §6.1: "plus the 9-hour UrduSpeech benchmark, romanized identically"),
so the words are already right and only the romanization needs review.

The corpus is CC-BY-4.0 — commercial use is permitted with attribution, which
resolves the licence half of gate R1.

Selection is balanced by category on purpose. 44 minutes of podcast tells you
how the model does on podcasts and nothing else; the same effort spread across
comedy, drama, news and vlogs tells you where it falls over. Categories differ
enormously in difficulty — drama has overlapping speech, news is clean read
speech, roadside interviews are noisy.

Output is one file per category, which is the unit a reviewer can finish in a
sitting. Still a draft: no line counts as reference until a human has heard the
audio and signed off (CLAUDE.md constraint 5).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.romanize_via_opencut import romanize
from src.labeling.lexicon import Lexicon
from src.labeling.normalize import Normalizer

BENCHMARK = Path("data/raw/urduspeech/benchmark/US-benchmark-CS")

_HEADER = """\
# {category} — {clips} clips — {minutes:.1f} min of audio
#
# HOW TO CORRECT
#   Edit the plain lines only. Lines starting with # are the corpus transcript
#   (human-verified Urdu) — leave them alone, they are your reference for what
#   was actually said. The [path] line above each pair is the audio to play.
#
# WHAT YOU ARE FIXING
#   The WORDS are already correct — a human verified them. You are only fixing
#   how they are SPELLED in Latin letters. If a word sounds right but is spelled
#   oddly, fix the spelling. If a word is simply wrong, fix it and put  ??  at
#   the end of the line so we can count how often the corpus itself is wrong.
#
# THE RULES (docs/SPELLING-SPEC.md)
#   English stays English:  meeting, percent, anesthesia — never mitting.
#   One word, one spelling, everywhere in this file and every other file.
#   Plain ASCII. No accents. No capitals mid-word.
#   Capital at sentence start and for names (Karachi, Ahmed, Pakistan).
#   Drop umm/uhh. Keep acha, matlab, yaani, bas.
#   Numbers: 0–10 as words (teen), 11+ as digits (500 rupay).
#   Canonical forms: hai, hain, nahin, mein, ye, wo, ke, kya, bohot, acha
#
# KNOWN MACHINE HABITS — these are observed, not hypothetical. Hunt for them.
#   1. ENGLISH RE-SPELLED PHONETICALLY — the worst error, fix every one.
#        fitness -> "fitnes"      percent -> "parsent"
#      If the # line has a Latin-script word, the draft must copy it EXACTLY.
#   2. DROPPED VOWELS in names and rarer words.
#        بسنت -> "Bsnt"  should be  Basant
#   3. WRONG WORD SUBSTITUTED — the draft is fluent but says something else.
#        "کروں" -> "mein"  should be  karun
#      This is why you read the # line. Fluent output is not correct output.
#   4. ACRONYMS AND NUMBERS mangled by spacing/case.
#        JF-17 -> "Jf - 17"  should be  JF-17
#   5. Trailing punctuation and filler (hmm, umm) silently dropped — fine,
#      leave them dropped, but make sure a real word was not dropped with them.
"""


@dataclass(slots=True)
class Clip:
    category: str
    length: str
    audio: Path
    transcript: str
    duration_s: float


def load_clips(root: Path) -> list[Clip]:
    """Pair every transcript row with the audio file we actually downloaded."""
    clips: list[Clip] = []
    for jsonl in sorted(root.rglob("clean_transcription.jsonl")):
        category = jsonl.parent.name
        length = jsonl.parent.parent.name
        audio_dir = jsonl.parent / "audio"
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            path = audio_dir / str(row.get("Audio_Clip", ""))
            if not path.exists():
                continue  # transcript row for a clip we did not download
            text = str(row.get("ground_truth") or row.get("Transcription") or "").strip()
            if not text:
                continue
            try:
                duration = float(row.get("Duration_seconds") or 0)
            except (TypeError, ValueError):
                duration = 0.0
            clips.append(Clip(category, length, path, text, duration))
    return clips


def select(clips: list[Clip], minutes_per_category: float) -> list[Clip]:
    """Take a per-category budget, shortest clips first.

    Short clips maximise coverage per minute of reviewing: ten 8-second clips
    from ten speakers teach more about the model's weaknesses than one
    80-second monologue. Categories with less audio than the budget simply
    contribute everything they have.
    """
    by_category: dict[str, list[Clip]] = {}
    for clip in clips:
        by_category.setdefault(clip.category, []).append(clip)

    chosen: list[Clip] = []
    for category in sorted(by_category):
        budget = minutes_per_category * 60
        for clip in sorted(by_category[category], key=lambda item: item.duration_s):
            if budget <= 0:
                break
            chosen.append(clip)
            budget -= clip.duration_s or 8.0
    return chosen


def main(
    corpus: str = str(BENCHMARK),
    out_dir: str = "data/eval/drafts-corpus",
    minutes_per_category: float = 3.0,
) -> None:
    """Romanize a balanced slice of the corpus into per-category draft files."""
    clips = load_clips(Path(corpus))
    if not clips:
        raise SystemExit(f"no transcript/audio pairs under {corpus}")

    chosen = select(clips, minutes_per_category)
    print(f"{len(clips)} clips available; selected {len(chosen)}")

    print("romanizing via OpenCut (model loads once, then batches)...")
    romanized = romanize([clip.transcript for clip in chosen])

    normalizer = Normalizer(Lexicon.load())
    drafts = [normalizer.normalize_text(text) for text in romanized]

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[tuple[Clip, str]]] = {}
    for clip, draft in zip(chosen, drafts, strict=True):
        grouped.setdefault(clip.category, []).append((clip, draft))

    manifest: list[dict[str, Any]] = []
    for category, rows in sorted(grouped.items()):
        minutes = sum(clip.duration_s for clip, _ in rows) / 60
        lines: list[str] = []
        for clip, draft in rows:
            lines += [f"[{clip.audio}]", f"#   {clip.transcript}", draft, ""]
            manifest.append(
                {
                    "category": category,
                    "length": clip.length,
                    "audio": str(clip.audio),
                    "duration_s": clip.duration_s,
                    "corpus_transcript": clip.transcript,
                    "draft": draft,
                    "reference_status": "draft-uncorrected",
                }
            )
        (destination / f"{category}.txt").write_text(
            _HEADER.format(category=category, clips=len(rows), minutes=minutes)
            + "\n"
            + "\n".join(lines)
            + "\n",
            encoding="utf-8",
        )
        print(f"  {category:<18}{len(rows):>4} clips {minutes:>6.1f} min")

    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    total = sum(item["duration_s"] for item in manifest) / 60
    print(
        f"\n{len(manifest)} clips · {total:.1f} min · {len(grouped)} categories -> {destination}/"
    )


if __name__ == "__main__":  # pragma: no cover
    import typer

    typer.run(main)
