"""Decide whether an ambiguous token is English or Roman Urdu, from parallel text.

    uv run python -m scripts.resolve_homographs

ADR-009 left `he`, `me`, `or`, `say`, `no` and their kind without a principled
default, and ADR-011 showed why frequency cannot supply one: Roman-Urdu-Parl is
Roman Urdu that code-switches, so 40,000 hits for `no` says nothing about how
many were English.

The parallel side does supply one. Roman-Urdu-Parl aligns each Roman line with
its Urdu-script original, and **English survives romanization in Latin script
while Urdu does not**. So for any token:

    Roman:  "mujhe no problem hai"
    Urdu:   "مجھے no problem ہے"      -> `no` is English, it stayed Latin
    Roman:  "no bajay aana"
    Urdu:   "نو بجے آنا"               -> `no` is Urdu نو, it is Perso-Arabic

Counting which side of that each occurrence falls on gives an English-share per
token, which is exactly the prior `ambiguous.tsv` needs.

**What this cannot do.** It measures how the *corpus authors* romanized, not
ground truth, and a token absent from the Urdu line entirely (dropped in
translation) is not counted either way. Tokens with few occurrences get a share
computed from a handful of lines and should not be trusted; the report prints
the support count next to every share so a thin one is visible rather than
implied.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src.labeling.lexicon import Lexicon

ROMAN = Path("data/raw/roman-urdu-parl/original_data/roman-urdu.txt")
URDU = Path("data/raw/roman-urdu-parl/original_data/urdu.txt")

_WORD = re.compile(r"[A-Za-z']+")
# Enough occurrences for the share to mean anything.
MIN_SUPPORT = 30
# Above this, treat as English by default; below the mirror, as Urdu.
ENGLISH_THRESHOLD = 0.66
URDU_THRESHOLD = 0.34


@dataclass(frozen=True, slots=True)
class Evidence:
    token: str
    kept_latin: int
    total: int

    @property
    def english_share(self) -> float:
        return self.kept_latin / self.total if self.total else 0.0

    @property
    def verdict(self) -> str:
        if self.total < MIN_SUPPORT:
            return "too few"
        if self.english_share >= ENGLISH_THRESHOLD:
            return "english"
        if self.english_share <= URDU_THRESHOLD:
            return "urdu"
        return "genuinely mixed"


def gather(tokens: set[str], roman: Path, urdu: Path) -> dict[str, Evidence]:
    """For each token, count how often the aligned Urdu line kept it in Latin."""
    kept: Counter[str] = Counter()
    seen: Counter[str] = Counter()
    with (
        roman.open(encoding="utf-8", errors="replace") as left,
        urdu.open(encoding="utf-8", errors="replace") as right,
    ):
        for roman_line, urdu_line in zip(left, right, strict=False):
            present = {word.lower() for word in _WORD.findall(roman_line)} & tokens
            if not present:
                continue
            latin_on_urdu_side = {word.lower() for word in _WORD.findall(urdu_line)}
            for token in present:
                seen[token] += 1
                if token in latin_on_urdu_side:
                    kept[token] += 1
    return {token: Evidence(token, kept[token], seen[token]) for token in tokens}


def main(roman: str = str(ROMAN), urdu: str = str(URDU)) -> None:
    """Report an English-share for every token that is both English and Urdu."""
    left, right = Path(roman), Path(urdu)
    for path in (left, right):
        if not path.exists():
            raise SystemExit(f"corpus not found: {path}")

    lexicon = Lexicon.load()
    # Everything English that also has a Roman Urdu reading, plus the words
    # ADR-009 removed from the variant map -- those are the open question.
    candidates = set(lexicon.english) & (
        set(lexicon.canonical) | set(lexicon.variants) | set(lexicon.ambiguous)
    )
    candidates |= {"he", "me", "or", "say", "no", "the", "they", "main", "hum", "bat", "din"}

    print(f"scanning 6.37M line pairs for {len(candidates)} tokens ...", flush=True)
    evidence = gather(candidates, left, right)

    print(f"\n{'token':<10}{'english %':>11}{'support':>10}   verdict")
    print("-" * 52)
    for item in sorted(evidence.values(), key=lambda e: -e.english_share):
        if not item.total:
            continue
        print(f"{item.token:<10}{item.english_share:>10.1%}{item.total:>10,}   {item.verdict}")
    print("-" * 52)
    tally = Counter(e.verdict for e in evidence.values() if e.total)
    print("  " + " · ".join(f"{k}: {v}" for k, v in sorted(tally.items())))
    print("\n`english` -> default english in ambiguous.tsv")
    print("`urdu`    -> safe to keep as a canonical/variant row")
    print("`genuinely mixed` -> needs a context list, not a default")


if __name__ == "__main__":  # pragma: no cover
    import typer

    typer.run(main)
