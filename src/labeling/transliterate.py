"""Urdu script → Roman, by dictionary lookup.

Replaces the neural romanization path for Urdu-script input (ADR-014). The
neural model produced fluent, confident, wrong words -- `دل` became `shayar`,
`بچانا` became `daman` -- at a rate that made 75% of generated labels unusable.
A lookup table has the one property that matters here: it can only emit a
spelling some human actually wrote for that word. It fails by not knowing a
word, which is visible, rather than by inventing one, which is not.

**Latin runs are never touched.** English arrives in Latin script and leaves in
Latin script, byte for byte (SPELLING-SPEC §5.1). This is the constraint the
whole architecture exists to protect, and here it is free: the lookup only ever
considers tokens containing Urdu characters.

Unknown words are returned unchanged and *reported*. A caller generating
training labels should drop those lines rather than ship a label with Urdu
script in it -- `unknown` is the honest signal that makes that possible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

DICTIONARY = Path("data/lexicon/transliteration.tsv")

# Perso-Arabic ranges. A token containing any of these is Urdu; anything else
# (English, digits, punctuation) passes through untouched.
_URDU = re.compile(r"[؀-ۿݐ-ݿ]")
# Urdu and Latin sentence punctuation, stripped before lookup and restored after.
_EDGE = "۔،؟!.,?:;\"'()[]"


class TransliterationError(RuntimeError):
    """The dictionary could not be loaded."""


@dataclass(frozen=True, slots=True)
class Romanized:
    """The result, with the unknown words that were left in Urdu script."""

    text: str
    unknown: tuple[str, ...]

    @property
    def complete(self) -> bool:
        """Whether every Urdu word was found. False means do not use as a label."""
        return not self.unknown


def load(path: Path | None = None) -> dict[str, str]:
    """Read the TSV built by `scripts/build_translit_dict.py`."""
    source = path or DICTIONARY
    if not source.exists():
        raise TransliterationError(
            f"{source} not found. Build it with: uv run python -m scripts.build_translit_dict"
        )
    table: dict[str, str] = {}
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 2 or parts[0] == "urdu":
            continue
        table[parts[0]] = parts[1]
    if not table:
        raise TransliterationError(f"{source} has no entries")
    return table


def romanize(text: str, table: dict[str, str]) -> Romanized:
    """Romanize Urdu-script words; leave everything else exactly as it is."""
    out: list[str] = []
    unknown: list[str] = []
    for token in text.split():
        if not _URDU.search(token):
            out.append(token)  # §5.1 -- English is never touched
            continue
        # Punctuation is stripped for the lookup and put back, so `ہے۔` finds
        # `ہے` rather than missing and being reported as an unknown word.
        core = token.strip(_EDGE)
        prefix = token[: len(token) - len(token.lstrip(_EDGE))]
        suffix = token[len(token.rstrip(_EDGE)) :]
        roman = table.get(core)
        if roman is None:
            unknown.append(core)
            out.append(token)
        else:
            out.append(f"{prefix}{roman}{suffix}")
    return Romanized(" ".join(out), tuple(unknown))
