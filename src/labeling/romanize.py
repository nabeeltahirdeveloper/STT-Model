"""Urdu script -> Roman Urdu transliteration for label generation.

STUB -- Phase 1. Deliberately not implemented yet.

Romanization is only used to manufacture *training labels* from Urdu-script
corpora (PROJECT.md §5.3). It is never in the inference path: routing
customer-facing text through Perso-Arabic is what mangles English words and is
the reason the two-stage architecture was rejected (ADR-001).

It is blocked on the frequency lexicon. Transliterating before knowing which
spelling Roman-Urdu-Parl actually favours would bake this project's guesses into
the labels, and label spelling is permanent once trained (ADR-004). Build
`scripts/build_lexicon.py` first.
"""

from __future__ import annotations

from pathlib import Path

from src.labeling.lexicon import Lexicon


def urdu_to_roman(text: str, lexicon: Lexicon | None = None) -> str:
    """Transliterate Urdu-script text to Roman Urdu.

    English words embedded in the Urdu text must come back in English
    orthography, not phonetically respelled (SPELLING-SPEC §5.1). The planned
    implementation is IndicXlit for the Urdu spans plus an English passthrough
    guard, followed by `normalize` for canonical spelling.

    Raises:
        NotImplementedError: Phase 1 work.
    """
    raise NotImplementedError(
        "Phase 1: needs the Roman-Urdu-Parl frequency lexicon first "
        "(scripts/build_lexicon.py), otherwise spelling choices are guesses."
    )


def romanize_corpus(src_dir: Path, out_dir: Path) -> int:
    """Romanize every transcript under `src_dir`, writing normalized labels to `out_dir`.

    Returns the number of utterances written.

    Raises:
        NotImplementedError: Phase 1 work.
    """
    raise NotImplementedError("Phase 1: depends on urdu_to_roman")
