"""Forced alignment: text + audio -> word-level timestamps.

MOSTLY STUB -- the strategy is unresolved and it is the highest-priority
experiment in the project (PROJECT.md §4.4, open question 3).

Neither Urdu nor Hindi is among Qwen3-ForcedAligner's 11 supported languages, so
three strategies are ordered by cost, to be tried in order:

1. `direct` -- feed Roman Urdu with `language="English"`. The text is already
   Latin and the aligner matches graphemes to acoustics. 30 minutes to test.
   `scripts/test_aligner.py` runs exactly this experiment.
2. `bridge` -- align the Urdu-script transcript, carry the timings across to the
   Roman tokens. Romanization is word-for-word so the timings transfer. This is
   the *only* sanctioned use of Urdu script, and it is for timing, never text.
3. `mms` -- MMS + torchaudio CTC. Lower accuracy, definitely covers Urdu.

`chunk_spans` is implemented because the 5-minute cap is a hard fact about the
model rather than a strategy choice, and getting offsets wrong is the classic
source of drift that only shows up an hour into a long video.
"""

from __future__ import annotations

from pathlib import Path

from src.config import AlignerConfig
from src.types import Segment, TimedToken


def chunk_spans(duration: float, max_chunk_seconds: float) -> list[tuple[float, float]]:
    """Split `duration` seconds into (start, end) spans no longer than the cap.

    Returns absolute offsets into the source media. Every downstream timestamp is
    chunk-relative and must have its span start added back -- see `reassemble`.
    """
    if duration <= 0:
        return []
    if max_chunk_seconds <= 0:
        raise ValueError("max_chunk_seconds must be positive")

    spans: list[tuple[float, float]] = []
    start = 0.0
    while start < duration:
        end = min(start + max_chunk_seconds, duration)
        spans.append((start, end))
        start = end
    return spans


def reassemble(
    chunked: list[list[TimedToken]],
    spans: list[tuple[float, float]],
) -> list[TimedToken]:
    """Shift each chunk's chunk-relative timestamps back into media time.

    Raises:
        NotImplementedError: Phase 1+ work -- gated on the §4.4 experiment, since
            the bridge strategy reassembles differently from the direct one.
    """
    raise NotImplementedError("Phase 1+: blocked on the §4.4 aligner experiment")


def align(
    audio: Path,
    segments: list[Segment],
    config: AlignerConfig | None = None,
) -> list[TimedToken]:
    """Produce word-level timings for the text in `segments`.

    Raises:
        NotImplementedError: Phase 1+ work. Run `scripts/test_aligner.py` first --
            it decides which strategy this function implements.
    """
    raise NotImplementedError("Phase 1+: run scripts/test_aligner.py to resolve PROJECT.md §4.4")
