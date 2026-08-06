"""Segment bound enforcement.

Implemented, because it is pure list arithmetic with no model dependency and the
alignment tests need it. Bounds come from PROJECT.md §4.1: drop anything under
2 s, split anything over 35 s.
"""

from __future__ import annotations

from src.types import Segment

MIN_SECONDS = 2.0
MAX_SECONDS = 35.0


def enforce_bounds(
    segments: list[Segment],
    min_s: float = MIN_SECONDS,
    max_s: float = MAX_SECONDS,
) -> list[Segment]:
    """Drop too-short segments and split too-long ones.

    Short segments are dropped rather than merged because merging across a
    speaker turn produces a segment the ASR cannot transcribe cleanly. Long
    segments are split evenly rather than at silence: silence detection belongs
    upstream in diarization, and an even split keeps this function pure.

    Splitting carries `text` on the first piece only -- there is no way to divide
    a transcript by time without alignment, and guessing would corrupt labels.
    """
    if min_s > max_s:
        raise ValueError(f"min_s ({min_s}) must not exceed max_s ({max_s})")

    out: list[Segment] = []
    for segment in segments:
        if segment.duration < min_s:
            continue
        if segment.duration <= max_s:
            out.append(segment)
            continue
        out.extend(_split(segment, max_s))
    return out


def _split(segment: Segment, max_s: float) -> list[Segment]:
    pieces = int(segment.duration // max_s) + (1 if segment.duration % max_s else 0)
    step = segment.duration / pieces
    return [
        Segment(
            start=segment.start + index * step,
            end=min(segment.start + (index + 1) * step, segment.end),
            text=segment.text if index == 0 else "",
            speaker=segment.speaker,
        )
        for index in range(pieces)
    ]
