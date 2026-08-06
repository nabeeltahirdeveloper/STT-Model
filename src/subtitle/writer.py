"""SRT and VTT serialization.

Implemented, unlike the rest of the subtitle stage. It is pure string formatting
with no model dependency, and subtitle format regressions are silent -- a
malformed timestamp separator produces a file that opens fine and shows nothing.
Golden-file tests in `tests/test_subtitle.py` are the guard.

The two formats differ in exactly three ways: the `WEBVTT` header, `,` versus
`.` as the millisecond separator, and VTT cue identifiers being optional.
"""

from __future__ import annotations

from src.types import Caption


def format_timestamp(seconds: float, *, millis_separator: str = ",") -> str:
    """Format seconds as `HH:MM:SS<sep>mmm`.

    Truncates rather than rounds: rounding up can push a cue's start past its own
    end for very short captions.
    """
    if seconds < 0:
        raise ValueError(f"negative timestamp: {seconds}")
    total_ms = int(seconds * 1000)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{millis_separator}{millis:03d}"


def to_srt(captions: list[Caption]) -> str:
    """Serialize captions to SubRip. Cue numbers are 1-based and contiguous."""
    blocks: list[str] = []
    for number, caption in enumerate(captions, start=1):
        start = format_timestamp(caption.start, millis_separator=",")
        end = format_timestamp(caption.end, millis_separator=",")
        blocks.append(f"{number}\n{start} --> {end}\n{caption.text}\n")
    return "\n".join(blocks)


def to_vtt(captions: list[Caption]) -> str:
    """Serialize captions to WebVTT."""
    blocks = ["WEBVTT\n"]
    for caption in captions:
        start = format_timestamp(caption.start, millis_separator=".")
        end = format_timestamp(caption.end, millis_separator=".")
        blocks.append(f"{start} --> {end}\n{caption.text}\n")
    return "\n".join(blocks)


def to_transcript(captions: list[Caption]) -> str:
    """Plain-text transcript with no timing (PROJECT.md §1.2 output formats)."""
    return "\n".join(caption.text.replace("\n", " ") for caption in captions) + "\n"
