"""Turning timed tokens into readable captions.

STUB -- Phase 3.

Correct timestamps do not make good subtitles. The limits in PROJECT.md §9 are
what separate technically-correct output from watchable output:

    42 chars/line · max 2 lines · 1.0-7.0 s · 17 chars/sec · >=100 ms gap
    line breaks at phrase boundaries, never mid-phrase

The phrase-boundary rule is the hard part and the reason this is stubbed rather
than approximated: breaking mid-phrase is more jarring to a reader than a
slightly over-long line, so a greedy word-wrap would be worse than nothing.
Roman Urdu also needs its own notion of a phrase boundary -- postpositions
(`ka`, `ke`, `ko`, `se`, `par`) bind to the preceding noun and must not be
split from it.
"""

from __future__ import annotations

from src.config import SubtitleConfig
from src.types import Caption, TimedToken


def break_lines(text: str, config: SubtitleConfig | None = None) -> list[str]:
    """Break `text` into at most `max_lines` lines at phrase boundaries.

    Raises:
        NotImplementedError: Phase 3 work.
    """
    raise NotImplementedError("Phase 3: needs a Roman Urdu phrase-boundary rule")


def to_captions(tokens: list[TimedToken], config: SubtitleConfig | None = None) -> list[Caption]:
    """Group timed tokens into caption cues obeying the PROJECT.md §9 limits.

    Raises:
        NotImplementedError: Phase 3 work.
    """
    raise NotImplementedError("Phase 3")


def check_limits(captions: list[Caption], config: SubtitleConfig | None = None) -> list[str]:
    """Report every §9 limit a caption list violates.

    Implemented ahead of the layout engine on purpose: this is the oracle that
    the layout engine will be tested against, and writing the checker first keeps
    the limits from quietly bending to whatever the implementation happens to do.
    """
    config = config or SubtitleConfig()
    problems: list[str] = []

    for index, caption in enumerate(captions):
        where = f"caption {caption.index}"
        if len(caption.lines) > config.max_lines:
            problems.append(f"{where}: {len(caption.lines)} lines > {config.max_lines}")
        for line_no, line in enumerate(caption.lines, start=1):
            if len(line) > config.max_chars_per_line:
                problems.append(
                    f"{where} line {line_no}: {len(line)} chars > {config.max_chars_per_line}"
                )
        if caption.duration < config.min_duration_s:
            problems.append(f"{where}: {caption.duration:.2f}s < {config.min_duration_s}s")
        if caption.duration > config.max_duration_s:
            problems.append(f"{where}: {caption.duration:.2f}s > {config.max_duration_s}s")
        if caption.chars_per_second > config.max_chars_per_second:
            problems.append(
                f"{where}: {caption.chars_per_second:.1f} chars/sec > {config.max_chars_per_second}"
            )
        if index + 1 < len(captions):
            gap = captions[index + 1].start - caption.end
            if gap < config.min_gap_s:
                problems.append(
                    f"{where}: gap {gap * 1000:.0f}ms < {config.min_gap_s * 1000:.0f}ms"
                )

    return problems
