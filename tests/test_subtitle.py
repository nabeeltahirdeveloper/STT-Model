"""Golden-file tests for SRT/VTT output, plus the PROJECT.md §9 limit checker.

Golden files rather than assertions on fragments, because subtitle format
regressions are silent: a `.` where a `,` belongs produces a file that opens
without complaint and displays nothing. A byte-for-byte comparison is the only
check that catches that class of bug.

Layout (line breaking, cue grouping) is Phase 3 and its tests are skipped.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import SubtitleConfig
from src.subtitle.layout import break_lines, check_limits, to_captions
from src.subtitle.writer import format_timestamp, to_srt, to_transcript, to_vtt
from src.types import Caption

GOLDEN = Path(__file__).parent / "golden"


# --------------------------------------------------------------------------- #
# Writers — implemented
# --------------------------------------------------------------------------- #
def test_srt_matches_golden(captions: list[Caption]) -> None:
    assert to_srt(captions) == GOLDEN.joinpath("sample.srt").read_text(encoding="utf-8")


def test_vtt_matches_golden(captions: list[Caption]) -> None:
    assert to_vtt(captions) == GOLDEN.joinpath("sample.vtt").read_text(encoding="utf-8")


def test_vtt_starts_with_the_required_header(captions: list[Caption]) -> None:
    assert to_vtt(captions).startswith("WEBVTT\n")


def test_srt_uses_comma_and_vtt_uses_period(captions: list[Caption]) -> None:
    """The one difference that silently breaks players if it is wrong."""
    assert "00:00:02,500" in to_srt(captions)
    assert "00:00:02.500" in to_vtt(captions)


def test_transcript_is_plain_text(captions: list[Caption]) -> None:
    assert to_transcript(captions) == "Aaj meeting hai\nProblem ye hai ke time nahin mila\n"


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.0, "00:00:00,000"),
        (2.5, "00:00:02,500"),
        (61.25, "00:01:01,250"),
        (3661.007, "01:01:01,007"),
    ],
)
def test_format_timestamp(seconds: float, expected: str) -> None:
    assert format_timestamp(seconds) == expected


def test_format_timestamp_truncates_rather_than_rounds() -> None:
    """Rounding up can push a short cue's start past its own end."""
    assert format_timestamp(1.9999) == "00:00:01,999"


def test_format_timestamp_rejects_negative_time() -> None:
    with pytest.raises(ValueError, match="negative timestamp"):
        format_timestamp(-0.1)


def test_empty_caption_list_produces_empty_srt() -> None:
    assert to_srt([]) == ""


# --------------------------------------------------------------------------- #
# Limit checker — implemented ahead of the layout engine it will police
# --------------------------------------------------------------------------- #
def test_well_formed_captions_have_no_violations(captions: list[Caption]) -> None:
    assert check_limits(captions) == []


def test_over_long_line_is_reported() -> None:
    caption = Caption(index=1, start=0.0, end=5.0, lines=["x" * 43])
    assert any("chars > 42" in problem for problem in check_limits([caption]))


def test_too_many_lines_is_reported() -> None:
    caption = Caption(index=1, start=0.0, end=5.0, lines=["a", "b", "c"])
    assert any("3 lines > 2" in problem for problem in check_limits([caption]))


def test_reading_speed_violation_is_reported() -> None:
    """40 chars in 1 second is 40 chars/sec against a limit of 17."""
    caption = Caption(index=1, start=0.0, end=1.0, lines=["x" * 40])
    assert any("chars/sec" in problem for problem in check_limits([caption]))


def test_short_gap_between_captions_is_reported() -> None:
    pair = [
        Caption(index=1, start=0.0, end=2.0, lines=["ek"]),
        Caption(index=2, start=2.01, end=4.0, lines=["do"]),
    ]
    assert any("gap" in problem for problem in check_limits(pair))


def test_limits_come_from_config_not_literals() -> None:
    caption = Caption(index=1, start=0.0, end=5.0, lines=["x" * 43])
    relaxed = SubtitleConfig(max_chars_per_line=50)
    assert check_limits([caption], relaxed) == []


# --------------------------------------------------------------------------- #
# Layout — Phase 3
# --------------------------------------------------------------------------- #
@pytest.mark.skip(reason="Phase 3: needs a Roman Urdu phrase-boundary rule")
def test_break_lines_respects_char_limit() -> None:
    lines = break_lines("Problem ye hai ke aaj meeting ka time nahin mila tha")
    assert all(len(line) <= 42 for line in lines)
    assert len(lines) <= 2


@pytest.mark.skip(reason="Phase 3: postpositions must not be split from their noun")
def test_break_lines_does_not_split_a_postposition_from_its_noun() -> None:
    lines = break_lines("meeting ke baad office se ghar jana hai aaj shaam ko")
    assert not any(line.strip().startswith(("ka ", "ke ", "ko ", "se ", "par ")) for line in lines)


@pytest.mark.skip(reason="Phase 3: cue grouping")
def test_to_captions_obeys_every_limit() -> None:
    assert check_limits(to_captions([])) == []
