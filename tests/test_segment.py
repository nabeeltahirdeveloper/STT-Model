"""Segment bound enforcement (PROJECT.md §4.1: drop < 2 s, split > 35 s)."""

from __future__ import annotations

import pytest

from src.preprocess.segment import enforce_bounds
from src.types import Segment


def test_short_segments_are_dropped(segments: list[Segment]) -> None:
    """The 0.7 s `haan` goes; merging it across a speaker turn would be worse."""
    out = enforce_bounds(segments)
    assert all(segment.duration >= 2.0 for segment in out)
    assert "haan" not in [segment.text for segment in out]


def test_long_segments_are_split(segments: list[Segment]) -> None:
    out = enforce_bounds(segments)
    assert all(segment.duration <= 35.0 + 1e-9 for segment in out)


def test_split_preserves_total_span(segments: list[Segment]) -> None:
    long_one = [Segment(start=10.0, end=100.0)]
    out = enforce_bounds(long_one)
    assert out[0].start == 10.0
    assert out[-1].end == pytest.approx(100.0)


def test_split_pieces_are_contiguous() -> None:
    out = enforce_bounds([Segment(start=0.0, end=90.0)])
    assert all(out[i].end == pytest.approx(out[i + 1].start) for i in range(len(out) - 1))


def test_split_keeps_text_on_the_first_piece_only() -> None:
    """There is no way to divide a transcript by time without alignment."""
    out = enforce_bounds([Segment(start=0.0, end=90.0, text="lamba jawab")])
    assert out[0].text == "lamba jawab"
    assert all(segment.text == "" for segment in out[1:])


def test_speaker_survives_a_split() -> None:
    out = enforce_bounds([Segment(start=0.0, end=90.0, speaker="SPEAKER_00")])
    assert all(segment.speaker == "SPEAKER_00" for segment in out)


def test_empty_input_is_empty_output() -> None:
    assert enforce_bounds([]) == []


def test_inverted_bounds_are_rejected() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        enforce_bounds([], min_s=40.0, max_s=2.0)
