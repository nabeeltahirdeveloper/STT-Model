"""Chunk reassembly for audio longer than the aligner's 5-minute cap.

Qwen3-ForcedAligner hard-caps at 300 seconds (PROJECT.md §4.4), so long video is
aligned in chunks and stitched back together. Every timestamp the aligner returns
is chunk-relative; forgetting to add the chunk offset produces subtitles that are
perfectly synced for the first five minutes and progressively wrong after that --
a bug that passes every short-clip test.

`chunk_spans` is implemented and tested. `reassemble` is xfail: it is blocked on
the §4.4 experiment, because the `bridge` strategy stitches Urdu-script timings
onto Roman tokens and reassembles differently from `direct`.
"""

from __future__ import annotations

import pytest

from src.inference.align import chunk_spans, reassemble
from src.types import TimedToken

FIVE_MINUTES = 300.0


# --------------------------------------------------------------------------- #
# Chunking — implemented
# --------------------------------------------------------------------------- #
def test_short_audio_is_one_chunk() -> None:
    assert chunk_spans(120.0, FIVE_MINUTES) == [(0.0, 120.0)]


def test_audio_over_five_minutes_is_split() -> None:
    assert chunk_spans(700.0, FIVE_MINUTES) == [(0.0, 300.0), (300.0, 600.0), (600.0, 700.0)]


def test_chunks_are_contiguous_and_cover_the_whole_file() -> None:
    spans = chunk_spans(1234.5, FIVE_MINUTES)
    assert spans[0][0] == 0.0
    assert spans[-1][1] == 1234.5
    assert all(spans[i][1] == spans[i + 1][0] for i in range(len(spans) - 1))


def test_no_chunk_exceeds_the_model_cap() -> None:
    assert all(end - start <= FIVE_MINUTES for start, end in chunk_spans(3600.0, FIVE_MINUTES))


def test_exact_multiple_produces_no_empty_trailing_chunk() -> None:
    assert chunk_spans(600.0, FIVE_MINUTES) == [(0.0, 300.0), (300.0, 600.0)]


def test_zero_duration_produces_no_chunks() -> None:
    assert chunk_spans(0.0, FIVE_MINUTES) == []


def test_non_positive_cap_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        chunk_spans(100.0, 0.0)


# --------------------------------------------------------------------------- #
# Reassembly — Phase 1+, blocked on the §4.4 aligner experiment
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(raises=NotImplementedError, reason="Phase 1+: blocked on PROJECT.md §4.4")
def test_offsets_are_added_back_to_chunk_relative_timings() -> None:
    """A 10-minute file: the second chunk's `0.5 s` is really `300.5 s`."""
    spans = chunk_spans(600.0, FIVE_MINUTES)
    chunked = [
        [TimedToken("aaj", 0.5, 1.0), TimedToken("meeting", 1.0, 1.8)],
        [TimedToken("problem", 0.5, 1.2), TimedToken("nahin", 1.2, 1.7)],
    ]
    tokens = reassemble(chunked, spans)
    assert [token.start for token in tokens] == [0.5, 1.0, 300.5, 301.2]


@pytest.mark.xfail(raises=NotImplementedError, reason="Phase 1+: blocked on PROJECT.md §4.4")
def test_reassembled_timings_are_monotonic() -> None:
    spans = chunk_spans(900.0, FIVE_MINUTES)
    chunked = [[TimedToken(f"tok{i}", 1.0, 2.0)] for i in range(len(spans))]
    tokens = reassemble(chunked, spans)
    assert all(tokens[i].start < tokens[i + 1].start for i in range(len(tokens) - 1))


@pytest.mark.xfail(raises=NotImplementedError, reason="Phase 1+: blocked on PROJECT.md §4.4")
def test_reassembly_rejects_a_chunk_count_mismatch() -> None:
    with pytest.raises(ValueError):
        reassemble([[TimedToken("a", 0.0, 1.0)]], chunk_spans(900.0, FIVE_MINUTES))


@pytest.mark.xfail(raises=NotImplementedError, reason="Phase 1+: blocked on PROJECT.md §4.4")
def test_timing_error_stays_under_the_perceptual_threshold() -> None:
    """PROJECT.md §1.4 targets < 200 ms. Offset drift blows straight through it."""
    spans = chunk_spans(1800.0, FIVE_MINUTES)
    chunked = [[TimedToken("x", 0.0, 1.0)] for _ in spans]
    tokens = reassemble(chunked, spans)
    assert abs(tokens[-1].start - spans[-1][0]) < 0.2
