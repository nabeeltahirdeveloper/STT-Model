"""Shared fixtures.

The lexicon fixtures load the *real* tracked lexicons rather than a miniature
fake. The lexicon is a product artifact, and a test that passes against a
made-up lexicon tells you nothing about whether shipping output is correct.
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import pytest

from src.config import LexiconConfig
from src.labeling.lexicon import Lexicon
from src.labeling.normalize import Normalizer
from src.types import Caption, Segment

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def lexicon() -> Lexicon:
    """The tracked lexicon, loaded once per session."""
    return Lexicon.load(LexiconConfig(), REPO_ROOT)


@pytest.fixture
def normalizer(lexicon: Lexicon) -> Normalizer:
    """A fresh Normalizer per test, so the unknown-token ledger does not leak."""
    return Normalizer(lexicon)


@pytest.fixture
def segments() -> list[Segment]:
    """A short diarization result: two speakers, one too-short segment, one too-long."""
    return [
        Segment(start=0.0, end=4.5, text="Aaj meeting hai", speaker="SPEAKER_00"),
        Segment(start=4.5, end=5.2, text="haan", speaker="SPEAKER_01"),  # < 2s, dropped
        Segment(start=5.2, end=48.0, text="lamba jawab", speaker="SPEAKER_00"),  # > 35s, split
        Segment(start=48.0, end=52.0, text="theek hai", speaker="SPEAKER_01"),
    ]


@pytest.fixture
def captions() -> list[Caption]:
    """Two well-formed cues, inside every PROJECT.md §9 limit."""
    return [
        Caption(index=1, start=0.0, end=2.5, lines=["Aaj meeting hai"]),
        Caption(index=2, start=2.75, end=6.0, lines=["Problem ye hai ke", "time nahin mila"]),
    ]


@pytest.fixture
def fake_audio(tmp_path: Path) -> Path:
    """A 0.5 s silent 16 kHz mono 16-bit WAV — the format the pipeline expects.

    Real enough for path handling, duration probing and format assertions;
    useless for anything acoustic, which is the point.
    """
    path = tmp_path / "silence.wav"
    frames = 8_000
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(struct.pack(f"<{frames}h", *([0] * frames)))
    return path
