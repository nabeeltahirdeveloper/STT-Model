"""Speaker diarization with pyannote 3.1.

STUB -- Phase 1 (prompt 4).

v1 does not emit speaker-attributed captions (PROJECT.md §1.3); diarization is
used to cut clean segment boundaries at speaker turns, which is what keeps a
caption from spanning two people talking over each other.

Note the pyannote models are gated on Hugging Face -- accept the terms on the
model page and export HF_TOKEN before this will run.
"""

from __future__ import annotations

from pathlib import Path

from src.types import Segment


def diarize(audio: Path, num_speakers: int | None = None) -> list[Segment]:
    """Split `audio` into single-speaker segments, ordered by start time.

    Raises:
        NotImplementedError: Phase 1 work.
    """
    raise NotImplementedError("Phase 1: pyannote/speaker-diarization-3.1")
