"""Music and noise separation with Demucs.

STUB -- Phase 1 (prompt 4).

Real target content is vlogs, dramas and street interviews (PROJECT.md §5.1),
which means background music over speech far more often than in read-speech
benchmarks. UrduSpeech used Demucs on this exact audio type, which is why it is
the default here rather than a spectral gate.
"""

from __future__ import annotations

from pathlib import Path


def remove_music(audio: Path, out_dir: Path | None = None) -> Path:
    """Return a path to the vocals-only stem of `audio`.

    Raises:
        NotImplementedError: Phase 1 work.
    """
    raise NotImplementedError("Phase 1: demucs --two-stems=vocals")
