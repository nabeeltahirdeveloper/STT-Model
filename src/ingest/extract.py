"""Audio extraction from video via ffmpeg.

STUB -- Phase 1 (prompt 4).

The output format is not negotiable: Qwen3-ASR expects 16 kHz mono 16-bit PCM
(PROJECT.md §4.1). Resampling anywhere else in the pipeline is a bug -- silent
sample-rate mismatches degrade recognition without ever raising.
"""

from __future__ import annotations

from pathlib import Path

from src.config import AudioConfig


def extract_audio(video: Path, out_dir: Path, config: AudioConfig | None = None) -> Path:
    """Extract the audio track of `video` to a WAV in `out_dir`.

    Args:
        video: Source media file, any container ffmpeg can demux.
        out_dir: Destination directory; created if absent.
        config: Target format. Defaults to 16 kHz mono s16.

    Returns:
        Path to the written WAV file.

    Raises:
        NotImplementedError: Phase 1 work.
    """
    raise NotImplementedError("Phase 1: ffmpeg -i <video> -ac 1 -ar 16000 -sample_fmt s16 <out>")


def probe_duration(media: Path) -> float:
    """Duration of `media` in seconds, via ffprobe.

    Needed before alignment, because the aligner caps at 5 minutes and long
    audio has to be chunked with correct offsets (PROJECT.md §4.4).

    Raises:
        NotImplementedError: Phase 1 work.
    """
    raise NotImplementedError("Phase 1")
