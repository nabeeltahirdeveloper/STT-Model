"""Qwen3-ASR transcription.

STUB -- Phase 1+.

Two constraints are already encoded in `ASRConfig` and must not be worked around
here (PROJECT.md §4.3):

* Urdu is not one of Qwen3-ASR's 30 supported languages. Do not pass a language
  token and do not trust the built-in language ID -- it cannot separate Hindi
  from Urdu, and guessing wrong emits Devanagari.
* Streaming mode cannot return timestamps, so captioning is offline-only.

Import torch/qwen_asr inside the function, not at module scope: Phase 0 lint,
typecheck and tests must run on a laptop with neither installed.
"""

from __future__ import annotations

from pathlib import Path

from src.config import ASRConfig
from src.types import Segment


def transcribe(
    audio: Path,
    config: ASRConfig | None = None,
    segments: list[Segment] | None = None,
) -> list[Segment]:
    """Transcribe `audio` to Roman Urdu, one Segment per utterance.

    Args:
        audio: 16 kHz mono 16-bit PCM WAV.
        config: Decoding settings; defaults enforce the language-agnostic prefix.
        segments: Pre-computed diarization bounds. If None, the whole file is
            treated as one utterance, which is only correct for short clips.

    Returns:
        The input segments with `text` populated. Raw model output -- it has NOT
        been through the spelling normalizer yet, and must not be shown to a user
        in this state (CLAUDE.md constraint 3).

    Raises:
        NotImplementedError: Phase 1+ work.
    """
    raise NotImplementedError("Phase 1+: qwen-asr offline decode with language-agnostic prefix")
