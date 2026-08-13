"""Audio loading for the training loop.

torchaudio 2.11 rerouted `torchaudio.load` through TorchCodec, which is a
separate package and was not installed on Colab: the first training step of the
session died on the import, 64 minutes after the download that fed it started.
These tests pin the properties the training loop depends on, against real files
written to disk rather than mocks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile
import torch

from src.training.finetune import load_audio


def _write(path: Path, data: np.ndarray, rate: int, subtype: str = "PCM_16") -> Path:
    soundfile.write(str(path), data, rate, subtype=subtype)
    return path


def test_mono_wav_keeps_its_samples(tmp_path: Path) -> None:
    tone = np.sin(np.linspace(0, 20, 16_000)).astype(np.float32)
    waveform, rate = load_audio(_write(tmp_path / "m.wav", tone, 16_000))
    assert rate == 16_000
    assert waveform.shape == (1, 16_000)
    assert waveform.dtype == torch.float32


def test_stereo_is_averaged_to_one_channel(tmp_path: Path) -> None:
    """The processor takes a single channel; a stereo clip must not reach it."""
    # 0.5 rather than 1.0: +1.0/-1.0 sit on the edge of the 16-bit range and do
    # not cancel exactly, which says nothing about the averaging.
    stereo = np.stack([np.full(800, 0.5, np.float32), np.full(800, -0.5, np.float32)], axis=1)
    waveform, _ = load_audio(_write(tmp_path / "s.wav", stereo, 16_000))
    assert waveform.shape == (1, 800)
    assert torch.allclose(waveform, torch.zeros(1, 800), atol=1e-4)


def test_the_real_sample_rate_is_reported(tmp_path: Path) -> None:
    """The caller resamples off this value; a wrong rate silently warps pitch."""
    _, rate = load_audio(_write(tmp_path / "8k.wav", np.zeros(800, np.float32), 8_000))
    assert rate == 8_000


def test_pcm16_is_normalised_to_minus_one_to_one(tmp_path: Path) -> None:
    """16-bit PCM is what this corpus ships. Integer samples would blow up the loss."""
    loud = np.array([1.0, -1.0, 0.5, -0.5], np.float32)
    waveform, _ = load_audio(_write(tmp_path / "p.wav", loud, 16_000))
    assert waveform.abs().max() <= 1.0
    assert waveform.abs().max() > 0.9


def test_output_feeds_the_resampler_unchanged(tmp_path: Path) -> None:
    """The regression path end to end: load, then resample, as training does."""
    torchaudio = pytest.importorskip("torchaudio")
    waveform, rate = load_audio(_write(tmp_path / "r.wav", np.zeros(8_000, np.float32), 8_000))
    resampled = torchaudio.functional.resample(waveform, rate, 16_000)
    assert resampled.shape == (1, 16_000)


def test_squeeze_to_numpy_matches_what_the_processor_is_given(tmp_path: Path) -> None:
    waveform, _ = load_audio(_write(tmp_path / "q.wav", np.zeros(1_600, np.float32), 16_000))
    audio = waveform.squeeze(0).numpy()
    assert audio.ndim == 1
    assert audio.shape == (1_600,)
