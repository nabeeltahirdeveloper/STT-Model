"""Saving a trained checkpoint must not be blocked by the base model's config.

Qwen3-ASR ships `generation_config.json` with `temperature=1e-06` next to
`do_sample=False`. transformers validates that strictly inside
`save_pretrained`, so a Colab run trained to completion and then could not write
its checkpoint -- the one failure mode that costs a whole session rather than a
minute, because it lands after the work rather than before it.

These run against the real GenerationConfig, not a stub: the contract under test
belongs to transformers, so a stub would only assert my reading of it.
"""

from __future__ import annotations

import tempfile

import pytest
from transformers import GenerationConfig

from src.training.finetune import sanitize_generation_config


def test_the_shipped_qwen_config_cannot_be_saved_before_the_fix() -> None:
    """The regression itself. If this stops raising, the fix is obsolete."""
    config = GenerationConfig(do_sample=False, temperature=1e-06)
    with pytest.raises(ValueError, match="temperature"):
        config.validate(strict=True)


def test_after_sanitizing_it_validates_and_saves() -> None:
    config = GenerationConfig(do_sample=False, temperature=1e-06)
    assert sanitize_generation_config(config) == ["temperature=1e-06"]
    config.validate(strict=True)
    with tempfile.TemporaryDirectory() as directory:
        config.save_pretrained(directory)


@pytest.mark.parametrize(
    ("field", "value"),
    [("temperature", 1e-06), ("top_p", 0.9), ("top_k", 10), ("typical_p", 0.5)],
)
def test_each_sampling_field_is_reset_when_greedy(field: str, value: float) -> None:
    config = GenerationConfig(do_sample=False, **{field: value})
    assert sanitize_generation_config(config) == [f"{field}={value!r}"]
    config.validate(strict=True)


def test_a_real_sampling_config_is_left_alone() -> None:
    """do_sample=True means these values are load-bearing, not leftovers."""
    config = GenerationConfig(do_sample=True, temperature=0.7, top_p=0.9)
    assert sanitize_generation_config(config) == []
    assert config.temperature == 0.7
    assert config.top_p == 0.9


def test_an_already_clean_greedy_config_reports_no_change() -> None:
    config = GenerationConfig(do_sample=False)
    assert sanitize_generation_config(config) == []
    config.validate(strict=True)


def test_reports_every_offending_field_at_once() -> None:
    """The message is what the operator reads; it must not stop at the first."""
    config = GenerationConfig(do_sample=False, temperature=1e-06, top_p=0.5)
    changed = sanitize_generation_config(config)
    assert len(changed) == 2
    assert any("temperature" in c for c in changed)
    assert any("top_p" in c for c in changed)
    config.validate(strict=True)
