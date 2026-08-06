"""Config validation tests.

The validators here are guards against expensive mistakes, not schema
decoration: training on the eval set invalidates every number the project
reports, and forcing a language token emits Devanagari. Both fail at load time.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import (
    ASRConfig,
    DataConfig,
    InferenceConfig,
    RunConfig,
    TrainingConfig,
    load_inference_config,
    load_run_config,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("name", ["phase1.yaml", "phase2.yaml"])
def test_shipped_training_configs_load(name: str) -> None:
    config = load_run_config(CONFIGS / name)
    assert config.training.full_finetune is True
    assert config.training.use_lora is False


def test_shipped_inference_config_loads() -> None:
    config = load_inference_config(CONFIGS / "inference.yaml")
    assert config.aligner.max_chunk_seconds == 300.0
    assert config.asr.force_language is None


def test_phase1_matches_srota_hyperparameters() -> None:
    """PROJECT.md §3.2 — the only recipe with evidence behind it."""
    training = load_run_config(CONFIGS / "phase1.yaml").training
    assert training.learning_rate == 2e-5
    assert training.lr_schedule == "linear"
    assert training.warmup_ratio == 0.02
    assert training.effective_batch_size == 32
    assert training.epochs == 2
    assert training.precision == "bf16"


def test_eval_set_cannot_be_used_for_training() -> None:
    with pytest.raises(ValidationError, match="never trained on"):
        DataConfig(manifests=[Path("data/eval/manifest.jsonl")])


def test_lora_without_an_ab_test_is_rejected() -> None:
    with pytest.raises(ValidationError, match="A/B against full fine-tuning"):
        TrainingConfig(use_lora=True, full_finetune=False)


def test_lora_is_allowed_once_ab_tested() -> None:
    config = TrainingConfig(use_lora=True, full_finetune=False, lora_ab_tested=True)
    assert config.use_lora is True


def test_batch_maths_must_add_up() -> None:
    with pytest.raises(ValidationError, match="effective_batch_size"):
        TrainingConfig(per_device_batch_size=4, gradient_accumulation_steps=4)


def test_forcing_a_language_is_rejected() -> None:
    with pytest.raises(ValidationError, match="not a Qwen3-ASR language"):
        ASRConfig(force_language="Urdu")


def test_streaming_is_rejected_because_it_cannot_emit_timestamps() -> None:
    with pytest.raises(ValidationError, match="cannot emit timestamps"):
        ASRConfig(streaming=True)


def test_unknown_config_keys_are_an_error() -> None:
    with pytest.raises(ValidationError):
        InferenceConfig.model_validate({"aligner": {"strategy": "direct", "typo": 1}})


def test_config_hash_is_stable_and_sensitive() -> None:
    base = RunConfig(name="x", phase=1)
    assert base.config_hash == RunConfig(name="x", phase=1).config_hash
    assert base.config_hash != RunConfig(name="x", phase=1, seed=7).config_hash


def test_configs_are_frozen() -> None:
    config = RunConfig(name="x", phase=1)
    with pytest.raises(ValidationError):
        config.seed = 99
