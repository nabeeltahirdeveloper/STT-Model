"""Typed configuration, loaded from YAML under `configs/`.

Nothing in this project reads a hyperparameter from a literal in a script
(CLAUDE.md > Code conventions). Two reasons it is worth the ceremony: every
training run must be reproducible from its config hash, and several of the
constraints below are not preferences but product requirements -- the eval-set
guard and the LoRA guard exist so a wrong config fails at load time rather than
after four hours of GPU spend.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# The eval set is never trained on -- no exceptions (CLAUDE.md constraint 5).
EVAL_DIR = "data/eval"


class _Base(BaseModel):
    """Reject unknown keys, so a typo in YAML is an error and not a silent default."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class AudioConfig(_Base):
    """Target audio format. Qwen3-ASR expects 16 kHz mono 16-bit PCM."""

    sample_rate: int = 16_000
    channels: int = 1
    sample_format: str = "s16"


class SegmentationConfig(_Base):
    """Segment bounds from PROJECT.md §4.1: drop under 2 s, split over 35 s."""

    min_seconds: float = 2.0
    max_seconds: float = 35.0
    remove_music: bool = True
    diarize: bool = True


class SubtitleConfig(_Base):
    """Caption formatting limits (PROJECT.md §9).

    Timestamps alone do not make readable captions; these are the difference
    between correct output and usable output.
    """

    max_chars_per_line: int = 42
    max_lines: int = 2
    min_duration_s: float = 1.0
    max_duration_s: float = 7.0
    max_chars_per_second: float = 17.0
    min_gap_s: float = 0.100


class LexiconConfig(_Base):
    """Where the spelling lexicons live. These are tracked in git, not generated."""

    canonical: Path = Path("data/lexicon/canonical.tsv")
    english: Path = Path("data/lexicon/english.txt")
    ambiguous: Path = Path("data/lexicon/ambiguous.tsv")
    acronyms: Path = Path("data/lexicon/acronyms.txt")


AlignerStrategy = Literal["direct", "bridge", "mms"]


class AlignerConfig(_Base):
    """Forced alignment settings.

    `strategy` follows the ordered fallback list in PROJECT.md §4.4. `direct`
    (Roman text, language="English") is untested as of Phase 0 -- that experiment
    is `scripts/test_aligner.py` and it gates the timestamp design.
    """

    model_id: str = "Qwen/Qwen3-ForcedAligner-0.6B"
    strategy: AlignerStrategy = "direct"
    # The aligner hard-caps at 5 minutes; longer audio is chunked and reassembled.
    max_chunk_seconds: float = 300.0
    language: str = "English"


class ASRConfig(_Base):
    """ASR decoding settings.

    Urdu is not one of Qwen3-ASR's supported languages (PROJECT.md §4.3), so we
    never force a language token and never trust the built-in language ID. The
    language-agnostic prefix is what Srota used.
    """

    model_id: str = "Qwen/Qwen3-ASR-1.7B"
    # Srota's language-agnostic decoding prefix (PROJECT.md §3.2).
    decoding_prefix: str = "language None<asr_text>"
    force_language: str | None = None
    trust_language_id: bool = False
    streaming: bool = False

    @model_validator(mode="after")
    def _reject_forced_language(self) -> ASRConfig:
        if self.force_language is not None:
            raise ValueError(
                "force_language is not supported: Urdu is not a Qwen3-ASR language and the "
                "built-in language ID cannot separate Hindi from Urdu (PROJECT.md §4.3). "
                "Use the language-agnostic decoding prefix instead."
            )
        if self.streaming:
            raise ValueError(
                "streaming mode cannot emit timestamps (PROJECT.md §4.4); captions are "
                "offline-only in v1."
            )
        return self


class TrainingConfig(_Base):
    """Fine-tuning hyperparameters.

    Defaults are Srota's (PROJECT.md §3.2), which is the closest thing to a
    proven recipe for this exact problem shape.
    """

    base_model: str = "Qwen/Qwen3-ASR-0.6B"
    full_finetune: bool = True
    use_lora: bool = False
    lora_ab_tested: bool = False
    optimizer: str = "adamw"
    learning_rate: float = 2e-5
    lr_schedule: str = "linear"
    warmup_ratio: float = 0.02
    effective_batch_size: int = 32
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    epochs: int = 2
    precision: str = "bf16"
    flash_attention: bool = True
    freeze_audio_encoder: bool = False
    freeze_projector: bool = False

    @model_validator(mode="after")
    def _guard_lora(self) -> TrainingConfig:
        # ADR-003: two independent studies found vanilla FFT beating LoRA on this
        # problem class. LoRA is allowed, but only with an A/B on the table.
        if self.use_lora and not self.lora_ab_tested:
            raise ValueError(
                "LoRA requires an A/B against full fine-tuning first (ADR-003). "
                "Set lora_ab_tested: true once the comparison exists, and record it "
                "in docs/DECISIONS.md."
            )
        if self.use_lora and self.full_finetune:
            raise ValueError("use_lora and full_finetune are mutually exclusive")
        return self

    @model_validator(mode="after")
    def _check_batch_math(self) -> TrainingConfig:
        product = self.per_device_batch_size * self.gradient_accumulation_steps
        if product != self.effective_batch_size:
            raise ValueError(
                f"per_device_batch_size × gradient_accumulation_steps = {product}, "
                f"but effective_batch_size is {self.effective_batch_size}"
            )
        return self


class DataConfig(_Base):
    """Which corpora a run trains on."""

    manifests: list[Path] = Field(default_factory=list)
    eval_manifest: Path = Path("data/eval/manifest.jsonl")
    max_seconds_per_utterance: float = 35.0

    @field_validator("manifests")
    @classmethod
    def _never_train_on_eval(cls, manifests: list[Path]) -> list[Path]:
        # CLAUDE.md constraint 5. Cheap to check, catastrophic to get wrong: a
        # leaked eval set invalidates every number the project reports.
        for manifest in manifests:
            if EVAL_DIR in manifest.as_posix():
                raise ValueError(
                    f"{manifest} is inside {EVAL_DIR}/. The eval set is never trained on."
                )
        return manifests


class RunConfig(_Base):
    """A complete training run: `configs/phase1.yaml`, `configs/phase2.yaml`."""

    name: str
    phase: int
    description: str = ""
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    output_dir: Path = Path("checkpoints")
    seed: int = 1337
    tracker: str = "wandb"

    @property
    def config_hash(self) -> str:
        """Short stable hash of the whole config. Logged with every run (CLAUDE.md)."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


class InferenceConfig(_Base):
    """End-to-end captioning settings: `configs/inference.yaml`."""

    asr: ASRConfig = Field(default_factory=ASRConfig)
    aligner: AlignerConfig = Field(default_factory=AlignerConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    subtitle: SubtitleConfig = Field(default_factory=SubtitleConfig)
    lexicon: LexiconConfig = Field(default_factory=LexiconConfig)
    work_dir: Path = Path("data/work")


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    return loaded


def load_run_config(path: Path | str) -> RunConfig:
    """Load and validate a training config."""
    return RunConfig.model_validate(_read_yaml(Path(path)))


def load_inference_config(path: Path | str) -> InferenceConfig:
    """Load and validate an inference config."""
    return InferenceConfig.model_validate(_read_yaml(Path(path)))
