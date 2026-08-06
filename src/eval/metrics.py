"""Evaluation metrics.

STUB -- follow-up prompt 5. This is the work that answers the question the whole
project hinges on (PROJECT.md §6.3), so it is next after Phase 0's gates.

Why not just use WER: Roman Urdu has no standard orthography, so WER scores
`nahin` against `nahi` as a full error even though both are correct. It
systematically understates quality here. CER is primary, SN-WER secondary, and
raw WER may appear in a table only next to CER -- never alone (ADR-005).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.labeling.normalize import Normalizer


@dataclass(frozen=True, slots=True)
class ErrorBreakdown:
    """The §6.3 diagnostic: which kind of error, and how much of it.

    The acoustic/orthographic ratio decides where the next month of effort goes.
    Acoustic errors need more data; orthographic errors need normalizer work;
    code-switch errors need better labels; timing errors need aligner work.
    Guessing which one dominates is how teams spend a month on the wrong thing.
    """

    acoustic: float
    orthographic: float
    code_switch: float
    timing: float

    def __post_init__(self) -> None:
        total = self.acoustic + self.orthographic + self.code_switch + self.timing
        if total and abs(total - 1.0) > 1e-6:
            raise ValueError(f"error breakdown must sum to 1.0, got {total}")


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate. The primary metric.

    Raises:
        NotImplementedError: prompt 5.
    """
    raise NotImplementedError("prompt 5: Levenshtein over characters")


def normalized_wer(reference: str, hypothesis: str, normalizer: Normalizer) -> float:
    """WER computed after both sides pass through the canonical normalizer.

    Removes the spelling-variance penalty that makes raw WER misleading.

    Raises:
        NotImplementedError: prompt 5.
    """
    raise NotImplementedError("prompt 5")


def sn_wer(reference: str, hypothesis: str) -> float:
    """Script-normalized WER. Reduces inflated Urdu error rates by 6.4-9.0%.

    Raises:
        NotImplementedError: prompt 5.
    """
    raise NotImplementedError("prompt 5")


def english_preservation(reference: str, hypothesis: str, normalizer: Normalizer) -> float:
    """Fraction of English words emitted in English spelling. Target > 90%.

    This is the metric that catches the failure the architecture exists to avoid:
    if it drops, the model is respelling `meeting` as `mitting` somewhere
    (SPELLING-SPEC §5.1).

    Raises:
        NotImplementedError: prompt 5.
    """
    raise NotImplementedError("prompt 5")


def spelling_consistency(hypotheses: list[str], normalizer: Normalizer) -> float:
    """Fraction of word types spelled identically everywhere they appear. Target > 98%.

    Raises:
        NotImplementedError: prompt 5.
    """
    raise NotImplementedError("prompt 5")
