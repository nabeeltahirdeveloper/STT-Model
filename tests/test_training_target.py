"""The training target must tell the model where a transcription ends.

The first full fine-tune scored CER 91.3% against a 34.9% baseline. Its
transcriptions were not the problem: it produced correct text and then kept
generating to the 512-token ceiling -- "Good Morning, Pakistan" came back 27
times. The training target ended at the last word of the transcript and never
carried an EOS token, so the model was never shown where to stop.

The tell was in the category breakdown: CER tracked prediction length almost
exactly. NEWS ran 1.01x the reference length and scored 25.3%; DRAMA ran 2.61x
and scored 175.7%. Where the model happened to stop, it beat the baseline.
"""

from __future__ import annotations

import pytest

BASE = "Qwen/Qwen3-ASR-0.6B"


@pytest.fixture(scope="module")
def tokenizer():  # type: ignore[no-untyped-def]
    return pytest.importorskip("transformers").AutoTokenizer.from_pretrained(BASE)


def test_the_target_this_project_builds_ends_with_eos(tokenizer) -> None:  # type: ignore[no-untyped-def]
    """The regression: exactly the string finetune.py feeds the processor."""
    target = "<asr_text>Good morning Pakistan" + tokenizer.eos_token
    ids = tokenizer(target, add_special_tokens=False).input_ids
    assert ids[-1] == tokenizer.eos_token_id


def test_without_eos_the_target_does_not_terminate(tokenizer) -> None:  # type: ignore[no-untyped-def]
    """What the first checkpoint was trained on. Kept so the bug stays legible."""
    ids = tokenizer("<asr_text>Good morning Pakistan", add_special_tokens=False).input_ids
    assert ids[-1] != tokenizer.eos_token_id


def test_eos_survives_a_transcript_ending_in_punctuation(tokenizer) -> None:  # type: ignore[no-untyped-def]
    target = "<asr_text>Aik mahinay ke liye." + tokenizer.eos_token
    ids = tokenizer(target, add_special_tokens=False).input_ids
    assert ids[-1] == tokenizer.eos_token_id


def test_eos_is_one_token_not_literal_text(tokenizer) -> None:  # type: ignore[no-untyped-def]
    """If it tokenized as characters the model would learn to spell '<|im_end|>'."""
    assert len(tokenizer(tokenizer.eos_token, add_special_tokens=False).input_ids) == 1
