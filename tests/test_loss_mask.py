"""The loss boundary is the end of the chat prompt. Both earlier tries missed it.

`processor(text=...)` expands the single `<|audio_pad|>` in the template into one
token per audio frame, so the real prompt is longer than the prompt string and
its length varies with clip duration. Two attempts, two opposite failures:

    fixed 16 tokens     -> padding and `<|im_end|><|im_start|>assistant` scored;
                           the model learned to open a new turn and repeated
                           "Good Morning, Pakistan" 27 times. CER 91.3%.

    through <asr_text>  -> only the transcript scored; nothing taught the model
                           to emit the tag from the bare prompt inference sends,
                           and by step 200 it emitted EOS immediately and
                           transcribed nothing at all.

Anchoring on the last `<|im_start|>` plus the `assistant\\n` after it keeps the
padding masked and the tag scored.
"""

from __future__ import annotations

import pytest
import torch

from src.training.finetune import mask_prompt

TURN = 151644  # <|im_start|>
ASR = 151704  # <asr_text>
EOS = 151645
TAIL = 3  # <|im_start|>, "assistant", "\n"


def test_the_tag_and_transcript_are_scored() -> None:
    """What inference must generate, so what training must score."""
    ids = torch.tensor([[1, TURN, 2, 3, ASR, 7, 8, EOS]])
    labels = mask_prompt(ids, TURN, TAIL)
    assert ids[labels != -100].tolist() == [ASR, 7, 8, EOS]


def test_the_prompt_is_not_scored() -> None:
    ids = torch.tensor([[1, TURN, 2, 3, ASR, 7]])
    assert (mask_prompt(ids, TURN, TAIL)[0, :4] == -100).all()


@pytest.mark.parametrize("pads", [1, 7, 50, 300])
def test_the_boundary_follows_the_audio_expansion(pads: int) -> None:
    """The first bug: a fixed length is right for exactly one clip duration."""
    ids = torch.tensor([[TURN, 9, 9] + [1] * pads + [TURN, 2, 3, ASR, 7, EOS]])
    assert ids[mask_prompt(ids, TURN, TAIL) != -100].tolist() == [ASR, 7, EOS]


def test_a_fixed_length_would_score_the_scaffolding() -> None:
    """Pins the first bug, so the reasoning stays in the suite."""
    ids = torch.tensor([[1] * 26 + [TURN, 2, 3, ASR, 7]])
    leaked = ids.clone()
    leaked[:, :16] = -100  # what the first run did
    assert (leaked[0, 16:26] != -100).all(), "10 scaffolding tokens were scored"
    assert (mask_prompt(ids, TURN, TAIL)[0, 16:26] == -100).all()


def test_masking_through_the_tag_would_lose_it() -> None:
    """Pins the second bug: without the tag scored, the model never starts."""
    ids = torch.tensor([[TURN, 2, 3, ASR, 7, EOS]])
    scored = ids[mask_prompt(ids, TURN, TAIL) != -100].tolist()
    assert ASR in scored, "the model must learn to emit <asr_text> itself"


def test_the_last_turn_marker_wins() -> None:
    """system and user turns open earlier; the assistant turn is the boundary."""
    ids = torch.tensor([[TURN, 1, 1, TURN, 2, 2, TURN, 3, 3, ASR, 7]])
    assert ids[mask_prompt(ids, TURN, TAIL) != -100].tolist() == [ASR, 7]


def test_a_missing_marker_raises_rather_than_scoring_garbage() -> None:
    with pytest.raises(ValueError, match="not found"):
        mask_prompt(torch.tensor([[1, 2, 3]]), TURN, TAIL)


def test_the_input_ids_are_not_mutated() -> None:
    ids = torch.tensor([[TURN, 2, 3, ASR, 7]])
    mask_prompt(ids, TURN, TAIL)
    assert ids.tolist() == [[TURN, 2, 3, ASR, 7]]
