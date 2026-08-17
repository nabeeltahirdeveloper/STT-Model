"""The loss mask must cover the prompt exactly, however long it turns out to be.

`processor(text=...)` expands the single `<|audio_pad|>` in the chat template
into one token per audio frame, so the prompt inside `input_ids` is much longer
than the prompt string, and its length varies with clip duration.

The first full fine-tune masked a constant 16 -- the un-expanded template
length. On a real clip that left 22 tokens scored where only 4 were transcript:
the model was trained to predict audio padding and the chat scaffolding,
including the `<|im_end|><|im_start|>assistant` that opens a new turn. It
learned to do exactly that. One clip came back as "Good Morning, Pakistan" 27
times, and CER read 91.3%.
"""

from __future__ import annotations

import pytest
import torch

from src.training.finetune import mask_prompt

MARKER = 151704  # <asr_text>


def test_only_what_follows_the_marker_is_scored() -> None:
    ids = torch.tensor([[1, 2, MARKER, 7, 8, 9]])
    labels = mask_prompt(ids, MARKER)
    assert labels.tolist() == [[-100, -100, -100, 7, 8, 9]]


def test_the_marker_itself_is_masked() -> None:
    """It is part of the prompt: inference emits it, the model need not predict it."""
    labels = mask_prompt(torch.tensor([[MARKER, 5]]), MARKER)
    assert labels[0, 0].item() == -100


@pytest.mark.parametrize("pads", [1, 7, 50, 300])
def test_the_mask_follows_the_audio_expansion(pads: int) -> None:
    """The regression: a fixed length is right for exactly one clip duration.

    A longer clip expands to more audio tokens, so a constant cut leaves
    scaffolding scored. The transcript must be the scored part at every length.
    """
    ids = torch.tensor([[1] * pads + [MARKER, 7, 8]])
    labels = mask_prompt(ids, MARKER)
    scored = ids[labels != -100]
    assert scored.tolist() == [7, 8]


def test_a_fixed_length_mask_would_have_been_wrong() -> None:
    """Pins why the constant was a bug, so the reasoning survives in the suite."""
    ids = torch.tensor([[1] * 26 + [MARKER, 7]])
    leaked = ids.clone()
    leaked[:, :16] = -100  # what the first run did
    assert (leaked[0, 16:26] != -100).all(), "10 scaffolding tokens were scored"
    assert (mask_prompt(ids, MARKER)[0, 16:26] == -100).all()


def test_the_last_marker_wins() -> None:
    """A transcript could contain the literal text; the boundary is the final one."""
    ids = torch.tensor([[1, MARKER, 5, MARKER, 9]])
    assert mask_prompt(ids, MARKER).tolist() == [[-100, -100, -100, -100, 9]]


def test_a_missing_marker_raises_rather_than_scoring_garbage() -> None:
    with pytest.raises(ValueError, match="not found"):
        mask_prompt(torch.tensor([[1, 2, 3]]), MARKER)


def test_the_input_ids_are_not_mutated() -> None:
    ids = torch.tensor([[1, MARKER, 7]])
    mask_prompt(ids, MARKER)
    assert ids.tolist() == [[1, MARKER, 7]]
