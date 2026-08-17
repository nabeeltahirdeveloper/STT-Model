"""Signals that would have caught the first run going wrong.

That run drove loss 13.56 -> 0.35 and was reported as healthy. It was in fact
learning to reproduce audio padding and chat scaffolding, and nothing in the
loss curve said so -- the damage only appeared once the model generated, after
the session was spent. Loss is necessary and not sufficient.

p(eos) is the cheap half: the probability the model puts on EOS where EOS is
the right answer, read off the forward pass the loss already computes. A model
learning to stop drives it toward 1.0. The broken run would have sat near zero
throughout.
"""

from __future__ import annotations

import torch

from src.training.finetune import eos_confidence

# A small id so the toy logits stay small. The real token is 151645; the
# function indexes by id, so the value itself is not what is under test.
EOS = 150
VOCAB = 200


def test_a_model_that_predicts_eos_scores_high() -> None:
    labels = torch.tensor([[-100, 7, EOS]])
    logits = torch.zeros(1, 3, VOCAB)
    logits[0, 1, EOS] = 20.0  # position 1 predicts the token at position 2
    assert eos_confidence(logits, labels, EOS) > 0.99


def test_a_model_that_never_predicts_eos_scores_near_zero() -> None:
    """The first run's signature: EOS was not in its targets at all."""
    labels = torch.tensor([[-100, 7, EOS]])
    logits = torch.zeros(1, 3, VOCAB)
    logits[0, 1, 42] = 20.0  # confidently predicts something else
    assert eos_confidence(logits, labels, EOS) < 0.01


def test_it_reads_the_position_before_the_eos_target() -> None:
    """Next-token prediction: position i-1 is what predicts token i."""
    labels = torch.tensor([[-100, -100, 7, EOS]])
    logits = torch.zeros(1, 4, VOCAB)
    logits[0, 2, EOS] = 20.0
    assert eos_confidence(logits, labels, EOS) > 0.99
    wrong = torch.zeros(1, 4, VOCAB)
    wrong[0, 3, EOS] = 20.0  # too late to matter
    assert eos_confidence(wrong, labels, EOS) < 0.01


def test_no_eos_in_the_batch_returns_zero_rather_than_raising() -> None:
    """A diagnostic must never be the thing that ends a four-hour run."""
    labels = torch.tensor([[-100, 7, 8]])
    assert eos_confidence(torch.zeros(1, 3, VOCAB), labels, EOS) == 0.0


def test_eos_at_position_zero_is_not_read_out_of_bounds() -> None:
    assert eos_confidence(torch.zeros(1, 2, VOCAB), torch.tensor([[EOS, 7]]), EOS) == 0.0


def test_it_does_not_require_gradients() -> None:
    """Called every step; it must not retain graph memory."""
    labels = torch.tensor([[-100, 7, EOS]])
    logits = torch.zeros(1, 3, VOCAB, requires_grad=True)
    value = eos_confidence(logits, labels, EOS)
    assert isinstance(value, float)
