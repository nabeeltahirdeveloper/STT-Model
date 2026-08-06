"""Fine-tune Qwen3-ASR for Roman Urdu output.

STUB -- Phase 2.

The recipe is not open: it is Srota's (PROJECT.md §3.2), which took Hinglish
conversational WER 24.73% -> 15.85% for about $6.50 of compute on the same base
model and the same two failure modes. Full-parameter, no LoRA, no frozen layers.
`TrainingConfig` refuses a config that departs from that without an A/B.

The compute is trivial; all the cost and risk is in data preparation. Do not
start here.
"""

from __future__ import annotations

from pathlib import Path

from src.config import RunConfig, load_run_config


def finetune(config: RunConfig) -> Path:
    """Run a fine-tune and return the checkpoint directory.

    Every run logs `config.config_hash` to the experiment tracker, so a result
    can always be traced back to the exact configuration that produced it.

    Raises:
        NotImplementedError: Phase 2 work.
    """
    raise NotImplementedError("Phase 2: blocked on Phase 1 labels")


def main(config_path: str = "configs/phase1.yaml") -> None:
    """CLI entrypoint: `python -m src.training.finetune --config configs/phase1.yaml`."""
    finetune(load_run_config(config_path))


if __name__ == "__main__":  # pragma: no cover
    import typer

    typer.run(main)
