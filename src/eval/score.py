"""Scoring CLI.

    python -m src.eval.score --pred out/predictions.txt --ref data/eval/reference.txt

STUB -- follow-up prompt 5. The CLI shape is fixed now because `run.sh eval`
already calls it and the two must not drift.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_METRICS = "cer,sn-wer,normalized-wer,english-preservation"


def main(
    pred: str = "out/predictions.txt",
    ref: str = "data/eval/reference.txt",
    metrics: str = DEFAULT_METRICS,
) -> None:
    """Score predictions against the held-out reference and print a table.

    Raises:
        NotImplementedError: prompt 5.
    """
    _ = Path(pred), Path(ref), metrics.split(",")
    raise NotImplementedError(
        "prompt 5: implement src/eval/metrics.py first, then print CER + SN-WER "
        "and the §6.3 error breakdown"
    )


if __name__ == "__main__":  # pragma: no cover
    import typer

    typer.run(main)
