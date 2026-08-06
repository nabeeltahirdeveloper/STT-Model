"""End-to-end orchestration: video in, subtitle files out.

STUB -- Phase 3, but the stage order is fixed and load-bearing:

    ingest -> preprocess -> ASR -> NORMALIZE -> align -> subtitle

Normalization sits between ASR and alignment, not after it. Two reasons: raw
model output must never reach a user (CLAUDE.md constraint 3), and normalizing
after alignment would change token lengths and invalidate the timings.
"""

from __future__ import annotations

from pathlib import Path

from src.config import InferenceConfig, load_inference_config


def run(video: Path, out_dir: Path, config: InferenceConfig | None = None) -> dict[str, Path]:
    """Caption `video`, returning the paths of the written outputs.

    Returns:
        A mapping of format name (`srt`, `vtt`, `txt`) to the file written.

    Raises:
        NotImplementedError: Phase 3 work.
    """
    raise NotImplementedError("Phase 3: blocked on the ASR fine-tune and the §4.4 aligner decision")


def main(input: str, out: str = "out/", config: str = "configs/inference.yaml") -> None:
    """CLI entrypoint: `python -m src.api.pipeline --input video.mp4 --out out/`."""
    run(Path(input), Path(out), load_inference_config(config))


if __name__ == "__main__":  # pragma: no cover
    import typer

    typer.run(main)
