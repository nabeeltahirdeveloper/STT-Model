"""Run stock Qwen3-ASR over the eval set and write scoreable predictions.

    uv run python -m scripts.run_baseline

This is Phase 0 item 3 -- the "before" number that Phase 2 compares against.
Without it a trained model produces a score with nothing to measure it by.

**Two outputs, because stock output is not Roman.** The probe showed the model
emits Devanagari for Urdu words (`ye bohot acha hai` -> `ये बहुत अच्छा है`),
so scoring it directly against a Roman reference measures script mismatch and
not accuracy. Both files are written and both get reported:

    out/baseline-raw.txt        exactly what the model emitted
    out/baseline-romanized.txt  the same text through our romanize + normalize

The raw number will look catastrophic. That *is* the finding -- it is the
evidence that a stock model cannot do this job, which is the whole argument for
fine-tuning. The romanized number measures the fallback architecture: stock
model plus our own converter. Reporting only one of the two would be either
alarmist or flattering.

Line order follows `data/eval/manifest.jsonl` exactly, because `src.eval.score`
matches predictions to references by line number and a silent off-by-one there
would misattribute every utterance after it.

Constraint 7: decoding is language-agnostic. Urdu is not in the model's 30
supported languages -- confirmed, not assumed -- and forcing a wrong one is
explicitly forbidden.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

MANIFEST = Path("data/eval/manifest.jsonl")
OUT = Path("out")


def _write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(
    model_id: str = "Qwen/Qwen3-ASR-1.7B",
    out_dir: str = str(OUT),
    batch_size: int = 8,
    limit: int = 0,
    romanize_output: bool = True,
    device: str = "auto",
) -> None:
    """Transcribe every eval clip, then romanize the result.

    Args:
        limit: stop after this many clips, for a quick smoke run. 0 means all.
        romanize_output: run the Devanagari->Roman converter over the output.
            True for a stock model, which emits Devanagari (ADR-010). **False
            for a fine-tuned one**, whose whole purpose is to emit Roman
            already -- passing its output through the converter a second time
            would rewrite correct Roman rather than measure it.
    """
    warnings.filterwarnings("ignore")
    from qwen_asr import Qwen3ASRModel

    rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines()]
    if limit:
        rows = rows[:limit]
    audio_minutes = sum(float(row["duration_s"]) for row in rows) / 60
    print(f"{len(rows)} clips · {audio_minutes:.1f} min", flush=True)

    # from_pretrained leaves the model wherever transformers puts it, which is
    # the CPU. A T4 run showed 0.0 GB of 15 GB in use and took hours; the card
    # was never asked to do anything. device_map is forwarded to
    # AutoModel.from_pretrained, so placement has to be requested explicitly.
    import torch

    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    # bfloat16 is what the checkpoint ships, and it is fine on CUDA. MPS support
    # for it is patchy, so Apple Silicon gets float16 instead -- same size, and
    # inference does not need bf16's exponent range.
    dtype = torch.float16 if device == "mps" else torch.bfloat16
    print(f"loading {model_id} on {device} ({dtype}) ...", flush=True)
    model = Qwen3ASRModel.from_pretrained(model_id, device_map=device, dtype=dtype)

    destination = Path(out_dir)
    raw: list[str] = []
    started = time.time()
    for index in range(0, len(rows), batch_size):
        batch = rows[index : index + batch_size]
        results = model.transcribe([str(row["audio"]) for row in batch], language=None)
        for result in results:
            text = str(getattr(result, "text", result)).strip()
            # A newline inside a prediction would shift every later line against
            # the wrong reference, so collapse it here rather than debug it later.
            raw.append(" ".join(text.split()))
        done = index + len(batch)
        rate = (time.time() - started) / max(done, 1)
        print(
            f"  {done}/{len(rows)}  ({rate * (len(rows) - done) / 60:.0f} min left)",
            flush=True,
        )
        # Written every batch: an hour-long run that dies at clip 260 should not
        # cost the first 259.
        _write(destination / "baseline-raw.txt", raw)

    print(f"\ntranscribed in {(time.time() - started) / 60:.1f} min", flush=True)

    if not romanize_output:
        print(f"\nraw output kept as the prediction -> {destination / 'baseline-raw.txt'}")
        print("(--no-romanize-output: this model is expected to emit Roman itself)")
        return

    print("romanizing (Devanagari -> Roman, English left alone) ...", flush=True)
    from scripts.romanize_via_opencut import romanize
    from src.labeling.lexicon import Lexicon
    from src.labeling.normalize import Normalizer

    normalizer = Normalizer(Lexicon.load())
    romanized = [normalizer.normalize_text(text) for text in romanize(raw)]
    _write(destination / "baseline-romanized.txt", romanized)

    print(f"\n  {destination}/baseline-raw.txt")
    print(f"  {destination}/baseline-romanized.txt")
    print("\nscore both:")
    print(f"  uv run python -m src.eval.score --pred {destination}/baseline-raw.txt")
    print(f"  uv run python -m src.eval.score --pred {destination}/baseline-romanized.txt")


if __name__ == "__main__":  # pragma: no cover
    import typer

    typer.run(main)
