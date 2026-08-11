"""Answer three questions about stock Qwen3-ASR before committing to a baseline.

    uv run python -m scripts.probe_asr            # 3 clips, quick
    uv run python -m scripts.probe_asr --clips 20 # a firmer speed estimate

Deliberately small. The point is to find out whether the full baseline is a
30-minute job or an overnight one *before* starting it, and to check two
assumptions that would quietly invalidate the result if they were wrong:

1. **Speed.** Realtime factor on this machine. 35.4 minutes of eval audio at 1x
   is a coffee break; at 10x it needs a rented GPU box.
2. **Languages.** CLAUDE.md constraint 7 says Urdu is not a supported language
   and the built-in language ID must not be trusted. This prints the supported
   list so that stays a checked fact rather than a remembered one.
3. **Shape of the output.** Whether stock output is Roman, Urdu script, or
   English translation decides how much of the baseline number is even
   meaningful.

Decoding uses `language=None` -- the language-agnostic prefix. Forcing
`language="Urdu"` is what constraint 7 forbids.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

MANIFEST = Path("data/eval/manifest.jsonl")


def main(model_id: str = "Qwen/Qwen3-ASR-1.7B", clips: int = 3) -> None:
    """Load the model, report supported languages, and time a few clips."""
    warnings.filterwarnings("ignore")
    from qwen_asr import Qwen3ASRModel

    print(f"loading {model_id} ...", flush=True)
    started = time.time()
    model = Qwen3ASRModel.from_pretrained(model_id)
    print(f"loaded in {time.time() - started:.0f}s\n", flush=True)

    try:
        languages = model.get_supported_languages()
        print(f"supported languages ({len(languages)}):")
        print(f"  {', '.join(languages)}")
        for probe in ("Urdu", "Hindi", "English"):
            print(f"  {probe:<8} {'SUPPORTED' if probe in languages else 'not supported'}")
    except Exception as error:  # noqa: BLE001 - informational probe only
        print(f"could not read supported languages: {error}")

    rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines()]
    sample = rows[:clips]
    audio_seconds = sum(float(row["duration_s"]) for row in sample)

    print(f"\ntranscribing {len(sample)} clips ({audio_seconds:.0f}s of audio) ...", flush=True)
    started = time.time()
    results = model.transcribe([str(row["audio"]) for row in sample], language=None)
    elapsed = time.time() - started

    factor = elapsed / audio_seconds if audio_seconds else 0.0
    total_minutes = sum(float(row["duration_s"]) for row in rows) / 60
    print(f"\n{elapsed:.1f}s for {audio_seconds:.0f}s audio = {factor:.2f}x realtime")
    print(
        f"full eval set ({total_minutes:.1f} min) projects to ~{total_minutes * factor:.0f} min\n"
    )

    for row, result in zip(sample, results, strict=True):
        print(f"  ref : {row['text'][:95]}")
        print(f"  hyp : {str(getattr(result, 'text', result))[:95]}\n")


if __name__ == "__main__":  # pragma: no cover
    import typer

    typer.run(main)
