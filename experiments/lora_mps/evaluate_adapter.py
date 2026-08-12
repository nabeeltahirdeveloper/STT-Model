"""Did the adapter move the output script from Devanagari toward Roman?

    uv run python -m experiments.lora_mps.evaluate_adapter --clips 8

See `experiments/lora_mps/README.md` for the question and its limits, and
ADR-016 for why no number here may be compared with the 27.9% baseline.

The headline is the **script mix**, not CER. CER against a Roman reference
cannot tell "wrong word" apart from "right word in the wrong alphabet" -- that
conflation is what made the raw stock baseline read 75.8% while the same output
scored 27.9% once romanized. For this experiment the script *is* the result.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

ADAPTER = Path("out/lora-adapter")
MANIFEST = Path("data/eval/manifest.jsonl")
BASE = "Qwen/Qwen3-ASR-0.6B"

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_URDU = re.compile(r"[؀-ۿ]")
_LATIN = re.compile(r"[A-Za-z]")


def script_mix(text: str) -> dict[str, float]:
    """Share of letters in each script."""
    counts = {
        "latin": len(_LATIN.findall(text)),
        "devanagari": len(_DEVANAGARI.findall(text)),
        "urdu": len(_URDU.findall(text)),
    }
    total = sum(counts.values()) or 1
    return {key: value / total for key, value in counts.items()}


def main(
    adapter: str = str(ADAPTER),
    base: str = BASE,
    manifest: str = str(MANIFEST),
    clips: int = 8,
) -> None:
    """Transcribe the same clips with and without the adapter, and compare."""
    warnings.filterwarnings("ignore")
    from peft import PeftModel
    from qwen_asr import Qwen3ASRModel

    if not Path(adapter).exists():
        raise SystemExit(f"no adapter at {adapter}. Train one: uv run python -m scripts.train")

    rows = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if Path(str(row["audio"])).exists()][:clips]
    if not rows:
        raise SystemExit(f"no eval audio found via {manifest}")

    # The wrapper is used rather than the bare model because generation needs
    # the language-agnostic decoding prefix it builds; the raw processor refuses
    # without a `text` prompt, and hand-rolling that prefix is how constraint 7
    # gets violated by accident.
    print(f"loading {base} ...", flush=True)
    wrapper = Qwen3ASRModel.from_pretrained(base)
    audio = [str(row["audio"]) for row in rows]

    print(f"transcribing {len(rows)} clips with the stock model ...", flush=True)
    stock = [str(getattr(r, "text", r)).strip() for r in wrapper.transcribe(audio, language=None)]

    print("attaching the adapter ...", flush=True)
    device = next(wrapper.model.parameters()).device
    wrapper.model = PeftModel.from_pretrained(wrapper.model, adapter).to(device).eval()
    print("transcribing with the adapter ...", flush=True)
    tuned = [str(getattr(r, "text", r)).strip() for r in wrapper.transcribe(audio, language=None)]

    print(f"\nscript mix over {len(rows)} clips")
    for name, texts in (
        ("stock", stock),
        ("adapter", tuned),
        ("reference", [str(row["text"]) for row in rows]),
    ):
        mix = script_mix(" ".join(texts))
        print(
            f"  {name:<10}latin {mix['latin']:>6.1%}   "
            f"devanagari {mix['devanagari']:>6.1%}   urdu {mix['urdu']:>6.1%}"
        )

    for row, before, after in list(zip(rows, stock, tuned, strict=True))[:3]:
        print(f"\n  ref     : {str(row['text'])[:84]}")
        print(f"  stock   : {before[:84]}")
        print(f"  adapter : {after[:84]}")

    print("\nADR-016: a pipeline check, not a quality measurement.")


if __name__ == "__main__":  # pragma: no cover
    import typer

    typer.run(main)
