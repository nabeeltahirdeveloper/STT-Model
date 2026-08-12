"""LoRA fine-tune Qwen3-ASR locally on Apple Silicon (MPS).

    uv run python -m scripts.train --manifest data/labels/train-subset.jsonl
    uv run python -m scripts.train --manifest ... --limit 50 --epochs 1   # smoke test

**This is an experiment, not the production recipe.** CLAUDE.md constraint 6
requires an A/B against full fine-tuning before adopting LoRA, and that A/B
cannot be run here: full fine-tuning ~780M parameters needs about 12.5 GB of
optimizer state before activations, against 16 GB shared with macOS. So the
comparison this run would need is exactly the one the hardware forbids.

What it *can* answer, and what it is for: **do the labels teach Roman output?**
Stock Qwen3-ASR emits Devanagari (ADR-010). If a small adapter shifts that
toward Roman, the label pipeline works. That validates the expensive artifact
(the data) with a cheap and disposable one (the adapter).

What it cannot answer: whether LoRA is the right method, or what CER a properly
trained model would reach. Numbers from here must not be quoted against the
27.9% baseline as though they settled either question (ADR-016).

Memory, on 16 GB shared with the OS:

- base weights frozen in fp16, adapters in fp32 -- roughly 1.6 GB
- batch size 1, gradients only for adapter parameters
- clips capped at `--max-seconds`; activation memory scales with audio length
  and long clips are what turns a fitting run into an OOM at step 900
- gradient accumulation gives the effective batch without the memory
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

MANIFEST = Path("data/labels/train-subset.jsonl")
OUT = Path("out/lora-adapter")
BASE = "Qwen/Qwen3-ASR-0.6B"


@dataclass(slots=True)
class Sample:
    audio: Path
    text: str
    seconds: float


def load_manifest(path: Path, max_seconds: float, limit: int) -> list[Sample]:
    """Rows whose audio is on disk, short enough, and whose label is complete."""
    samples: list[Sample] = []
    missing = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        audio = Path(str(row["audio"]))
        if not audio.exists():
            missing += 1
            continue
        seconds = float(row.get("duration_s") or 0)
        if not (1.0 <= seconds <= max_seconds):
            continue
        label = str(row.get("label") or "").strip()
        if not label:
            continue
        samples.append(Sample(audio, label, seconds))
        if limit and len(samples) >= limit:
            break
    if missing:
        print(f"  {missing:,} rows skipped: audio not downloaded yet")
    return samples


def main(
    manifest: str = str(MANIFEST),
    base: str = BASE,
    out: str = str(OUT),
    epochs: int = 1,
    learning_rate: float = 1e-4,
    accumulate: int = 16,
    max_seconds: float = 20.0,
    rank: int = 16,
    limit: int = 0,
    save_every: int = 500,
) -> None:
    """Train a LoRA adapter on MPS and save it.

    Raises:
        SystemExit: if MPS is unavailable or the manifest has no usable rows.
    """
    import torch

    if not torch.backends.mps.is_available():
        raise SystemExit(
            "MPS is not available. This script targets Apple Silicon; on a CUDA "
            "box use full fine-tuning instead (constraint 6)."
        )

    samples = load_manifest(Path(manifest), max_seconds, limit)
    if not samples:
        raise SystemExit(
            f"no usable rows in {manifest}. Fetch the audio first:\n"
            f"  uv run python -m scripts.fetch_training_audio --manifest {manifest}"
        )
    hours = sum(s.seconds for s in samples) / 3600
    print(f"{len(samples):,} clips · {hours:.1f} h · clips 1-{max_seconds:.0f}s")

    import torchaudio
    from peft import LoraConfig, get_peft_model
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
        Qwen3ASRForConditionalGeneration,
    )
    from qwen_asr.core.transformers_backend.processing_qwen3_asr import Qwen3ASRProcessor

    device = torch.device("mps")
    print(f"loading {base} ...", flush=True)
    processor = Qwen3ASRProcessor.from_pretrained(base)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(base, dtype=torch.float32)

    # Attention projections only. The audio encoder is left frozen: the acoustic
    # problem is not what these labels teach, and adapting it would cost memory
    # that batch-size-1 training does not have spare.
    peft_config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=learning_rate
    )
    steps = math.ceil(len(samples) * epochs / accumulate)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=max(steps, 1)
    )

    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=True)
    started = time.time()
    seen = 0
    running = 0.0

    for epoch in range(epochs):
        for sample in samples:
            waveform, rate = torchaudio.load(sample.audio)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if rate != 16_000:
                waveform = torchaudio.functional.resample(waveform, rate, 16_000)

            batch = processor(
                audio=waveform.squeeze(0).numpy(),
                text=sample.text,
                sampling_rate=16_000,
                return_tensors="pt",
            )
            batch = {k: v.to(device) for k, v in batch.items() if hasattr(v, "to")}

            # Loss on the transcript only. Cloning input_ids wholesale trains
            # the model to predict its own prompt as well as the speech, and it
            # learns to: the first adapter emitted "language Hindi ..." -- the
            # decoding prefix -- as if it were transcribed audio, and burned
            # half its output budget doing so.
            #
            # The prompt precedes the target, so masking everything but the
            # final `len(target)` tokens leaves loss on the transcript alone.
            target = processor.tokenizer(sample.text, add_special_tokens=False).input_ids
            labels = batch["input_ids"].clone()
            span = min(len(target), labels.shape[1])
            if span < labels.shape[1]:
                labels[:, :-span] = -100
            batch["labels"] = labels

            loss = model.get_base_model().thinker(**batch).loss / accumulate
            loss.backward()
            running += loss.item() * accumulate
            seen += 1

            if seen % accumulate == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step = seen // accumulate
                elapsed = time.time() - started
                remaining = (len(samples) * epochs - seen) * elapsed / seen / 60
                print(
                    f"  epoch {epoch + 1}/{epochs} step {step}/{steps} "
                    f"loss {running / accumulate:.4f} · {remaining:.0f} min left",
                    flush=True,
                )
                running = 0.0
                if step % save_every == 0:
                    model.save_pretrained(destination)

    model.save_pretrained(destination)
    print(f"\nadapter saved -> {destination}")
    print(f"total {(time.time() - started) / 60:.1f} min")
    print("\nEvaluate it, then read ADR-016 before quoting the number anywhere.")


if __name__ == "__main__":  # pragma: no cover
    import typer

    typer.run(main)
