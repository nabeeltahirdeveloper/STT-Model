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
    max_minutes: float = 0.0,
    resume: str = "",
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
    # float16 halves the weights (3.1 GB -> 1.6 GB). At float32 the working set
    # reached 24 GB on a 16 GB machine, so macOS swapped and the run went from
    # 35 to 59 minutes -- bound by SSD rather than GPU. LoRA parameters stay
    # float32 for stable updates; peft handles that.
    model = Qwen3ASRForConditionalGeneration.from_pretrained(base, dtype=torch.float16)

    # Training inputs must match what inference produces, or the model is
    # optimized for a format it will never be asked for. Read from the package
    # source rather than assumed:
    #
    #   input  = processor.apply_chat_template(msgs, add_generation_prompt=True)
    #   output = "language {Lang}<asr_text>{transcript}"   (parse_asr_output)
    #
    # Feeding the bare transcript, as this script first did, taught the model to
    # start mid-sentence from a prompt it had never seen and to omit the tag the
    # parser needs. Its output then could not be parsed at all.
    messages = [
        {"role": "user", "content": [{"type": "audio", "audio": ""}, {"type": "text", "text": ""}]}
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    prompt_len = len(processor.tokenizer(prompt, add_special_tokens=False).input_ids)
    print(f"prompt: {prompt_len} tokens · {prompt[:60]!r}", flush=True)

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

    # Colab free tier disconnects at roughly four hours, and an unfinished run
    # that saved nothing is worth exactly zero. Stopping on a clock rather than
    # on an epoch boundary means the adapter always exists when time runs out.
    if resume and Path(resume).exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model.get_base_model(), resume, is_trainable=True)
        model.to(device)
        model.train()
        print(f"resumed from {resume}", flush=True)

    deadline = (
        started_at + max_minutes * 60 if (started_at := time.time()) and max_minutes else None
    )
    started = started_at
    seen = 0
    running = 0.0

    for epoch in range(epochs):
        for sample in samples:
            waveform, rate = torchaudio.load(sample.audio)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if rate != 16_000:
                waveform = torchaudio.functional.resample(waveform, rate, 16_000)

            # The tag is what parse_asr_output looks for. Urdu is not a
            # supported language (constraint 7), and "language None<asr_text>"
            # is read as empty audio, so the label carries no language claim --
            # only the separator the parser needs.
            target = f"<asr_text>{sample.text}"
            batch = processor(
                audio=waveform.squeeze(0).numpy(),
                text=prompt + target,
                sampling_rate=16_000,
                return_tensors="pt",
            )
            # Audio features arrive float32; the model is float16. Inference
            # does the same cast (`inputs.to(model.dtype)`), so training must
            # too or the first conv layer rejects the input. Integer tensors --
            # input_ids, masks -- must keep their dtype.
            batch = {
                key: (
                    value.to(device, dtype=model.dtype)
                    if value.is_floating_point()
                    else value.to(device)
                )
                for key, value in batch.items()
                if hasattr(value, "to")
            }

            # Loss on the target only. The prompt is now genuinely present in
            # input_ids, so masking it is genuinely required -- unlike the
            # earlier version of this code, where input_ids held nothing but
            # the transcript and the mask was a no-op that silently did nothing.
            labels = batch["input_ids"].clone()
            labels[:, :prompt_len] = -100
            batch["labels"] = labels

            loss = model.get_base_model().thinker(**batch).loss / accumulate
            loss.backward()
            running += loss.item() * accumulate
            seen += 1

            # MPS keeps freed blocks in its cache, so a long run accumulates
            # allocations until the machine swaps. Releasing periodically costs
            # a little throughput and prevents that.
            if seen % 50 == 0:
                torch.mps.empty_cache()

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

                if deadline and time.time() >= deadline:
                    model.save_pretrained(destination)
                    print(
                        f"\nstopped at {max_minutes:.0f} min (step {step}/{steps}); "
                        f"adapter saved. Re-run with --resume {destination} to continue.",
                        flush=True,
                    )
                    return

    model.save_pretrained(destination)
    print(f"\nadapter saved -> {destination}")
    print(f"total {(time.time() - started) / 60:.1f} min")
    print("\nEvaluate it, then read ADR-016 before quoting the number anywhere.")


if __name__ == "__main__":  # pragma: no cover
    import typer

    typer.run(main)
