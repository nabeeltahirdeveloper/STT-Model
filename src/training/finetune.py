"""Full fine-tune Qwen3-ASR. The production recipe (ADR-003, constraint 6).

    uv run python -m src.training.finetune --limit 60          # verify first
    uv run python -m src.training.finetune --max-minutes 150   # real run

**Full-parameter, not LoRA.** Two independent studies found vanilla FFT beating
LoRA substantially on this problem class, and a LoRA run on 20 h of these labels
scored CER 64.1% against 34.9% for the stock model plus our romanizer -- worse
than not training at all (`experiments/lora_mps/README.md`). LoRA is not adopted
here for lack of memory; the memory is made to fit instead.

Fitting ~780M trainable parameters in a 16 GB T4:

    bf16 weights                  1.6 GB
    bf16 gradients                1.6 GB
    fp32 master copy              3.1 GB
    AdamW **8-bit** states        1.6 GB   (6.2 GB at fp32 -- this is the trick)
    activations, batch 1,
      gradient checkpointing      2-4 GB
    ------------------------------------
                                 ~10-12 GB

`bitsandbytes` supplies the 8-bit optimizer and gradient checkpointing trades
compute for activation memory. Both are CUDA-only, which is why this cannot run
on Apple Silicon and `scripts/train.py` exists separately.

**Input format is copied from `scripts/train.py`, where it was verified against
the qwen_asr source rather than inferred.** Inference builds
`apply_chat_template(msgs, add_generation_prompt=True)` and expects the model to
emit `...<asr_text>{transcript}`. Feeding the bare transcript -- the first
version of this pipeline -- taught the model to start mid-sentence from a prompt
it had never seen, and its output could not be parsed at all.

**Stops on a clock, not an epoch.** Colab's free tier disconnects at roughly
four hours. A run that dies having saved nothing is worth zero; a partial
checkpoint is worth resuming.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # torch is imported inside the functions that need it
    import torch

MANIFEST = Path("data/labels/train-subset.jsonl")
OUT = Path("out/finetuned")
BASE = "Qwen/Qwen3-ASR-0.6B"


@dataclass(slots=True)
class Sample:
    audio: Path
    text: str
    seconds: float


def load_audio(path: Path) -> tuple[torch.Tensor, int]:
    """Read one clip as mono float32, shaped (1, samples).

    soundfile rather than `torchaudio.load`, which since torchaudio 2.11
    delegates to TorchCodec -- an extra dependency that pulls in an ffmpeg
    toolchain and was absent on Colab, so the first training step of the session
    died on the import. soundfile is already present via librosa, reads the
    16-bit PCM WAV this corpus ships, and needs no codec chain.

    `torchaudio.functional.resample` is unaffected: it is pure tensor maths and
    is still what does the resampling at the call site.
    """
    # Imported here, not at module scope, to match the rest of this file: the
    # heavy imports stay inside the functions that need them so `--help` and the
    # manifest checks do not pay for loading torch.
    import soundfile
    import torch

    data, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data).T  # soundfile gives (samples, channels)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, int(rate)


def sanitize_generation_config(config: object) -> list[str]:
    """Reset sampling settings that transformers refuses to save. Returns changes.

    Qwen3-ASR ships a generation_config.json carrying `temperature=1e-06`
    alongside `do_sample=False`. Since 4.5x, `save_pretrained` validates that
    config strictly and raises, so a run trains to completion and then cannot
    write its own checkpoint -- the failure lands at the end, where it costs the
    whole session rather than a minute.

    Greedy decoding never reads these values, so resetting them to the defaults
    transformers considers unset changes no behaviour. Fixing this at save time
    rather than at load keeps the loaded model identical to the published one.
    """
    unset = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 50,
        "typical_p": 1.0,
        "epsilon_cutoff": 0.0,
        "eta_cutoff": 0.0,
    }
    if getattr(config, "do_sample", False):
        return []
    changed = []
    for name, default in unset.items():
        current = getattr(config, name, default)
        if current != default:
            changed.append(f"{name}={current!r}")
            setattr(config, name, default)
    return changed


def load_manifest(path: Path, max_seconds: float, limit: int) -> list[Sample]:
    """Rows whose audio is present, short enough, and whose label is complete."""
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
        print(f"  {missing:,} rows skipped: audio not downloaded")
    return samples


def main(
    manifest: str = str(MANIFEST),
    base: str = BASE,
    out: str = str(OUT),
    epochs: int = 1,
    learning_rate: float = 2e-5,
    accumulate: int = 16,
    max_seconds: float = 20.0,
    limit: int = 0,
    max_minutes: float = 0.0,
    resume: str = "",
    hf_repo: str = "",
    push_every: int = 400,
    warmup: int = 50,
) -> None:
    """Full fine-tune and save. Push to a private HF repo if one is given.

    Raises:
        SystemExit: without CUDA, without bitsandbytes, or with no usable rows.
    """
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "Full fine-tuning needs CUDA. On Apple Silicon the optimizer state "
            "alone exceeds available memory -- use scripts/train.py there, and "
            "read ADR-016 on why its numbers cannot settle the method question."
        )
    try:
        from bitsandbytes.optim import AdamW8bit
    except ImportError as error:  # pragma: no cover - environment-specific
        raise SystemExit(
            "bitsandbytes is required: it is what makes the optimizer state fit "
            "(6.2 GB -> 1.6 GB). Install with: uv pip install bitsandbytes"
        ) from error

    samples = load_manifest(Path(manifest), max_seconds, limit)
    if not samples:
        raise SystemExit(
            f"no usable rows in {manifest}. Fetch the audio first:\n"
            f"  uv run python -m scripts.fetch_training_audio --manifest {manifest}"
        )

    import torchaudio
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
        Qwen3ASRForConditionalGeneration,
    )
    from qwen_asr.core.transformers_backend.processing_qwen3_asr import Qwen3ASRProcessor

    device = torch.device("cuda")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    hours = sum(s.seconds for s in samples) / 3600
    print(f"{len(samples):,} clips · {hours:.1f} h · {torch.cuda.get_device_name(0)} {vram:.0f} GB")

    source = resume if resume and Path(resume).exists() else base
    print(f"loading {source} ...", flush=True)
    processor = Qwen3ASRProcessor.from_pretrained(base)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(source, dtype=torch.bfloat16)

    # Verified in scripts/train.py against the package source: input_ids from
    # processor(text=...) contains ONLY what is passed, so the chat template has
    # to be prepended explicitly and then masked out of the loss.
    messages = [
        {"role": "user", "content": [{"type": "audio", "audio": ""}, {"type": "text", "text": ""}]}
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    prompt_len = len(processor.tokenizer(prompt, add_special_tokens=False).input_ids)
    print(f"prompt: {prompt_len} tokens", flush=True)

    model.gradient_checkpointing_enable()
    model.config.use_cache = False  # incompatible with checkpointing
    model.to(device)
    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable:,}  (full fine-tune, not LoRA)")

    optimizer = AdamW8bit(model.parameters(), lr=learning_rate)
    print("optimizer created", flush=True)
    steps = math.ceil(len(samples) * epochs / accumulate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=max(steps, 2),
        pct_start=min(0.3, warmup / max(steps, 1)),
    )

    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=True)

    def save(note: str) -> None:
        if changes := sanitize_generation_config(model.generation_config):
            print(f"  generation_config reset to defaults: {', '.join(changes)}", flush=True)
        model.save_pretrained(destination)
        processor.save_pretrained(destination)
        if hf_repo:
            from huggingface_hub import HfApi, create_repo

            create_repo(hf_repo, private=True, exist_ok=True)
            HfApi().upload_folder(
                folder_path=str(destination), repo_id=hf_repo, commit_message=note
            )
            print(f"  pushed -> {hf_repo} ({note})", flush=True)

    started = time.time()
    deadline = started + max_minutes * 60 if max_minutes else None
    seen, running = 0, 0.0

    for _epoch in range(epochs):
        print(f"Starting epoch {_epoch}", flush=True)
        for sample in samples:
            waveform, rate = load_audio(sample.audio)
            if rate != 16_000:
                waveform = torchaudio.functional.resample(waveform, rate, 16_000)

            batch = processor(
                audio=waveform.squeeze(0).numpy(),
                text=prompt + f"<asr_text>{sample.text}",
                sampling_rate=16_000,
                return_tensors="pt",
            )
            batch = {
                key: (
                    value.to(device, dtype=model.dtype)
                    if value.is_floating_point()
                    else value.to(device)
                )
                for key, value in batch.items()
                if hasattr(value, "to")
            }
            labels = batch["input_ids"].clone()
            labels[:, :prompt_len] = -100  # loss on the transcript only
            batch["labels"] = labels

            loss = model.thinker(**batch).loss / accumulate
            loss.backward()
            running += loss.item() * accumulate
            seen += 1

            if seen % accumulate == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step = seen // accumulate
                elapsed = time.time() - started
                print(
                    f"  step {step}/{steps} loss {running / accumulate:.4f} · "
                    f"{(len(samples) * epochs - seen) * elapsed / seen / 60:.0f} min left",
                    flush=True,
                )
                running = 0.0

                if step % push_every == 0:
                    save(f"step {step}")
                if deadline and time.time() >= deadline:
                    save(f"time-boxed at step {step}")
                    print(
                        f"\nstopped at {max_minutes:.0f} min (step {step}/{steps}).\n"
                        f"Resume with: --resume {destination}",
                        flush=True,
                    )
                    return

    save("final")
    print(f"\nsaved -> {destination} · {(time.time() - started) / 60:.1f} min")
    print(
        "\nNow score it. Script mix is not the metric -- it read 100% for a model\n"
        "that was twice as wrong. CER against data/eval/reference.txt is, and\n"
        "34.9% is the number to beat (stock 0.6B plus our romanizer)."
    )


if __name__ == "__main__":  # pragma: no cover
    import typer

    typer.run(main)
