# lora_mps

**Question.** Do the generated labels teach the model to emit Roman Urdu?

Stock Qwen3-ASR emits **Devanagari** for Urdu speech (ADR-010) — that is the
measured reason fine-tuning is needed at all. If a small LoRA adapter, trained
on our labels, shifts the output script toward Latin, then the label pipeline
works. That validates the expensive artifact (the labels) using a cheap and
disposable one (the adapter).

**What this cannot answer.** Whether LoRA is the right method, or what CER a
properly trained model would reach. CLAUDE.md constraint 6 requires an A/B
against full fine-tuning, and that A/B cannot run on 16 GB of shared memory —
full FT of ~780M parameters needs ~12.5 GB of optimizer state before
activations. **No number from here may be quoted against the 27.9% stock
baseline** (ADR-016): different model size, different data volume, different
method.

## Run

```bash
# training lives in scripts/ because it is reusable; evaluation is the experiment
uv run python -m scripts.train
uv run python -m experiments.lora_mps.evaluate_adapter --clips 8
```

## Setup used

| | |
|---|---|
| Base | `Qwen/Qwen3-ASR-0.6B` |
| Method | LoRA r=16 on q/k/v/o projections, audio encoder frozen |
| Trainable | 6,135,808 params (0.78%) |
| Data | 8,676 clips, 13.5 h — the 20 h subset minus clips whose audio was not yet downloaded |
| Hardware | M4, 16 GB shared, MPS |
| Time | 37.4 min, 1 epoch, batch 1 × accumulate 16 |
| Loss | 9.43 → ~4.0 |

Two `nan` losses appeared (steps 23 and 89) and did not recur. On MPS at fp32
that is most likely a single bad batch rather than divergence, but if the rate
rises in a longer run it wants investigating before the result is trusted.

## Result

**Yes — the labels teach Roman output.** Script mix over 8 eval clips:

| | Latin | Devanagari |
|---|---|---|
| Stock | 5.7% | 94.3% |
| **Adapter** | **48.3%** | 51.7% |
| Reference | 100% | 0% |

37 minutes on 13.5 hours moved Latin share from 5.7% to 48.3%. The question
this experiment existed to answer is answered: the label pipeline works.

## Bug found: the decoding prefix leaks into the output

Some outputs begin `language Hindi super stars nazar nahi aaya...` — and
`language Hindi` is the prompt, not speech. The model learned to emit its own
prefix.

Cause, in `scripts/train.py`:

```python
batch["labels"] = batch["input_ids"].clone()
```

Loss is computed over every token, prompt included, so the model is trained to
predict the prompt as well as the transcript. The prompt span must be masked to
`-100` so loss falls only on the label text. This also explains the two `nan`
losses and one output collapsing into repetition: the model is optimizing two
objectives at once.

**So 48.3% understates what these labels can do.** Half the model's output
budget is being spent reproducing a prefix it should never emit. Re-run after
the masking fix before drawing any conclusion about how far the labels get.
