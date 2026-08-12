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

**The labels teach Roman output. The adapter is nevertheless worse than doing
nothing.**

Scored on the same 12 eval clips, same base model, same reference:

| | CER |
|---|---|
| Stock 0.6B + our romanizer | **34.9%** |
| Adapter 0.6B | **64.1%** |

Script mix, over 8 clips:

| | Latin | Devanagari |
|---|---|---|
| Stock | 5.7% | 94.3% |
| Adapter | 100% | 0% |
| Reference | 100% | 0% |

Output length matches the reference within 2% (474 words against 466), so this
is real transcription and not a collapse. The output also uses `aik`, `woh`,
`bohat`, `nahi` -- the canonical forms chosen in ADR-011 and ADR-012. The model
learned this project's orthography.

**And it is still nearly twice as wrong as not training at all.**

## The mistake this experiment records

Script mix was reported as the headline for three runs. It measures the
alphabet, not whether the words are right, and a model producing fluent Roman
nonsense scores 100% on it. CER -- the metric the product is judged on -- was
not measured until the fourth attempt, and it showed the opposite of what the
script number implied.

The warning was visible and ignored: training loss fell to 4.4 by step 71 and
then rose steadily to ~5.3 by step 866, under a decaying schedule. A rising
training loss means the run is not converging. That was the moment to stop and
measure accuracy; instead it was written up as an "open concern" underneath a
success headline.

**Rule for any successor: measure CER on a dozen clips before and after. Never
let a proxy metric stand in for the one that matters.**

## What this cost, and what it bought

Three runs of 37, 53 and 99 minutes. One durable finding -- the labels teach
Roman output and our orthography -- which the *first* run already established.
Everything after it produced no new knowledge that a 48-second train-and-
generate check would not have surfaced faster.

Probable cause of the regression: the run never converged. Learning rate 1e-4 at
rank 16 is likely too high; 2e-5 with warmup is the obvious next setting. But
the open question -- *does training on these labels produce a better model* --
is the one ADR-003 says needs full fine-tuning on real hardware, and constraint
6 requires that A/B regardless. This machine cannot answer it.
