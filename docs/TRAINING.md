# Training Guide

How to fine-tune the Roman Urdu ASR model on your own machine, start to finish.

Written for someone cloning this repository for the first time. It assumes a
Linux box with an NVIDIA GPU and nothing else.

---

## 0. What you are training, and why

A fine-tune of **Qwen3-ASR** that transcribes Urdu–English code-switched speech
directly into **Roman Urdu** (Latin script), not Perso-Arabic.

The stock model emits Devanagari for Urdu words. Passing that through a
transliterator gives **CER 34.9%** — the baseline every result is measured
against. Fine-tuning on 20 hours moved that to **16.4%**.

| | Stock 0.6B + romanizer | Fine-tuned 0.6B |
|---|---|---|
| CER | 34.9% | **16.4%** |
| English preserved | 41.9% | **82.5%** |
| Orthographic share of errors | 39.8% | 22.3% |

Targets are CER **< 12%** and English preservation **> 90%**. Neither is met yet.

---

## 1. Hardware

Full fine-tuning, batch size 1, 8-bit Adam, gradient checkpointing:

| Model | Trainable params | Fixed VRAM | Verdict on 16 GB |
|---|---|---|---|
| Qwen3-ASR-0.6B | 782 M | ~6 GB | fits, ~10 GB peak |
| Qwen3-ASR-1.7B | **2.04 B** | ~12.2 GB | **OOMs on long clips** |

Fixed cost is `params × 6 bytes` — bf16 weights + bf16 gradients + 8-bit Adam
states — before any activations. Activation memory scales with clip length, and
the corpus has clips up to 20 s.

**Measured:** 1.7B ran to step 303 of 865 on a T4 (14.56 GB usable), peaking at
14.4 GB, then OOMed in `backward()`. It is not a tuning problem; it is
arithmetic.

**Recommended:** A100 40 GB or better for 1.7B. Any 16 GB card is fine for 0.6B.

Disk: ~15 GB for the 20-hour subset, ~60 GB for the full corpus.

---

## 2. Setup

```bash
git clone <this-repo> && cd model
./run.sh setup            # installs Python 3.12 + all dependencies via uv
./run.sh doctor           # reports anything missing
./run.sh check            # lint + typecheck + 376 tests, all must pass
```

There is **no virtualenv to activate**. `uv run` resolves the environment per
invocation. Never use `pip install` — use `uv add`, and commit `uv.lock`.

You need a HuggingFace token for the dataset:

```bash
export HF_TOKEN=hf_...     # or: uv run hf auth login
```

---

## 3. The data pipeline

Four steps, in this order. Order matters — step 3 must exclude step 2's output.

```bash
# 1. Transcripts (~1 min). Text only, no audio.
uv run python -m scripts.download_transcripts --corpus-transcripts

# 2. Labels: Urdu -> Roman via dictionary lookup, not a neural model (ADR-014).
#    ~10 seconds for 29,749 utterances. Regenerated, never committed.
uv run python -m scripts.build_labels --split US-CS

# 3. Dev set: 150 held-out clips for measuring CER *during* training.
uv run python -m scripts.build_dev_set --clips 150

# 4. Training subset, EXCLUDING the dev set.
uv run python -m scripts.select_training_subset --hours 20 \
    --exclude data/labels/dev-set.jsonl

# 5. Audio. --archive pulls one 12.5 GB tar instead of 13,470 files.
uv run python -m scripts.fetch_training_audio \
    --manifest data/labels/train-subset.jsonl \
    --archive MubeenAmjad205/roman-urdu-captions-audio
```

### Why `--archive` matters

The HuggingFace Hub throttles by **request count** — roughly 1,000 per 5
minutes — and one clip is one request. 13,470 clips therefore have a floor near
**35 minutes** no matter how many threads you use. Raising concurrency past 8
produces 429s, not throughput: an early attempt with 16 workers lost 11,910 of
13,470 clips.

One tar is one request, at CDN speed: **2–3 minutes**. If the archive is
missing the script falls back to per-clip automatically, so the flag is always
safe to pass.

### Why the dev set must be excluded

`--exclude` keeps the 150 dev clips out of training. Without it the mid-run CER
measures recall on memorised audio, not generalisation, and reads far better
than the model deserves. A test asserts the manifests never overlap.

---

## 4. Rehearse before you train

**Always run this first.** It costs ~4 minutes and exercises every line the real
run will execute: select, fetch, load, forward, backward, generate, save.

```bash
uv run python -m scripts.select_training_subset --hours 0.05 \
    --out data/labels/rehearsal.jsonl
uv run python -m scripts.fetch_training_audio --manifest data/labels/rehearsal.jsonl

uv run python -m src.training.finetune \
    --manifest data/labels/rehearsal.jsonl \
    --base Qwen/Qwen3-ASR-0.6B \
    --limit 22 --accumulate 4 --probe-every 5 \
    --out /tmp/rehearsal
```

Read three things from the output:

**`GB peak`** — the memory ceiling. The rehearsal uses the *shortest* clips in
the corpus, so the real run will exceed this. If the rehearsal is already near
your card's limit, the real run will OOM.

**`p(eos)`** — probability the model assigns to the stop token where stopping is
correct. Should climb toward 1.0. Pinned near zero means the model is not
learning to terminate.

**The probe** — two held-out clips transcribed and printed beside their labels.
At step 5 the model is still essentially the base model, so Devanagari output
here is expected and not a failure.

Two bugs cost full sessions because the rehearsal only trained and never
generated. It generates now. That is the entire point of it.

---

## 5. Train

```bash
uv run python -m src.training.finetune \
    --base Qwen/Qwen3-ASR-0.6B \
    --out out/finetuned \
    --max-minutes 180 \
    --hf-repo <your-user>/<your-model> \
    --push-every 300 \
    --dev-manifest data/labels/dev-set.jsonl \
    --dev-every 150
```

### Every flag

| Flag | Default | What it does |
|---|---|---|
| `--base` | `Qwen/Qwen3-ASR-0.6B` | Model to fine-tune |
| `--manifest` | `data/labels/train-subset.jsonl` | Training clips |
| `--out` | `out/finetuned` | Checkpoint directory |
| `--epochs` | 1 | Passes over the data |
| `--learning-rate` | 2e-5 | OneCycle schedule with warmup |
| `--accumulate` | 16 | Gradient accumulation; effective batch size |
| `--precision` | `bf16` | `auto`/`fp16`/`bf16`/`fp32` — see below |
| `--checkpointing` | on | `--no-checkpointing` is ~30% faster, more memory |
| `--max-seconds` | 20 | Skip longer clips. **Lower this if you OOM** |
| `--limit` | 0 | Stop after N clips (0 = all) |
| `--max-minutes` | 0 | Time-box; saves and exits (0 = no limit) |
| `--resume` | — | Continue from a checkpoint directory |
| `--hf-repo` | — | Push to HuggingFace during the run |
| `--push-every` | 400 | Steps between pushes |
| `--probe-every` | 50 | Steps between generation probes |
| `--dev-manifest` | — | Held-out clips for mid-run CER |
| `--dev-every` | 150 | Steps between CER measurements |
| `--dev-clips` | 25 | Clips scored per CER pass |
| `--warmup` | 50 | Warmup steps |

Full list: `uv run python -m src.training.finetune --help`

### On `--precision`

bfloat16 has **no tensor-core path before Ampere (sm_80)**. On Turing cards
(T4, sm_75) bf16 works but never touches the fast silicon, and inference in bf16
there multiplies memory. `--precision auto` selects fp16 below sm_80 and bf16 at
or above it.

The default is `bf16`, not `auto`, deliberately: pure fp16 keeps no fp32 master
weights and is the less numerically forgiving of the two. Change it as a
deliberate experiment against a known-good baseline, not by accident.

### Reading the log

```
step 300/865 loss 0.8869 · p(eos) 0.735 · 14.4 GB peak · 142 min left
  --- probe at step 300 ---
    label(  12): Mere husband
    model(  18): Mere husband ya ho
    chars 1.50x
  >>> step 300: dev CER 15.7% on 25 held-out clips  (target < 34.9%)
```

**`dev CER` is the only line that matters.** Falling means it is working. Flat
or rising by step 450 means stop — do not wait for the final evaluation.

Loss alone is not enough and has misled this project repeatedly. One run drove
loss from 13.56 to 0.35 while learning to reproduce audio padding. `p(eos)`,
script mix and character ratios are all proxies; CER is the goal.

---

## 6. Evaluate

```bash
# 1. Eval audio (gitignored, so a fresh clone does not have it)
uv run python -m scripts.fetch_training_audio --manifest data/eval/manifest.jsonl

# 2. Transcribe the 269-clip benchmark
uv run python -m scripts.run_baseline \
    --model-id out/finetuned \
    --out-dir out/preds \
    --no-romanize-output \
    --device cuda --batch-size 4

# 3. Score
uv run python -m src.eval.score --pred out/preds/baseline-raw.txt
```

**`--no-romanize-output` is required for a fine-tuned model.** It already emits
Roman; sending its output through the Devanagari converter a second time
rewrites correct spellings and measures the converter instead of the model.
Omit the flag only for the stock-model baseline.

**`--batch-size`**: generation holds a KV cache per clip and pads the batch to
its longest member. Batch 8 pushed even a 0.6B model into OOM retries on a T4.
Use 4, or 2 if memory is tight.

### Reading the score

```
cer                     16.4%   FAIL  (target < 12%)
sn-wer                  33.1%   FAIL  (target < 25%)
english-preservation    82.5%   FAIL  (target > 90%)

§6.3 error breakdown
  acoustic        70.1%   more/better training data
  orthographic    22.3%   normalizer and lexicon work
  code-switch      7.6%   label quality
```

**Never quote raw WER as a headline.** Roman Urdu has no standard orthography,
so WER charges a full error for a correct-but-variant spelling. Use CER and
SN-WER (ADR-005).

The error breakdown tells you where to spend effort. At 70% acoustic, more or
better audio data and a larger model help. At 40% orthographic — where this
project started — the fix is the lexicon and romanizer, and costs no GPU time.

---

## 7. Hard rules

These are not style preferences. Violating them invalidates results.

**Never train on `data/eval/`.** It is the only honest measurement you have.
`select_training_subset` and `build_dev_set` both draw from `data/labels/`,
which is a disjoint split, and tests assert zero overlap. Do not add eval clips
to any manifest, not even "just to check something".

**Never route text through Urdu or Devanagari script.** Audio → Urdu → Roman was
tried and failed; it mangles English words. Urdu script is permitted only as a
timing bridge, never for customer-facing text.

**English stays in English orthography.** `meeting`, not `mitting`. Any code
path that phonetically re-spells an English word is a bug.

**Do not use LoRA without an A/B against full fine-tuning.** A LoRA run on these
labels scored CER 64.1% against 34.9% for the stock model plus the romanizer —
worse than not training. Full fine-tuning is the default (ADR-003, ADR-016).

**Do not force `language="Urdu"`.** Urdu is not among Qwen3-ASR's supported
languages, and the built-in language ID is unreliable on it. Decoding is
language-agnostic by design.

**Rehearse before every long run.** Four minutes against four hours.

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA out of memory` in `backward()` | Model too large for the card | Lower `--max-seconds`, keep `--checkpointing`, or use a bigger GPU |
| Model outputs nothing (0 chars) | Loss mask covering the `<asr_text>` tag | Fixed in `mask_prompt`; verify with the decode check in §9 |
| Output repeats forever | Missing EOS in the training target, or mask scoring the turn scaffolding | Both fixed; `p(eos)` should climb |
| Eval takes hours, GPU at 0.0 GB | Model left on CPU | Pass `--device cuda` |
| `429 Too Many Requests` on download | Hub request rate limit | Use `--archive`, or wait; the retry with backoff handles the rest |
| 36 eval clips 404 | `Podcast2/` is a macOS-only path | Handled by `REPO_ALIASES` in the fetcher |
| Dev CER wildly unstable | Reference clips too short | `build_dev_set` selects near the median length |
| `GenerationConfig is invalid` on save | Qwen ships `temperature` with `do_sample=False` | Handled by `sanitize_generation_config` |

---

## 9. Verifying the training contract

The two costliest bugs in this project were both invisible to the loss curve and
obvious the moment the tensors were decoded. If you change anything about the
input format, run this:

```bash
uv run pytest tests/test_loss_mask.py tests/test_training_target.py -v
```

What must hold, on a real clip:

```
MASKED  (given to the model): '...<|audio_end|><|im_end|>\n<|im_start|>assistant\n'
SCORED  (must be generated) : '<asr_text>Mere husband<|im_end|>'
```

The scored region **starts with `<asr_text>`** — the model must learn to emit
the tag, or it never starts — and **ends with EOS** — or it never stops. Audio
padding and turn scaffolding must not appear in it.

---

## 10. Recommended plan for a larger machine

In order, cheapest information first:

1. **0.6B, 40 hours** instead of 20. Establishes whether data scaling helps.
   This is the biggest untested assumption in the project.
2. **0.6B, 2–3 epochs.** One pass is rarely convergence.
3. **1.7B, full corpus**, on 40 GB+. Blocked on a T4 by arithmetic, not tuning.
4. **Demucs preprocessing on FILM.** That category scores 41.7% CER against
   ROADSIDE's 7.0% — a 6× spread that looks like music and overlapping speech,
   not model capacity.

### The ceiling nobody can train past

The labels carry roughly **3.5% wrong words and 2% unknowns**, inherited from
the source corpus (Phase 1). A model cannot learn past a wrong label — it learns
the error. If CER stalls around 13% with more data and more epochs, that is the
labels talking, and the next work is in `data/lexicon/` and the romanizer rather
than on a bigger card.
