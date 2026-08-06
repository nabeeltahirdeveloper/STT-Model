# Roman Urdu Auto-Captioning — Project Specification

**Status:** Pre-build / architecture locked
**Owner:** _[you]_
**Last updated:** 2026-08-06

---

## 1. Scope

### 1.1 What this is

A commercial service that takes an uploaded video and returns burned-in or sidecar
subtitles in **Roman Urdu** (Urdu written in Latin script), handling the
Urdu–English code-switching that is normal in Pakistani speech.

### 1.2 In scope (v1)

- Video upload → audio extraction → transcription → timed Roman Urdu subtitles
- Output formats: `.srt`, `.vtt`, plain transcript
- Code-switched Urdu–English speech (the primary case)
- Offline / batch processing of files
- A canonical Roman Urdu spelling specification (this is a product artifact, not a chore)

### 1.3 Out of scope (v1)

- Real-time / live streaming captions (Qwen streaming mode cannot emit timestamps)
- Translation (Urdu → English) — transcription only
- Text-to-speech / dubbing
- Urdu Nastaliq script as a customer-facing output (used internally only, see §4.4)
- Speaker-attributed captions ("Speaker 1:") — deferred to v2
- Languages other than Urdu/English

### 1.4 Success criteria

| Metric | v1 target | Notes |
|---|---|---|
| CER on held-out code-switched set | < 12% | Primary quality metric |
| Script-normalized WER (SN-WER) | < 25% | Secondary; WER alone is misleading here |
| English word preservation rate | > 90% | English words must stay in English spelling |
| Spelling consistency | > 98% | Same word → same spelling, every time |
| Subtitle timing error | < 200 ms | Perceptual threshold for sync |
| Processing speed | < 0.3× realtime | 10-min video in under 3 min |

**Kill criterion:** if CER on real target content exceeds ~25% after Phase 2,
the fully-automatic product is not viable and the plan pivots to a
human-in-the-loop editor with ASR as a first draft.

---

## 2. Problem

### 2.1 The core linguistic problem

**Roman Urdu has no standard orthography.** A single Urdu word has many valid
Latin spellings:

| Urdu | Common Roman variants |
|---|---|
| نہیں | nahi, nahin, nai, nhi |
| ہے | hai, hei, he, hae |
| کیا | kya, kia, keya |

Three consequences that drive the entire architecture:

1. **WER is a broken metric.** It scores `nahin` vs `nahi` as a full error even
   though both are correct. Use CER and SN-WER instead.
2. **Consistency is the perceived quality.** Two systems with identical accuracy
   will be judged very differently if one spells inconsistently. Whoever defines a
   clean, consistent convention owns quality perception in this category.
3. **The spelling spec must be applied to training labels, not at inference.**
   In an end-to-end model, label inconsistency is baked into the weights permanently.

### 2.2 The code-switching problem

Everyday Pakistani Urdu mixes English constantly: *"meeting kal hai," "record kar
lo," "problem ye hai ke..."*. This is not an edge case; it is the dominant case.

This has been measured. On a code-switched Pakistani Urdu benchmark:

| Model | WER (no code-switch) | WER (code-switched) |
|---|---|---|
| Whisper-Large-v3 | 0.289 | **0.532** |
| OmniASR-LLM-1B | 0.295 | 0.499 |
| Gemini 2.5 Pro | 0.023 | **0.028** |

Documented failure modes:
- **Whisper** transliterates or *translates* English into Urdu script instead of
  keeping the literal content.
- **OmniASR** hallucinates in Arabic/Persian and loops words on accented segments.
- **Whisper language ID** cannot distinguish Hindi from Urdu (they are the same
  spoken language) and emits Devanagari unpredictably.

### 2.3 Why the naive architecture fails

The obvious pipeline — `audio → Urdu script → Roman Urdu` — was tried and failed:

- English words round-trip through Perso-Arabic script and come back mangled
  (`meeting` → میٹنگ → `mitting` / `meetng`, inconsistently)
- Script selection is unstable (Devanagari appears)
- Two stages compound errors
- No amount of transliteration tuning fixes an English word that was destroyed
  at the ASR stage

**This architecture is rejected for the text path.** It survives only as a
possible fallback for *timing* (§4.4).

---

## 3. Solution

### 3.1 Core decision: end-to-end audio → Roman Urdu

Train a single model that emits Roman Urdu directly. Rationale:

| Property | Why it matters here |
|---|---|
| Same alphabet for both languages | `meeting` and `kya` coexist natively — no script boundary to mangle |
| Better tokenization | Latin text is far cheaper in the model's BPE than Nastaliq |
| No compounding errors | One stage, not two |
| Cheaper inference | One model to serve |

### 3.2 Proof this works (sister language)

**Srota** — a Hinglish (Hindi-English code-switched) fine-tune released May 2026,
built to solve the exact same two failure modes (all-Devanagari collapse, mangled
English words):

- Base: `Qwen/Qwen3-ASR-0.6B`
- **Full-parameter** fine-tune of ~780M params (0.6B LLM + ~180M audio encoder + projector) — **no LoRA, no frozen layers**
- ~95h training data, 2 epochs (~3,352 steps)
- AdamW, LR 2e-5 linear, warmup_ratio 0.02, effective batch 32, bf16 + FlashAttention 2
- **Language-agnostic decoding** — target prefix `language None<asr_text>...`
- Result: conversational WER **24.73% → 15.85%**
- Cost: 2× H100 via Modal, **~49 minutes, ~$6.50**

The compute is trivial. **All the cost and risk is in data preparation.**

Note the LoRA finding: independent evidence from a Pashto fine-tuning study also
found vanilla full fine-tuning beat LoRA by 33.36 percentage points. **Default to
full fine-tuning. Do not reach for LoRA without testing it against FFT first.**

---

## 4. Architecture

### 4.1 Pipeline

```
┌─────────────┐
│ 1. INGEST   │  video upload → ffmpeg → 16 kHz mono 16-bit PCM WAV
└──────┬──────┘
       ▼
┌─────────────┐
│ 2. PREPROC  │  Demucs (music/noise separation)
│             │  pyannote 3.1 (diarization)
│             │  segment: drop < 2s, split > 35s
└──────┬──────┘
       ▼
┌─────────────┐
│ 3. ASR      │  fine-tuned Qwen3-ASR → Roman Urdu text
│             │  offline mode, language-agnostic decoding
└──────┬──────┘
       ▼
┌─────────────┐
│ 4. NORMALIZE│  canonical spelling map (§5.3)
│             │  English word passthrough guard
└──────┬──────┘
       ▼
┌─────────────┐
│ 5. ALIGN    │  forced alignment → word timestamps
│             │  ≤ 5-min chunks, offsets reassembled
└──────┬──────┘
       ▼
┌─────────────┐
│ 6. SUBTITLE │  line breaking, reading-speed limits, SRT/VTT
└──────┬──────┘
       ▼
┌─────────────┐
│ 7. SERVE    │  vLLM + FastAPI + job queue
└─────────────┘
```

### 4.2 Stack

| Layer | Choice | License | Why |
|---|---|---|---|
| ASR base | `Qwen/Qwen3-ASR-0.6B` (iterate) → `1.7B` (prod) | Apache 2.0 | Proven on Hinglish; commercial use fine |
| Forced aligner | `Qwen/Qwen3-ForcedAligner-0.6B` | Apache 2.0 | 32.4 ms avg error; holds on long audio |
| Aligner fallback | MMS + torchaudio CTC alignment | CC-BY-NC / check | Covers 1000+ languages incl. Urdu |
| Source separation | Demucs | MIT | Used by UrduSpeech on this exact audio type |
| Diarization | pyannote 3.1 | MIT (models gated) | Same |
| Transliteration (labels) | IndicXlit (~11M) | MIT | SOTA on Dakshina; supports Urdu |
| Spelling reference | Roman-Urdu-Parl vocab | Research | 42,927-word frequency-ranked Roman vocab |
| Inference server | vLLM (`qwen-asr-serve`) | Apache 2.0 | Day-0 Qwen3-ASR support |
| API | FastAPI | MIT | |
| Scoring | jiwer + SN-WER | MIT | CER + script-normalized WER |
| Media | ffmpeg | LGPL/GPL | |
| Experiment tracking | Weights & Biases or MLflow | — | Non-negotiable for a training project |

### 4.3 ⚠️ Critical constraint: Urdu is not an officially supported language

**Qwen3-ASR supports 30 languages. Urdu is NOT among them. Hindi IS.**
**Qwen3-ForcedAligner supports 11 languages. Neither Urdu nor Hindi is among them.**

This is survivable but must be handled explicitly:

- **ASR side:** low risk. Hindi coverage means the encoder has heard Hindustani
  phonetics at scale, and fine-tuning *adds* the capability rather than relying on
  it. Use Srota's language-agnostic decoding prefix rather than forcing a language
  token. Do **not** trust the built-in language ID.
- **Aligner side:** real gap. See §4.4.

### 4.4 Timestamp strategy (ordered — test in this order)

1. **Direct.** Feed Roman Urdu text to `Qwen3-ForcedAligner` with `language="English"`.
   The output is Latin script and the aligner matches graphemes to acoustics; it
   reports a cross-lingual benchmark at 34.2 ms. **Test this first — 30 minutes.**
2. **Bridge.** Align on an Urdu-script transcript, then carry timings across to the
   Roman tokens (romanization is word-for-word, so timings transfer). Reintroduces
   an Urdu-script intermediate *for timing only*, never for text.
3. **MMS/torchaudio CTC.** Lower accuracy, but definitely covers Urdu.

Additional hard constraints:
- Aligner caps at **5 minutes** → chunk long audio, reassemble with offsets.
- **Streaming mode cannot return timestamps** → captions use offline mode only.

### 4.5 Repository layout

```
roman-urdu-captions/
├── CLAUDE.md                    # agent operating brief
├── README.md
├── pyproject.toml
├── docs/
│   ├── PROJECT.md               # this file
│   ├── SPELLING-SPEC.md         # canonical Roman Urdu orthography
│   └── DECISIONS.md             # ADR log
├── data/
│   ├── raw/                     # UrduSpeech, Common Voice (gitignored)
│   ├── labels/                  # romanized + normalized transcripts
│   └── eval/                    # held-out benchmark, hand-corrected
├── src/
│   ├── ingest/                  # ffmpeg, format normalization
│   ├── preprocess/              # demucs, pyannote, segmentation
│   ├── labeling/                # romanization + normalization pipeline
│   ├── training/                # fine-tuning scripts, configs
│   ├── inference/               # ASR + aligner wrappers
│   ├── subtitle/                # SRT/VTT assembly, line breaking
│   └── api/                     # FastAPI service
├── scripts/                     # one-off CLI utilities
├── tests/
└── notebooks/                   # Colab experiments (not production)
```

---

## 5. Data

### 5.1 Speech corpora

| Dataset | Hours | Role | Access |
|---|---|---|---|
| **UrduSpeech US-CS** | **89.4** | **Primary — code-switched conversational** | Free/open |
| UrduSpeech US-Std | 59.2 | Standard Urdu acoustics | Free/open |
| UrduSpeech US-EngPk | 7.3 | Pakistani-accented English | Free/open |
| UrduSpeech benchmark | 9 | Human-verified eval | Free/open |
| Common Voice Urdu | ~81 | Volume (few unique speakers) | Open |
| FLEURS Urdu | 12 | Benchmarking | Open |

UrduSpeech is 156h total / 71,792 utterances, sourced from in-the-wild YouTube and
archival PTV covering vlogs, street interviews, podcasts, news and drama —
i.e. real target content, not read speech.

**⚠️ Verify before building on it:** the corpus is recent (May 2026) and from a
single lab. Confirm the 91 GB is actually downloadable and check the license terms
before any commercial dependency.

### 5.2 Text corpora (for romanization + spelling)

| Dataset | Size | Role |
|---|---|---|
| **Roman-Urdu-Parl** | 6.37M sentence pairs, 42,927 Roman vocab | Canonical spelling frequency source |
| Dakshina (Urdu) | 10K pairs, Wikipedia domain | Domain generalization |
| Aksharantar | 26M word pairs, 21 languages | IndicXlit training data |

Roman-Urdu-Parl deliberately captures spelling variation *and* frequency —
e.g. `hai / hei / he / hae` with `hai` dominant. **This is the canonical spelling
list, pre-built. Do not hand-build a dictionary.**

### 5.3 Label generation pipeline

```
UrduSpeech US-CS (Urdu script, code-switched)
        │
        ▼
[romanize]  IndicXlit or LLM pass
        │   → keep English words in English orthography
        ▼
[normalize] map each token to canonical spelling
        │   → ranked by Roman-Urdu-Parl frequency
        ▼
[QA sample] hand-check 500 random segments
        │
        ▼
(audio, Roman Urdu) training pairs
```

**This stage is the highest-leverage work in the project.** Model quality is
capped by label quality.

**⚠️ Legal:** if a commercial LLM API is used to generate labels, check its terms
regarding training competing models. Get a lawyer's view before taking revenue.

### 5.4 Two-phase training curriculum

| Phase | Data | Teaches | Cost |
|---|---|---|---|
| 1 | Romanized Common Voice + FLEURS + US-Std (~150h) | How to spell Roman Urdu | Cheap, automated |
| 2 | UrduSpeech US-CS + own domain audio (~90h+) | Code-switching behaviour | Expensive, hand-verified |

Phase 1 warm-starts the model so Phase 2 needs less hand-labeled data.
**Neither phase is optional.** Read-speech corpora contain essentially zero
code-switching, so Phase 1 alone teaches spelling and nothing about the actual problem.

---

## 6. Evaluation

### 6.1 Build the eval set before writing any training code

- 30–60 minutes of **your actual target content type**
- Hand-transcribed in the canonical Roman convention
- Never used for training, ever
- Plus the 9-hour UrduSpeech benchmark, romanized identically

### 6.2 Metrics

| Metric | Use |
|---|---|
| **CER** | Primary — robust to spelling variance |
| **SN-WER** | Script-normalized WER; reduces inflated Urdu error rates by 6.4–9.0% |
| Normalized WER | Map spelling variants to canonical forms before scoring |
| English preservation rate | % of English words emitted in English spelling |
| Spelling consistency | Same word → same spelling across the corpus |
| Alignment AAS (ms) | Timestamp accuracy |

**Never report raw WER as the headline number.** It systematically misrepresents
Roman Urdu quality.

### 6.3 Error breakdown — the diagnostic that gates everything

Classify every error as:
- **(a) acoustic** — model misheard the words → fix with more/better training data
- **(b) orthographic** — words correct, spelling wrong → fix in the normalizer
- **(c) code-switch** — English word mangled → fix with label quality
- **(d) timing** — text right, sync wrong → fix in the aligner

The (a)/(b) ratio decides where the next month of effort goes. **This has not been
measured yet.** It is the first thing to produce.

---

## 7. Risks

| # | Risk | Impact | Mitigation |
|---|---|---|---|
| R1 | UrduSpeech not actually downloadable / restrictive license | **Critical** — plan collapses | Verify in week 1 before any other work |
| R2 | UrduSpeech labels are machine-generated (98% >0.9 confidence, only 9h human-verified) | High | Hand-verify a sample; budget for correction |
| R3 | Forced aligner doesn't handle Roman Urdu | Medium | Three fallbacks in §4.4 |
| R4 | LLM API terms prohibit training-data use | High (legal) | Read ToS before pipeline build; legal review before revenue |
| R5 | Romanizer introduces systematic errors into labels | High | QA sample every batch; hold out a hand-labeled control |
| R6 | Real content is harder than benchmarks (music, overlap, noise) | High | Eval set must be real content from day one |
| R7 | Target content type still undecided | Medium | Blocks scoping; decide before Phase 2 |
| R8 | Spelling spec churns after training | Medium | Freeze the spec before Phase 2 begins |

---

## 8. Roadmap

### Phase 0 — Verify & baseline (week 1)
- [ ] Download UrduSpeech; confirm size, license, integrity → **gate on R1**
- [ ] Build 30–60 min eval set from real target content
- [ ] Run stock `Qwen3-ASR-1.7B` on it → baseline numbers
- [ ] Test forced aligner on Roman Urdu with `language="English"` → resolve §4.4
- [ ] Produce the error breakdown from §6.3

### Phase 1 — Labels & spelling spec (weeks 2–3)
- [ ] Extract frequency-ranked vocab from Roman-Urdu-Parl
- [ ] Write `SPELLING-SPEC.md`; **freeze it**
- [ ] Build romanize + normalize pipeline
- [ ] Generate Phase 1 training labels; QA 500 samples

### Phase 2 — Train (week 4)
- [ ] Full fine-tune `Qwen3-ASR-0.6B` on Phase 1 data (Srota hyperparameters)
- [ ] Evaluate; compare against Phase 0 baseline
- [ ] Add Phase 2 code-switched data; retrain
- [ ] Scale to `1.7B` only once the recipe works on `0.6B`

### Phase 3 — Product (weeks 5–7)
- [ ] Chunking + offset reassembly for long video
- [ ] Subtitle assembly: line breaks, reading speed, SRT/VTT
- [ ] FastAPI service + vLLM serving + job queue
- [ ] Upload UI

### Phase 4 — Harden (week 8+)
- [ ] Speaker diarization labels
- [ ] Human-in-the-loop correction editor
- [ ] Domain-specialist model variants

---

## 9. Subtitle formatting rules

Timestamps alone don't make good captions. Enforce:

| Rule | Value |
|---|---|
| Max characters per line | 42 |
| Max lines per caption | 2 |
| Min caption duration | 1.0 s |
| Max caption duration | 7.0 s |
| Max reading speed | 17 chars/sec |
| Line breaks | At phrase boundaries, never mid-phrase |
| Gap between captions | ≥ 100 ms |

---

## 10. Open questions

1. **What is the target content type?** Podcasts, user-uploaded general, news, or
   vlogs? This changes difficulty, data mix, and whether v1 is viable at all.
2. **What is the (a)/(b) error ratio** on real content? (§6.3)
3. **Does the forced aligner handle Roman Urdu?** (§4.4, test 1)
4. **Fully automatic, or human-in-the-loop editor?** At ~40% WER the editor *is*
   the product. Decide after Phase 0 numbers land.
5. **Is UrduSpeech genuinely open for commercial use?** (R1)

---

## 11. Reference

| Resource | Link |
|---|---|
| Qwen3-ASR repo + finetuning guide | https://github.com/QwenLM/Qwen3-ASR |
| Qwen3-ASR technical report | https://arxiv.org/abs/2601.21337 |
| UrduSpeech corpus paper | https://arxiv.org/abs/2605.17846 |
| Srota (Hinglish reference fine-tune) | https://huggingface.co/moorlee/qwen3-asr-0.6b-hinglish |
| IndicXlit | https://github.com/AI4Bharat/IndicXlit |
| Roman-Urdu-Parl | https://dl.acm.org/doi/10.1145/3464424 |
| WER We Stand (Urdu ASR benchmark) | https://aclanthology.org/2025.coling-main.397/ |
