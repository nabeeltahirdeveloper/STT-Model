# Architecture Decision Record

Append-only. Never rewrite history — supersede instead.

Format: context → options → decision → consequence.

---

## ADR-001 — Reject the two-stage script pipeline

**Date:** 2026-08-06 · **Status:** Accepted

**Context.** The obvious pipeline is `audio → Urdu (Nastaliq) → Roman Urdu`.
It was built and tested. It failed.

**Options.**
1. Two-stage via Perso-Arabic intermediate
2. Two-stage via Devanagari intermediate (better-resourced romanization tooling)
3. End-to-end `audio → Roman Urdu`

**Decision.** Option 3.

**Reasoning.** English words round-trip through Perso-Arabic and come back
mangled — `meeting` → میٹنگ → `mitting`. Measured externally: Whisper-Large-v3
goes 0.289 → 0.532 WER when code-switching is present, specifically because it
transliterates or translates English into Urdu script. Option 2 has the same
defect plus script-selection instability.

In end-to-end Roman, English and Urdu share one alphabet, so there is no script
boundary to destroy the word.

**Consequence.** No public (audio → Roman Urdu) training corpus exists; labels
must be manufactured. Urdu script survives only as a possible *timing* bridge
(PROJECT.md §4.4 option 2), never for text.

---

## ADR-002 — Qwen3-ASR as the base model

**Date:** 2026-08-06 · **Status:** Accepted

**Context.** Need a base checkpoint to fine-tune for Roman Urdu output.

**Options.** Whisper large-v3-turbo · Qwen3-ASR-0.6B/1.7B · Urdu-specific Whisper
fine-tunes (`kingabzpro/whisper-large-v3-urdu`, `ihanif/whisper-medium-urdu`)

**Decision.** Qwen3-ASR — `0.6B` for iteration, `1.7B` for production.

**Reasoning.** Apache-2.0 with explicit commercial use. Proven on the sister
problem: Srota, a Hinglish fine-tune from Qwen3-ASR-0.6B, took conversational WER
24.73% → 15.85% for ~$6.50 of compute, built to fix these exact failure modes.
Ships a matched forced aligner. Whisper has a *documented* failure on code-switched
Urdu. Urdu-script fine-tunes are actively wrong here — their decoders are pushed
toward Nastaliq output.

**Consequence.** ⚠️ Urdu is not among Qwen3-ASR's 30 supported languages (Hindi
is). The forced aligner supports 11 languages, neither Urdu nor Hindi. Mitigations
in PROJECT.md §4.3–§4.4. Use language-agnostic decoding; do not trust built-in LID.

---

## ADR-003 — Full fine-tuning, not LoRA

**Date:** 2026-08-06 · **Status:** Accepted

**Context.** LoRA is the reflexive default for adapting large models cheaply.

**Decision.** Full-parameter fine-tuning is the default. LoRA requires an A/B
against FFT before use.

**Reasoning.** Two independent results point the same way. A Pashto study found
vanilla full fine-tuning beat LoRA (rank 64) by 33.36 percentage points. Srota
used full fine-tuning of ~780M params explicitly with no LoRA and no frozen
layers. At 0.6B scale the compute argument for LoRA is weak — the reference run
cost about $6.50.

**Consequence.** Higher VRAM per run. Negligible cost impact at this scale.

---

## ADR-004 — Spelling consistency is a product artifact

**Date:** 2026-08-06 · **Status:** Accepted

**Context.** Roman Urdu has no standard orthography. Multiple spellings of the
same word are all "correct."

**Decision.** Maintain a canonical spelling specification
(`docs/SPELLING-SPEC.md`), frozen before Phase 2 training, applied to training
labels rather than at inference.

**Reasoning.** In an end-to-end model, label inconsistency is baked into the
weights permanently. Applying normalization at inference cannot fix a model that
learned ambiguity. Two systems with equal word accuracy are judged very
differently on consistency alone, so this is where quality perception is won.

The lexicon does not need to be hand-built: Roman-Urdu-Parl provides 6.37M
sentence pairs and a 42,927-word frequency-ranked Roman vocabulary that already
captures real spelling distribution.

**Consequence.** Spec changes after freeze require a major version bump and a
full retrain. Freeze deliberately, then leave it alone.

---

## ADR-005 — CER and SN-WER over raw WER

**Date:** 2026-08-06 · **Status:** Accepted

**Context.** WER is the standard ASR metric.

**Decision.** CER is primary; SN-WER (script-normalized WER) is secondary. Raw
WER may appear in tables only alongside CER, never as a headline.

**Reasoning.** WER scores `nahin` vs `nahi` as a full error though both are
correct, systematically understating quality for a language with no fixed
orthography. Script-normalized scoring has been shown to reduce inflated Urdu
error rates by 6.4–9.0%.

**Consequence.** Numbers are not directly comparable to published Urdu ASR WER
figures. Always state which metric is being quoted.

---

## ADR-006 — English orthography outranks the variant map

**Date:** 2026-08-06 · **Status:** Accepted

**Context.** `docs/SPELLING-SPEC.md` §8 lists `the` and `they` as rejected
variants of تھے (canonical `thay`), and `main` as a rejected variant of میں
(canonical `mein`). All three are also ordinary English words. A variant map that
contains them will rewrite English text: `the best` becomes `thay best`.

**Options.**
1. Follow §8 literally and accept that some English words get respelled
2. Drop the colliding rows from the variant map
3. Resolve every collision by context at runtime

**Decision.** Option 2 for `the` and `they` — they are absent from
`data/lexicon/canonical.tsv`. Option 3 for `main`, which is declared in
`data/lexicon/ambiguous.tsv` with the Urdu reading as default and a short list of
following nouns (`road`, `gate`, `street`, …) that force the English reading.

**Reasoning.** Respelling an English word is a P0 bug (SPELLING-SPEC §5.1) and
is the specific failure that rejected the two-stage architecture (ADR-001).
A missed variant costs one inconsistent Roman spelling; a wrong English
respelling costs the thing the product is for. Context resolution is only worth
its false-positive risk where the Urdu reading is genuinely frequent, which is
true for `main` (میں = "I") and not for `the`/`they`.

**Consequence.** `data/lexicon/canonical.tsv` deliberately diverges from
SPELLING-SPEC §8 in two rows; `tests/test_lexicon.py` pins the divergence so it
cannot be silently reverted. `Lexicon.load` now refuses to start if a token
appears in both `english.txt` and the Roman Urdu lexicon without a row in
`ambiguous.tsv` — collisions are a load-time error, not a runtime surprise.

Note also that §11 row 7 (`main TV dekh raha tha` → `Main TV dekh raha tha`)
contradicts §8, which makes `mein` canonical. The test for it is `xfail(strict)`
pending the frequency data required by §2.

---

## ADR-007 — uv for dependency management

**Date:** 2026-08-06 · **Status:** Accepted

> Numbered 007, not 006: ADR-006 is already taken by the English-orthography
> decision above, and this log is append-only with stable identifiers.

**Context.** The project was scaffolded with `python -m venv` plus
`pip install -e ".[dev]"`, which produces no lockfile. Two contributors
installing on the same day get different versions of torch, transformers and
pyannote, and neither can reproduce the other's result. For a project whose
central output is a *trained model*, an unreproducible environment means an
unreproducible experiment — the config hash logged with every run (CLAUDE.md >
Code conventions) is only as meaningful as the environment it ran in.

**Options.** pip + venv (status quo) · Poetry · PDM · uv

**Decision.** uv. Dev tooling moves to a PEP 735 `[dependency-groups]` block;
`uv.lock` is committed; `.python-version` pins the interpreter.

**Reasoning.**

- **Lockfile reproducibility.** `uv.lock` is universal — one file resolves
  correctly for macOS laptops and Linux training boxes, so the Mac used for
  Phase 0 normalizer work and the H100 box used for Phase 2 agree on versions.
  Poetry and PDM also lock; pip alone does not.
- **Resolver speed.** Full resolution of 197 packages takes seconds rather than
  minutes. This matters more than it sounds: a slow resolve is a resolve people
  skip, and skipped resolves are how lockfiles drift out of date.
- **uv manages Python itself.** `uv python install 3.12` removes the assumption
  that the right interpreter is already on PATH. This was not hypothetical — the
  machine this migration ran on had only Python 3.14 installed, and the previous
  `run.sh` would have silently proceeded on it after a warning. Poetry and PDM
  manage packages but not interpreters.
- **The upstream stack already recommends it.** The Qwen3-ASR documentation uses
  uv as the environment manager for the vLLM install path, which is the exact
  serving path in PROJECT.md §4.2. Matching upstream's tooling means their
  install instructions work here unmodified.

**Consequence.**

- Contributors now need uv installed. It is a single-command install on every
  platform and replaces both the Python and the pip install, so this is a net
  reduction in setup steps, but it *is* a new hard prerequisite — `./run.sh
  doctor` and `./run.sh setup` both fail fast without it.
- **flash-attn needs a build-isolation escape hatch.** Its `setup.py` imports
  torch to read the CUDA version, so it cannot build in uv's default isolated
  build environment. It is declared as the `cuda` extra, marker-gated to Linux,
  with `no-build-isolation-package = ["flash-attn"]`, and `./run.sh setup` syncs
  twice — once excluding the extra so torch is present, then again so flash-attn
  builds against it. Getting this wrong produces a confusing `ModuleNotFoundError:
  torch` from inside a build backend.
- Torch is resolved from default PyPI. A training box must add the matching
  `[[tool.uv.index]]` for its CUDA version and re-run `uv lock`; the block is
  commented in `pyproject.toml` next to the setting.
- `uv.lock` must be committed with every dependency change, and reviewed like
  code. A lockfile that is edited but not reviewed is worse than no lockfile.

---

## ADR-008 — UrduSpeech as the eval-set source, romanized from its transcripts

**Date:** 2026-08-10 · **Status:** Accepted · Closes the licence half of gate R1

**Context.** PROJECT.md §6.1 wants a 30–60 minute held-out eval set of real
code-switched speech. Two sources were on the table: hand-transcribing target
content from scratch, or the ASLP-lab/UrduSpeech benchmark. Nothing is trained
yet, so this set gates every downstream number (CLAUDE.md > Current status,
items 2 and 3).

**Licence.** CC-BY-4.0, stated in the dataset README — *not* in the HF card
metadata, where `license` is null. Commercial use is permitted with attribution.
Anything shipped from this data must carry the attribution; the null metadata
field means an automated licence check on this repo will report "unknown" and
must not be trusted to clear it.

**Decision.** Build the eval set from UrduSpeech, and romanize from its
*human-verified Urdu transcripts* rather than from ASR output.

**Reasoning.**

- **Correcting a transcript-derived draft is about half the work of correcting
  an ASR-derived one.** From ASR, a reviewer fixes both what the machine heard
  and how it spelled it. From the corpus transcripts the words are already
  verified, so only the romanization is in question. Same reviewer-hours buy
  roughly twice the audio.
- **The corpus code-switches the way the product needs.** Sample rows keep
  English in English orthography inside Urdu-script sentences (`بہت rare ہوتا ہے
  ... anesthesia والوں کے`), which is exactly the pattern constraint 2 protects.
- **Balanced by category, not by convenience.** The first download was 44
  minutes of podcast, which measures podcast performance and nothing else. The
  set now spans 12 categories — comedy, drama, film, news, roadside, vlog,
  reviews, prose, poetry, interviews, two podcast sources — at a per-category
  minute budget, shortest clips first. Categories differ enormously in
  difficulty, and a single-genre eval set cannot show that.

**Consequence.**

- Attribution to ASLP-lab/UrduSpeech is now a shipping requirement.
- **The drafts are still not the eval set.** Machine romanization of a verified
  transcript is still machine output; constraint 5 and ADR-004 both require a
  human ear on every line before `data/eval/reference.txt` exists. The drafts
  live in `data/eval/drafts-corpus/` and carry `reference_status:
  draft-uncorrected` in their manifest until corrected.
- Reviewers flag lines where the *corpus itself* is wrong with a trailing `??`,
  which gives a cheap measure of corpus transcript quality as a side effect.
- Being a public benchmark, UrduSpeech may appear in some future base model's
  training data. That is tolerable for Phase 0 baselining but makes it a weak
  claim of generalisation; PROJECT.md §6.1's "real target content" still needs
  its own set before any external result is published.

---

## ADR-009 — The normalizer's unknown-token default corrupts English

**Date:** 2026-08-10 · **Status:** Proposed — blocks the SPELLING-SPEC freeze

**Context.** Constraint 3 says every Roman Urdu output passes through the
canonical normalizer. Constraint 2 says English words keep English spelling.
The normalizer cannot honour both, because it decides which rule set to apply
by lexicon membership and **treats every unrecognised token as Roman Urdu**.

Measured on the 35-minute eval corpus (`data/eval/drafts-corpus/`):

| | |
|---|---|
| words in `data/lexicon/english.txt` | 393 |
| distinct English words spoken in the audio | 676 |
| of those absent from `english.txt` | 598 (88%) |
| of those the normalizer actively rewrites | 31 |

The rewrites are not cosmetic. `fitness` → `fitnes`, `greatness` → `greatnes`,
`will` → `wil`, `three` → `thri`, `no` → `nau`, `say` → `se`, `he` → `hai`.
The last four are the dangerous class: the output is not merely misspelled
English, it is a *different, valid Roman Urdu word*. Nothing downstream can
detect that it happened.

This is ADR-001 and ADR-006 recurring one layer down. Those decisions kept
English out of Perso-Arabic script in the *model*; this is the same corruption
re-entering through the *normalizer*, after the model got it right.

**Why the eval set does not expose it.** The eval drafts are romanized from
UrduSpeech transcripts, which are code-switched — English appears in Latin
script in the source. That gives an exact mask of which tokens are English, and
masking those tokens produced a clean pass over all 269 lines. **Production has
no such mask.** ASR output is undifferentiated Roman text, so the one signal
that makes the eval pipeline safe is exactly what inference lacks. A green eval
run is therefore not evidence that this is fixed.

**Options.**

1. *Grow `english.txt` from a standard wordlist* (~25k common English words).
   Cheap and immediate, but every English/Roman-Urdu homograph it introduces —
   `he`, `me`, `or`, `say`, `no`, `all`, `will`, `well`, `fall`, `tell`, `off`,
   `add`, `free`, `guess`, `mill`, `three` — becomes a token the normalizer can
   now get wrong in the *other* direction. `ambiguous.tsv` currently holds three
   rows; this would need dozens, each a hand-tuned collocate list.
2. *Invert the default*: an unrecognised token passes through untouched, and
   only tokens present in `canonical.tsv` are rewritten. Safe for English by
   construction, but it silently weakens "one word, one spelling" — Roman Urdu
   variants absent from the map stop being canonicalised, and the map is a seed
   of ~150 rows, not a vocabulary.
3. *Resolve homographs with frequency data.* SPELLING-SPEC §2 and §12 already
   name Roman-Urdu-Parl (6.37M pairs, 42,927-word Roman vocabulary) as the
   authority for contested spellings. The same corpus gives, per token, how
   often it is Roman Urdu versus English, which is precisely the prior needed to
   set each `ambiguous.tsv` default rather than guessing.
4. *Tag English at the model.* Have the fine-tuned ASR emit English spans
   marked, restoring in production the mask the eval set gets free. Most
   robust, most expensive, and unavailable until Phase 2.

**Recommendation.** 1 + 3 together, with 2 as the interim default: stop
rewriting unknown tokens now, grow `english.txt` from a wordlist, and use
Roman-Urdu-Parl frequency to set the ambiguity defaults before the spec freezes.
Option 4 is the right long-term answer and should be revisited when the Phase 2
training data format is designed.

**Consequence.**

- **`scripts/build_lexicon.py` is no longer only about spelling variants.** It
  must also emit the English/Roman-Urdu collision set with frequencies. That
  raises its priority — the SPELLING-SPEC §12 freeze checklist cannot be
  completed honestly without it.
- Until this is resolved, **no normalizer output is trustworthy on English**,
  and the eval reference must be built with the transcript mask, not by running
  the normalizer unguarded.
- Related defects found in the same pass, filed separately from this decision
  because they are bugs rather than design choices: `isi` (اسی) is rewritten to
  the acronym `ISI`; proper nouns are not exempt from the §4.4 final-`h` rule
  (`Shah Rukh` → `Sha Rukh`); and §4.4's own "where the h is pronounced"
  exception is unimplemented, so `panah`, `sarbarah`, `aagah` and `fatah` all
  lose a pronounced consonant.
- The 88% figure is a floor, not a ceiling. It comes from 35 minutes of
  benchmark audio; real customer content will use a wider English vocabulary.

---

## ADR-010 — Phase 0 results: the baseline, and timing strategy 1

**Date:** 2026-08-10 · **Status:** Accepted · Closes Phase 0 items 3, 4, 5

**Context.** Nothing had been measured. PROJECT.md §6.3 called the error
breakdown "the first thing to produce" and §4.4 called the aligner test "the
highest-priority experiment in the project". Both now have numbers, against the
269-utterance eval set built in ADR-008.

### Baseline — stock `Qwen3-ASR-1.7B`, 35.4 min, decoded language-agnostically

| | Raw output | After our romanizer | §1.4 target |
|---|---|---|---|
| CER | 75.8% | **26.7%** | < 12% |
| SN-WER | 131% | 57.4% | < 25% |
| English preserved | 8.9% | 41.9% | > 90% |

**Stock output is Devanagari, not Roman.** Urdu is not among the model's 30
supported languages (checked, not assumed). Scoring Devanagari against a Roman
reference measures script mismatch, so both columns are reported: the raw one is
the evidence that no stock model does this job, and the second is the honest
starting line. Neither alone is the truth.

The 8.9% English-preservation figure is the sharpest result here. Stock output
destroys nine English words in ten — `IPL` → `आईपीएल` → `aaeepeeel`. ADR-001
rejected the two-stage pipeline on this exact failure, previously on other
people's evidence; it is now measured on ours.

Per category, CER after romanization ranges 21.0% (Podcast2) to 41.0% (FILM).

### Error breakdown (§6.3)

| | | |
|---|---|---|
| acoustic | 59.7% | more/better training data |
| orthographic | 38.1% | romanizer and lexicon work |
| code-switch | 2.2% | label quality |

**The first version of this number was wrong and would have misdirected the next
month.** Classifying a substitution as orthographic only when the lexicon maps
both spellings together reported 97.7% acoustic — because the lexicon holds
~150 rows while the errors included 74 × `mein`/`men`, 21 × `rahi`/`rahee`,
11 × `karte`/`karate`. Plainly the same words, respelled.

The fix is not an edit-distance threshold: `mein`→`men` and `aya`→`gaya` are
both one character in four, and one is a respelling while the other is "came"
against "went". `src/eval/metrics.py` now compares **consonant skeletons**.
Roman Urdu variation is almost entirely vowel variation, because Perso-Arabic
does not write short vowels and every writer guesses differently; consonants are
what the source script does record. Same consonants, different vowels → the same
word. This is a heuristic and `hain`/`hai` sits on the wrong side of it, so the
split is a band rather than a decimal to defend.

**Consequence for planning: roughly two-fifths of current errors are fixable
without touching the model.** That is a different plan from what 97.7% acoustic
would have justified.

### Timing — §4.4 strategy 1 works

`Qwen3-ForcedAligner-0.6B` (a separate checkpoint from the ASR model) fed Roman
Urdu with `language="English"`, over 430 word onsets in 20 clips:

| | |
|---|---|
| Median disagreement vs MMS | 46 ms |
| Mean | 90 ms |
| Within the 200 ms sync target | 91% |

Token counts matched the reference on all 20 clips — the aligner never dropped
or invented a word, which §4.4 flagged as the harder failure.

**Decision: strategy 1 (direct). The Urdu-script timing bridge is not needed.**

**Caveats that belong with the number.**

- This is agreement between two aligners, not against hand-marked truth. MMS
  (torchaudio, §4.4 fallback 3) supplied the comparison because it aligns
  romanized text natively. Two machines can be wrong together; a human review of
  a subset is what upgrades this from promising to settled.
- **62 of 430 tokens (14%) come back zero-duration**, `start == end`, spread
  through the clips rather than at the edges, mostly isolated but once five in a
  row. It affects short function words and longer ones alike, so it is not
  simply the aligner's 80 ms output quantization swallowing brief words. A
  caption window can be derived from its first and last word, so this is
  survivable — but co-located timestamps lose word ordering, and subtitle
  line-breaking must not assume durations are positive.
- The eval set is benchmark audio, not the target content PROJECT.md §6.1 wants
  (ADR-008). R6 says real content is harder. Treat 26.7% as optimistic.

**Consequence.**

- The kill criterion (§1.4) is CER > ~25% on real target content *after* Phase 2.
  The pre-training figure is 26.7% on easier audio. Training has to at least
  halve it, and the margin is not comfortable.
- English preservation at 41.9% after romanization, against a > 90% target, makes
  the code-switch problem a Phase 1 labelling priority rather than a Phase 2 one.
- `data/eval/alignment/` holds 20 clips with MMS-drafted onsets awaiting human
  review. Its earlier contents were synthetic — every word given an identical
  duration to the millisecond — and would have produced a confident, meaningless
  timing number had they been scored.

---

## ADR-011 — Corpus frequency overturns 19 canonical spellings

**Date:** 2026-08-10 · **Status:** Accepted

**Context.** Every spelling in `canonical.tsv` was a hypothesis; SPELLING-SPEC §2
names Roman-Urdu-Parl as the authority and says frequency wins on disagreement.
`scripts/build_lexicon.py` is now implemented and the corpus counted: 6.37M
lines, 132M tokens.

**Result: 85 rows agree, 19 disagree, 1 near-tie, 4 unseen.** The disagreements
are not marginal.

| Ours | Corpus | Ratio |
|---|---|---|
| `nahin` 260 | `nahi` 901,811 | 3,468× |
| `bohot` 3,934 | `bohat` 353,879 | 90× |
| `ek` 11,143 | `aik` 745,286 | 67× |
| `wo` 43,143 | `woh` 533,423 | 12× |
| `ye` 53,548 | `yeh` 221,269 | 4× |

Plus `zyada`→`ziyada`, `hun`→`hoon`, `bari`→`barri`, `kitab`→`kitaab`,
`kaun`→`kon`, `yaani`→`yani`, `sahi`→`sahih`, `chai`→`chaye`, `janta`→`jaanta`,
`ana`→`aana`, `baje`→`bajay`, `mahina`→`maheena`, `tumhein`→`tumhen`,
`chhay`→`chay`. Counts verified independently with `grep -c` before acting,
because a 3,468:1 ratio is the shape of a tokenizer bug as much as a finding.

**Decision.** Adopt all 19. `canonical.tsv`, SPELLING-SPEC §8 and the test suite
now carry the corpus spellings.

**Reasoning.**

- §2 already made this call in advance. The value of a written tie-break is that
  it binds when the answer is inconvenient.
- **The corpus is not applying a blanket rule**, which is the thing that would
  make it an artifact rather than evidence. It keeps the final nasal in `mein`
  (2.7M vs `main` 639) and drops it in `nahi`. Real writing is inconsistent
  word-by-word, so §3.5's nun-ghunna rule is recorded as per-word in the lexicon
  rather than as a blanket letter rule.
- These spellings are what the product's readers actually use. A model trained
  on `nahin` would be correct by our spec and wrong to its audience.

**Consequence.**

- **The eval reference was re-normalized: 405 words across 170 of 269 lines.**
  The human corrections remain valid — a reviewer verified *what was said*, and
  the spelling convention is a separate mechanical layer by design. This is the
  first time that separation has paid for itself.
- **The baseline number moved without the model changing.** CER 26.7% → 27.9%,
  SN-WER 57.4% → 61.5%, because the reference now says `aik`/`nahi`/`woh` while
  the stock romanizer still emits `ek`/`nahin`/`wo`. `normalized-wer` is
  unchanged at 57.4%, since it canonicalizes both sides — which is what makes it
  the spec-invariant metric and the only one comparable across a spec change.
  **27.9% supersedes 26.7% in ADR-010** as the figure to beat, because it is
  measured against the spec Phase 2 will train on.
- 20 tests failed on the flip and were updated. That is the spec guard working:
  the tests encode §8, so changing §8 must turn them red.
- Four rows are `UNSEEN` in the corpus (`Ramzan`, `behen`, `humein`, `tamatar`)
  and 1 is a near-tie (`kaise`/`kese`). These are still hypotheses and block the
  §12 freeze along with the ⚠️ rows the table still carries.
- **ADR-009's homograph question is not resolved by this run.** The corpus is
  Roman Urdu text and code-switches, so a count for `no` cannot separate English
  "no" from Urdu نو. Sense disambiguation needs a different signal than raw
  frequency; the freeze checklist should not treat this ADR as closing it.

---

## ADR-012 — The last contested §8 rows, and one override

**Date:** 2026-08-10 · **Status:** Accepted · Clears §8 for the freeze

**Context.** ADR-011 left seven ⚠️ rows, four `UNSEEN`, and one near-tie.

**Three of the four "unseen" rows were not unseen.** We had guessed spellings the
corpus never uses, so the lookup missed the word entirely rather than the word
being absent:

| Ours | Corpus | |
|---|---|---|
| `humein` 0 | `hamein` 60,603 | adopted |
| `behen` 0 | `behan` 15,750 | adopted |
| `kaise` 8 | `kaisay` 52,463 | adopted |

The `kaise`/`kese` "near-tie" was a tie between 8 hits and 7, while the form
people actually write — `kaisay` — was not in the lexicon at all. A near-tie
between two rare spellings is a signal that the real answer is missing, not that
the choice is finely balanced.

**A merge bug found in passing.** `keh` was a rejected variant of `ke`. Its
59,337 hits belong to the کہہ / `kehna` / `kehta` family — *"to say"* — which is
a different word from کہ, *"that"*. Mapping them together loses the distinction
in every training label. Removed. This is the third homograph of this shape
(after `baray`/`baare` and `they`), which suggests the variant lists should be
audited for it rather than fixed one at a time as they surface.

**The override: تھے stays `thay`.**

The corpus writes it `they` **179,481 times and `thay` zero times**. We are
keeping the spelling nobody uses, because `they` is one of the commonest English
words and §5.1 outranks §8 (ADR-001, ADR-006). Mapping it would respell English
`they` in every subtitle, invisibly.

This is the first case where the corpus and the architecture genuinely conflict,
and it costs something real: the model will be trained on a form its readers do
not write. That is the price of constraint 2, paid deliberately. §8 marks it
**⛔** — resolved *against* the corpus — so it is not later mistaken for an
oversight and "fixed".

**Consequence.**

- §8 is clear. The freeze is **not**, and the blocking item is not the one §12
  expected: `english.txt` holds 418 words against 676 distinct English words in
  35 minutes of audio, and ADR-009's homographs have no principled defaults.
  §12 has been amended to say so, because a checklist that hides its real
  blocker is worse than no checklist.
- Two words remain undecidable by frequency: رمضان (no variant attested at all)
  and ٹماٹر (`timatar` 999 against our unattested `tamatar` — but `tamatar` is
  §3.4's own illustration of the retroflex rule, and 999 hits is thin grounds to
  overturn a rule example). Both need a human decision, both are low-frequency,
  neither blocks anything else.

---

## ADR-013 — english.txt is mined from code-switched speech, not a dictionary

**Date:** 2026-08-10 · **Status:** Accepted · Closes the ADR-009 coverage gap

**Context.** ADR-009 named the binding constraint on the spec freeze:
`english.txt` held 418 words against 676 distinct English words in 35 minutes of
audio, and the normalizer corrupted English it did not recognise.

**The parallel-corpus idea failed, and the failure is worth recording.** The
plan was to settle the homographs from Roman-Urdu-Parl: it aligns each Roman
line with its Urdu original, so English ought to survive in Latin script while
Urdu appears in Perso-Arabic, giving an English-share per token. The corpus does
not work that way. **Its Urdu side transliterates English too** — `tablet
computer` → `ٹیبلٹ کمپیوٹر`, `videos` → `ویڈیوز` — and 200,000 sampled lines
contain zero Latin characters. Every token scored 0.0% English, including
`record` and `sale`, which is how the premise announced itself as broken rather
than as a finding. `scripts/resolve_homographs.py` is kept because the method is
sound on a corpus that preserves code-switching; this one does not.

**Decision. Mine `english.txt` from UrduSpeech transcripts instead.** Those keep
English in Latin script inside Urdu-script sentences (`بہت rare ہوتا ہے …
anesthesia والوں کے`), so a Latin run inside an Urdu sentence *is* English by
construction. 1,853 transcript rows yielded 3,218 distinct English words.

**Reasoning.** A general dictionary is the obvious alternative and the wrong
one: it adds tens of thousands of words that never occur in this domain, and
every one is a new chance to collide with a Roman Urdu spelling. Mining the
vocabulary that actually appears in code-switched Pakistani speech gives
coverage where it is needed and nowhere else. `the` (441×) and `they` (114×)
appear in the mined set, which is independent support for ADR-006's refusal to
map them onto Urdu.

**Result.**

| | Before | After |
|---|---|---|
| `english.txt` | 418 words | 3,343 |
| English in the eval audio unknown to the lexicon | 88% | **1%** |
| Eval lines the normalizer would alter | 49 | **1** |

**Consequence.**

- Only 23 mined words collided with the Roman Urdu lexicon, and the loader's own
  validator caught 4 more I had missed. `to` became the fourth `ambiguous.tsv`
  row (378 English hits against a very common تو, so neither reading can be a
  blanket default). `log`, `per` and `urdu` were dropped from `english.txt` — the
  Urdu reading dominates and the English one barely occurs.
- **A fourth homograph merge found:** `such` was a rejected variant of `sach`, so
  English "such" became Urdu. After `they`, `baray`/`baare` and `keh`/`ke`, this
  is a recurring shape rather than four coincidences, and the variant lists were
  audited for it rather than waiting for the fifth to surface in output.
- Acronyms outrank bulk-mined English: a word whose uppercase form is in
  `acronyms.txt` is excluded from `english.txt`, so `tv` still becomes `TV`.
  `acronyms.txt` is hand-maintained and explicit; `english.txt` is not.
- Six words remain unknown, all explained: `do`/`to` are ambiguous rows, `hi` and
  `pakistan` have Urdu readings, and `jf`/`ment` are fragments of `JF-17` and
  `maintain-ment` in the source transcripts.
- **The rule fallback stays disabled.** 1% coverage is not 0%, and re-enabling
  respelling would corrupt whatever remains outside the lexicon. ADR-009's
  interim decision holds until there is a reason stronger than "coverage looks
  good now".

---

## ADR-014 — Roman-Urdu-Parl is 83% misaligned; romanize by dictionary

**Date:** 2026-08-11 · **Status:** Accepted

**Context.** A human reviewer marked 45 of 60 generated labels bad — a 75% line
error rate — and the failures were not spelling but meaning: `یہ` ("this")
became `ki`, `میرے` ("my") became `ne`, `دل` became `shayar`, `بچانا` became
`daman`. All fluent, all real Urdu words, none catchable by an automatic check.
Risk R5 predicted exactly this and the mitigation was "QA sample every batch".
The QA worked; the pipeline did not.

**Root cause: the parallel corpus is not parallel.** Roman-Urdu-Parl's two files
have identical line counts (6,365,808), which is what makes the defect
survivable — they *look* aligned. Probing with `یہ`, whose correct
romanizations are known independently from SPELLING-SPEC §8, shows only 17 of
100 file regions align: roughly 16–21% and 89–99%. Across the whole file `یہ`
maps to `ki` 634,695 times against `yeh` 144,059, because a shifted window lands
on whatever word is frequent at that offset.

**This was nearly missed.** A spot-check of four sentence pairs showed perfect
alignment and was taken as confirmation. Those four fell in the aligned
minority. Sampling four successes says nothing about a corpus; the probe now
scans every line and reports per-region scores.

**Decision.** Romanize Urdu script by dictionary lookup
(`src/labeling/transliterate.py`), built only from the aligned regions
(`scripts/build_translit_dict.py`, 30,591 entries). The neural path is retired
for label generation.

**Reasoning.**

- **A dictionary cannot hallucinate.** It emits only spellings a human actually
  wrote for that word. The neural model answered every input, including
  `shayar` for `دل`, and a confident wrong answer cannot be filtered downstream
  — a missing one can.
- Measured against the 269-line human-corrected eval set: wrong words **8.0% →
  4.1%**, CER **7.6% → 5.4%**, lines with no wrong word **29% → 57%**.
- It is also roughly a thousand times faster. 300 utterances took seconds
  against minutes; the projected 28-hour label run becomes minutes, which makes
  regenerating labels after a spec change cheap rather than an overnight commit.
- English is untouched by construction: only tokens containing Urdu characters
  are looked up, so §5.1 is enforced by the shape of the code rather than by a
  rule that could be got wrong.

**Consequence.**

- **ADR-011 is unaffected.** Those spelling decisions came from unigram counts
  on the Roman side alone, which involve no alignment. `nahi`, `aik`, `woh`,
  `yeh` all stand. Only the alignment-derived work was contaminated.
- `data/lexicon/transliteration.tsv` is a build artifact but is version
  controlled: it is the difference between reproducible labels and labels that
  depend on which corpus revision someone happened to download.
  `tests/test_transliterate.py` pins the words that regress first if it is ever
  rebuilt from the whole corpus.
- **2.0% of words are still unknown** and are left in Urdu script, reported per
  row. Because utterances average ~66 words, that concentrates into 57% of lines
  carrying at least one — so a per-line "usable" flag is too blunt a filter and
  the unknown *rate* is what training should threshold on.
- **English written in Urdu script remains unsolved and is now the largest
  defect**: `ایکٹرز` → `ayktrz`, `ہیروئین` → `heroen`, `وائس اوور` →
  `wise over`. The dictionary reproduces whatever the corpus did, and the corpus
  transliterated loanwords phonetically. This needs a loanword map from Urdu
  spellings back to English orthography — the next piece of work.

---

## Spelling spec decisions

Mirrors `docs/SPELLING-SPEC.md` §10 — amend in both places.

| Date | Rule | Decision | Reason |
|---|---|---|---|
| 2026-08-06 | §3.4 | Retroflex → plain `t`/`d`/`r` | ITRANS capitals unreadable in subtitles |
| 2026-08-06 | §3.2 | Keep `q` separate from `k` | `qismat` dominant in real usage |
| 2026-08-06 | §4.3 | Word-final long i → `i` not `ee` | Shorter; matches common typing |
| 2026-08-06 | §5.1 | English keeps English spelling | Core architectural requirement (ADR-001) |

---

## Dependency licenses

Every new dependency gets a row. Apache-2.0 / MIT preferred; copyleft needs sign-off.

| Package | License | Commercial OK | Notes |
|---|---|---|---|
| qwen-asr / Qwen3-ASR | Apache-2.0 | ✅ | Weights and code |
| Qwen3-ForcedAligner-0.6B | Apache-2.0 | ✅ | |
| IndicXlit | MIT | ✅ | |
| vLLM | Apache-2.0 | ✅ | |
| FastAPI | MIT | ✅ | |
| pyannote.audio | MIT | ✅ | Models gated on HF — accept terms |
| Demucs | MIT | ✅ | |
| ffmpeg | LGPL/GPL | ⚠️ | Depends on build flags — use LGPL build |
| **UrduSpeech corpus** | **TBD** | **❓** | **Blocking — see PROJECT.md R1** |
| Roman-Urdu-Parl | Research | ❓ | Verify commercial terms |
| Dakshina | CC BY-SA 4.0 | ⚠️ | Share-alike — check derived-work implications |

Added at Phase 0 scaffolding (2026-08-06):

| Package | License | Commercial OK | Notes |
|---|---|---|---|
| torch | BSD-3-Clause | ✅ | |
| torchaudio | BSD-2-Clause | ✅ | `[align]` extra |
| transformers | Apache-2.0 | ✅ | |
| pydantic | MIT | ✅ | Config validation |
| PyYAML | MIT | ✅ | |
| typer | MIT | ✅ | CLI entrypoints |
| uvicorn | BSD-3-Clause | ✅ | |
| python-multipart | Apache-2.0 | ✅ | Upload handling |
| hatchling | MIT | ✅ | Build backend |
| ruff | MIT | ✅ | dev only |
| mypy | MIT | ✅ | dev only |
| pytest / pytest-cov | MIT | ✅ | dev only |
| pre-commit | MIT | ✅ | dev only |
| types-pyyaml | Apache-2.0 | ✅ | dev only |
| uv | MIT / Apache-2.0 (dual) | ✅ | Dependency + interpreter management (ADR-007). A tool, not a linked dependency |
| flash-attn | BSD-3-Clause | ✅ | `cuda` extra, Linux-only; needs `no-build-isolation` |

No copyleft dependency has been added. `jiwer` was **not** added: CER and SN-WER
are short pure functions and an extra dependency for edit distance is not worth
the licence surface (`src/eval/metrics.py`).
