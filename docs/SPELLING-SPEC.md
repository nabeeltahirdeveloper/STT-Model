# Roman Urdu Spelling Specification

**Version:** 0.1.0 — DRAFT (not yet frozen)
**Status:** Requires corpus validation before freeze
**Freeze deadline:** before Phase 2 training begins

---

## 0. Why this document exists

Roman Urdu has **no standard orthography**. کیا is written `kya`, `kia`, or `keya`.
نہیں is `nahi`, `nahin`, `nai`, or `nhi`. All are "correct."

That means spelling consistency is not a cosmetic concern — it is the single
largest driver of *perceived* caption quality. Two systems with identical word
accuracy will be judged very differently if one is consistent and the other isn't.

This document defines the one spelling this project uses for every word.

> **⚠️ This spec is frozen before Phase 2 training.**
> In an end-to-end model, label spelling is baked into the weights. Changing a rule
> after training means retraining. Argue about rules now; do not touch them later.

---

## 1. Governing principles

Applied in this order when rules conflict:

1. **Follow real usage, not academic transliteration.**
   ALA-LC, ITRANS, and IJMES schemes are precise and nobody writes that way.
   Roman Urdu spelling is set by how Pakistanis actually type. Where this spec
   disagrees with observed frequency, **frequency wins**.

2. **Plain ASCII only.** No diacritics, no macrons, no capital letters used as
   phonetic markers. `tamatar`, never `ṭamāṭar` or `TamaTar`.

3. **Readability over phonetic precision.** A reader should recognise the word at
   a glance. Distinctions a Roman Urdu reader doesn't make are distinctions this
   spec doesn't encode.

4. **English stays English.** `meeting`, not `mitting`. See §5 — this is the rule
   most likely to be violated by a model and the most damaging when it is.

5. **One word, one spelling, forever.** No context-dependent variants.
   Predictability beats elegance.

---

## 2. Corpus validation (required before freeze)

The rules below are a **starting hypothesis**, not derived output. Before v1.0:

```bash
python -m scripts.build_lexicon \
  --corpus data/raw/roman-urdu-parl \
  --top-n 5000 \
  --out docs/lexicon-frequency.tsv
```

Then, for every word in the top 5,000:

- If this spec and corpus frequency **agree** → confirm the rule
- If they **disagree** → corpus wins; amend the rule and note it in §10
- If frequency is **near-tied** (within 15%) → apply the rules in §3–§4 to break the tie

Roman-Urdu-Parl contains 6.37M sentence pairs and a 42,927-word Roman vocabulary,
built specifically to capture spelling variation with realistic distribution. It is
the authority for this document.

---

## 3. Consonants

### 3.1 Merged sets

Roman Urdu collapses Perso-Arabic distinctions that Urdu speakers don't
pronounce differently. Encode the merges.

| Urdu letters | Roman | Example |
|---|---|---|
| ز ذ ض ظ | `z` | زندگی → `zindagi`, ضرور → `zaroor` |
| س ص ث | `s` | سال → `saal`, صبح → `subah` |
| ت ط | `t` | تم → `tum`, طرح → `tarah` |
| ح ہ ھ | `h` | حال → `haal`, ہم → `hum` |
| ا آ ع (as vowel carrier) | vowel | عام → `aam` |

### 3.2 Preserved distinctions

These *are* distinguished by real Roman Urdu writers. Keep them.

| Urdu | Roman | Example |
|---|---|---|
| ق | `q` | قسمت → `qismat`, قبول → `qabool` |
| ک | `k` | کام → `kaam` |
| خ | `kh` | خوش → `khush` |
| غ | `gh` | غلط → `ghalat` |
| ف | `f` | فون → `phone` (English) / فکر → `fikar` |

**Note the `q`/`k` split.** Some schemes merge them. Real usage does not —
`qismat` is overwhelmingly more common than `kismat`. Keep `q`.

### 3.3 Aspirated consonants

Aspiration is written with a following `h`.

| Sound | Roman | Example |
|---|---|---|
| بھ | `bh` | بھائی → `bhai` |
| پھ | `ph` | پھر → `phir` |
| تھ | `th` | تھا → `tha` |
| جھ | `jh` | جھوٹ → `jhoot` |
| چھ | `chh` | چھوٹا → `chhota` |
| دھ | `dh` | دھوپ → `dhoop` |
| کھ | `kh` | کھانا → `khana` |
| گھ | `gh` | گھر → `ghar` |

> **Known ambiguity:** `kh` serves both خ and کھ; `gh` serves both غ and گھ.
> This is accepted. Real writers don't disambiguate, and forcing a distinction
> would violate Principle 1. Context resolves it for readers.

### 3.4 Retroflex consonants

**Collapsed to their dental counterparts. No capitals, no dots.**

| Urdu | Roman | Example |
|---|---|---|
| ٹ | `t` | ٹماٹر → `tamatar` |
| ڈ | `d` | ڈر → `dar` |
| ڑ | `r` | بڑا → `bara` |

Rejected alternative: ITRANS-style capitals (`TamaTar`, `baRa`). Precise, but
nobody types that way and it looks broken in subtitles.

### 3.5 Nasalization (nun ghunna, ں)

| Position | Rule | Example |
|---|---|---|
| Word-final | write `n` | میں → `mein`, نہیں → `nahin` |
| Medial | write `n` | چاند → `chand` |

> **Resolved against corpus (ADR-011).** Not close at all, and not consistent:
> `nahi` beats `nahin` 901,811 : 260, while `mein` beats `main` 2,765,083 : 639.
> The same final nasal is dropped in one word and kept in the other, so this is
> a per-word fact carried by the lexicon, not a letter rule.

---

## 4. Vowels

### 4.1 Core mapping

| Length | Roman | Example |
|---|---|---|
| Short a | `a` | کل → `kal` |
| Short i | `i` | دن → `din` |
| Short u | `u` | تم → `tum` |
| Long aa | `aa` | کام → `kaam` |
| Long ee | `ee` | تیز → `teez` |
| Long oo | `oo` | دور → `door` |
| o | `o` | لوگ → `log` |
| e | `e` | میل → `mel` |

### 4.2 Diphthongs

| Sound | Roman | Example |
|---|---|---|
| ai | `ai` | ہے → `hai` |
| au | `au` | اور → `aur` |

### 4.3 Word-final long i

**Write `i`, not `ee`.**

`kabhi` not `kabhee` · `zindagi` not `zindagee` · `abhi` not `abhee`

Medially, `ee` is retained: `teez`, `cheez`.

### 4.4 Word-final -a vs -ah

**Write `a`.** `zyada` not `zyadah`, `wapas` not `wapass`.
Exception: where `h` is genuinely pronounced — `subah`, `wajah`.

---

## 5. English words — the critical rule

### 5.1 The rule

**Any word of English origin is written in standard English spelling, regardless
of Pakistani pronunciation.**

| Spoken | ✅ Correct | ❌ Wrong |
|---|---|---|
| /miːʈɪŋ/ | `meeting` | `mitting`, `miting` |
| /skuːl/ | `school` | `iskool`, `askool` |
| /prɒbləm/ | `problem` | `prablam`, `problam` |
| /ʈaɪm/ | `time` | `taim` |
| /rɪkɔːɖ/ | `record` | `rikaard` |

This rule exists because the entire two-stage architecture was rejected over it.
Routing English through Perso-Arabic script destroys these words. If the model
emits `mitting`, that is a P0 bug, not a spelling preference.

### 5.2 Naturalized loanwords

Words fully absorbed into Urdu keep English spelling anyway:

`station`, `ticket`, `hospital`, `doctor`, `bus`, `phone`, `computer`, `office`

**Exception:** words whose Urdu form has diverged enough to be a different word.
Judge case by case; record each decision in §10.

### 5.3 Acronyms

Uppercase, no periods: `TV`, `CNG`, `PTI`, `FBR`, `NADRA`, `PIA`

### 5.4 Detection

The normalizer maintains `data/lexicon/english.txt`. Matching tokens pass through
untouched. Ambiguous tokens (`sale` vs Urdu `sale`) are resolved by a context list
in `data/lexicon/ambiguous.tsv`.

---

## 6. Numbers, casing, punctuation

| Item | Rule | Example |
|---|---|---|
| 0–10 | Words | `teen log`, `paanch baje` |
| 11+ | Digits | `25 saal`, `100 rupay` |
| Currency | Digits + `rupay` | `500 rupay` |
| Time | Digits | `3 baje` |
| Years | Digits | `2026` |
| Sentence start | Capital | `Aaj mausam acha hai.` |
| Proper nouns | Capital | `Karachi`, `Ahmed`, `Pakistan` |
| Everything else | Lowercase | — |
| Punctuation | Standard Latin `. , ? !` | Never `۔` or `،` |

---

## 7. Filler words and disfluencies

For captions, transcribe meaningful fillers and drop noise.

| Type | Rule | Example |
|---|---|---|
| Discourse fillers | Keep | `acha`, `matlab`, `yaani`, `bas` |
| Hesitation sounds | Drop | `umm`, `uhh`, `aaa` |
| Repetitions | Drop the stutter | `me-me-meeting` → `meeting` |
| False starts | Drop | `wo jo — main keh raha tha` → `main keh raha tha` |
| Laughter, applause | Drop (v1); bracket tags in v2 | — |

---

## 8. Seed lexicon — high-frequency words

Canonical spelling for the most common tokens.

- **✔** resolved against Roman-Urdu-Parl frequency (ADR-011, ADR-012).
- **⛔** resolved *against* the corpus, on purpose. The corpus spelling is a
  common English word and §5.1 outranks §8. Recorded so the override is not
  mistaken later for an oversight.
- **⚠️** still contested; blocks the freeze (§12).
- Rows with no mark were never contested.

| Urdu | Canonical | Rejected variants | |
|---|---|---|---|
| ہے | `hai` | hei, he, hae | |
| ہیں | `hain` | hein, hen | |
| تھا | `tha` | thaa | |
| تھی | `thi` | thee | |
| تھے | `thay` | the, they | ⛔ |
| نہیں | `nahi` | nahin, nai, nhi | ✔ |
| میں | `mein` | main, mai, me | ✔ |
| کیا | `kya` | kia, keya | |
| کا | `ka` | — | |
| کی | `ki` | kee | |
| کے | `ke` | kay | ✔ |
| کو | `ko` | — | |
| سے | `se` | say | |
| پر | `par` | per | |
| تک | `tak` | — | |
| اور | `aur` | or, aor | |
| یہ | `yeh` | ye, yh | ✔ |
| وہ | `woh` | wo, wh | ✔ |
| کہ | `ke` | k | ✔ |
| بہت | `bohat` | bohot, bahut, buhat | ✔ |
| اچھا | `acha` | achha, achcha | ✔ |
| کرنا | `karna` | — | |
| ہونا | `hona` | — | |
| جانا | `jana` | jaana | |
| آنا | `aana` | ana | ✔ |
| دینا | `dena` | daina | |
| لینا | `lena` | laina | |
| ابھی | `abhi` | abhee | |
| کبھی | `kabhi` | kabhee | |
| زیادہ | `ziyada` | zyada, zyadah | ✔ |
| تھوڑا | `thora` | thoda, thodha | ✔ |
| صرف | `sirf` | serf | |
| ضرور | `zaroor` | zarur, zaroor | |
| مطلب | `matlab` | matlub | |
| بالکل | `bilkul` | bilkool, bilqul | |
| شکریہ | `shukriya` | shukria | ✔ |
| اللہ | `Allah` | — | |
| پاکستان | `Pakistan` | — | |

Full list lives in `data/lexicon/canonical.tsv`. This table is illustrative;
the TSV is authoritative at runtime.

---

## 9. Normalizer algorithm

```
for each token in transcript:
    if token in english_lexicon:
        emit token unchanged                    # §5
    elif token in acronym_list:
        emit token.upper()                      # §5.3
    elif token in canonical_lexicon:
        emit canonical_lexicon[token]           # §8 — exact match
    elif token in variant_map:
        emit variant_map[token]                 # known misspelling → canonical
    else:
        emit apply_rules(token)                 # §3–§4 fallback
        log_unknown(token)                      # for lexicon growth
```

Requirements:

- **Deterministic.** Same input → same output, always. No randomness, no context.
- **Pure.** No I/O inside the transform; lexicons loaded once at init.
- **Logged.** Every unknown token is recorded. Weekly review grows the lexicon.
- **Tested.** Every rule in §3–§8 has a unit test with a real example pair.

---

## 10. Decision log

Amendments go here with reasoning. Append only.

| Date | Rule | Decision | Reason |
|---|---|---|---|
| 2026-08-06 | §3.4 | Retroflex → plain `t`/`d`/`r` | ITRANS capitals unreadable in subtitles |
| 2026-08-06 | §3.2 | Keep `q` separate from `k` | `qismat` dominant in real usage |
| 2026-08-06 | §4.3 | Word-final long i → `i` not `ee` | Shorter, matches common typing |
| 2026-08-06 | §5.1 | English keeps English spelling | Core architectural requirement |
| 2026-08-10 | §8 | 19 canonical rows replaced by corpus-preferred spellings | §2 — frequency beats a hand-written guess. `nahi` outnumbers `nahin` 3,468:1, `aik` beats `ek` 67:1 (ADR-011) |
| 2026-08-10 | §3.5 | Word-final nun ghunna is **per-word**, not a blanket rule | The corpus keeps the `n` in `mein` (2.7M) and drops it in `nahi` (902k). Real writing is inconsistent here; the lexicon carries it word by word |
| 2026-08-10 | §8 | تھے stays `thay` **against** the corpus | Corpus writes it `they` 179,481× and `thay` 0×, but `they` is an English word. §5.1 outranks §8 (ADR-012) |
| 2026-08-10 | §8 | `humein`→`hamein`, `behen`→`behan`, `kaise`→`kaisay` | Our spellings had 0 corpus hits; the attested forms have 60,603 / 15,750 / 52,463 (ADR-012) |
| 2026-08-10 | §8 | `keh` removed as a variant of `ke` | Its 59,337 hits are the کہہ / kehna / kehta family ("to say"), a different word from کہ ("that") (ADR-012) |

---

## 11. Test cases

Every row is an integration test in `tests/test_spelling_spec.py`.

| # | Urdu / spoken | Expected Roman Urdu |
|---|---|---|
| 1 | آج میٹنگ ہے | `Aaj meeting hai` |
| 2 | مجھے نہیں پتہ | `Mujhe nahin pata` |
| 3 | یہ بہت اچھا ہے | `Ye bohot acha hai` |
| 4 | record kar lo | `Record kar lo` |
| 5 | ٹماٹر پانچ سو روپے | `Tamatar 500 rupay` |
| 6 | problem ye hai ke | `Problem ye hai ke` |
| 7 | main TV dekh raha tha | `Main TV dekh raha tha` |
| 8 | umm... matlab kya hai | `Matlab kya hai` |
| 9 | school se aa raha hun | `School se aa raha hun` |
| 10 | teen baje office jana hai | `Teen baje office jana hai` |

---

## 12. Freeze checklist

Do not begin Phase 2 training until every box is ticked.

- [x] Frequency lexicon built from Roman-Urdu-Parl (`docs/lexicon-frequency.tsv`, ADR-011)
- [x] Every ⚠️ row in §8 resolved against frequency data (ADR-011, ADR-012)
- [x] `data/lexicon/canonical.tsv` populated and version-controlled
- [x] `data/lexicon/english.txt` populated — 3,343 words mined from
      code-switched UrduSpeech transcripts. Coverage of English in the eval
      audio: 99% (was 12%) (ADR-013)
- [x] English/Roman-Urdu homographs given principled defaults (ADR-013).
      `he`, `me`, `or`, `say`, `no`, `such` are English; `to` is context-decided
      in ambiguous.tsv. Roman-Urdu-Parl could not settle these — its Urdu side
      transliterates English, so it carries no code-switch signal
- [x] All §11 test cases passing
- [x] Normalizer determinism test passing (same input × 1000 → identical output)
- [ ] 500 random labels hand-reviewed for spec compliance
- [ ] Words the corpus never attests decided by hand: رمضان (0 hits for every
      variant) and ٹماٹر (`timatar` 999, our `tamatar` 0 — but `tamatar` is
      §3.4's own rule example, so 999 is thin grounds to overturn it)
- [ ] Version bumped to 1.0.0 and tagged in git
- [ ] `docs/DECISIONS.md` records the freeze

**After freeze:** changes require a new major version *and* a full retrain.
Treat amendments as expensive, because they are.
