"""Evaluation metrics.

Why not just use WER: Roman Urdu has no standard orthography, so WER scores
`nahin` against `nahi` as a full error even though both are correct. It
systematically understates quality here. CER is primary, SN-WER secondary, and
raw WER may appear in a table only next to CER -- never alone (ADR-005).

Everything here is a pure function over strings. `jiwer` was deliberately not
added (DECISIONS.md > Dependency licenses): edit distance is twenty lines and
the four metrics that matter to this project are not the ones a generic WER
library ships.

**Corpus scores are not the mean of sentence scores.** A one-word utterance with
one error scores 100%, and averaging that against a fifty-word utterance gives
short lines fifty times their weight. Every metric here therefore returns a
`Score` carrying its counts, and `aggregate` sums the counts before dividing.
Call `.rate` on a single pair only for per-line diagnostics.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from src.labeling.normalize import Normalizer, tokenize
from src.types import TokenKind


@dataclass(frozen=True, slots=True)
class ErrorBreakdown:
    """The §6.3 diagnostic: which kind of error, and how much of it.

    The acoustic/orthographic ratio decides where the next month of effort goes.
    Acoustic errors need more data; orthographic errors need normalizer work;
    code-switch errors need better labels; timing errors need aligner work.
    Guessing which one dominates is how teams spend a month on the wrong thing.
    """

    acoustic: float
    orthographic: float
    code_switch: float
    timing: float

    def __post_init__(self) -> None:
        total = self.acoustic + self.orthographic + self.code_switch + self.timing
        if total and abs(total - 1.0) > 1e-6:
            raise ValueError(f"error breakdown must sum to 1.0, got {total}")


@dataclass(frozen=True, slots=True)
class Score:
    """An error count and the reference size it is measured against.

    Carrying both is what makes corpus aggregation correct: `rate` is only
    meaningful for a single pair, `errors`/`total` compose across a corpus.
    """

    errors: int
    total: int

    @property
    def rate(self) -> float:
        """Errors per reference unit. An empty reference scores 0.0, not 1.0.

        An empty reference with a non-empty hypothesis is a pathological input
        rather than a 100%-error one -- there is nothing to be wrong about --
        so it contributes its insertions to `errors` and nothing to `total`,
        and the corpus rate absorbs it.
        """
        return self.errors / self.total if self.total else 0.0


def _levenshtein(reference: list[str], hypothesis: list[str]) -> int:
    """Edit distance with unit costs, two rows of memory."""
    if not reference:
        return len(hypothesis)
    previous = list(range(len(reference) + 1))
    for j, hyp in enumerate(hypothesis, start=1):
        current = [j]
        for i, ref in enumerate(reference, start=1):
            current.append(
                previous[i - 1]
                if ref == hyp
                else 1 + min(previous[i - 1], previous[i], current[i - 1])
            )
        previous = current
    return previous[-1]


# --------------------------------------------------------------------------- #
# Script normalization (SN-WER)
# --------------------------------------------------------------------------- #
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE = re.compile(r"\s+")


def script_normalize(text: str) -> str:
    """Surface normalization only: case, punctuation, diacritics, whitespace.

    This is the "SN" in SN-WER and it is deliberately *not* the canonical
    normalizer. It removes differences that no reader would call an error --
    `Hai,` vs `hai` -- without consulting the lexicon, so SN-WER stays a
    property of the text rather than of how good our spelling map happens to be
    on the day. `normalized_wer` is the one that uses the lexicon, and the two
    are reported separately on purpose: if they diverge, the gap is exactly the
    work the normalizer is doing.
    """
    decomposed = unicodedata.normalize("NFD", text.casefold())
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _SPACE.sub(" ", _PUNCT.sub(" ", stripped)).strip()


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def cer(reference: str, hypothesis: str) -> Score:
    """Character error rate. The primary metric (PROJECT.md §1.4, target < 12%).

    Characters rather than words because a one-character spelling difference
    costs one character here and a whole word under WER. That is the entire
    reason ADR-005 makes this primary.
    """
    ref, hyp = script_normalize(reference), script_normalize(hypothesis)
    return Score(_levenshtein(list(ref), list(hyp)), len(ref))


def wer(reference: str, hypothesis: str) -> Score:
    """Raw word error rate. Never report this alone (ADR-005)."""
    ref, hyp = reference.split(), hypothesis.split()
    return Score(_levenshtein(ref, hyp), len(ref))


def sn_wer(reference: str, hypothesis: str) -> Score:
    """Script-normalized WER (PROJECT.md §1.4, target < 25%)."""
    ref, hyp = script_normalize(reference).split(), script_normalize(hypothesis).split()
    return Score(_levenshtein(ref, hyp), len(ref))


def normalized_wer(reference: str, hypothesis: str, normalizer: Normalizer) -> Score:
    """WER with both sides passed through the canonical normalizer.

    Removes the spelling-variance penalty that makes raw WER misleading. Read
    it against `sn_wer`: the difference between them is the share of apparent
    errors that were only ever spelling variants the lexicon knows about.
    """
    ref = normalizer.normalize_text(reference)
    hyp = normalizer.normalize_text(hypothesis)
    return sn_wer(ref, hyp)


def english_preservation(reference: str, hypothesis: str, normalizer: Normalizer) -> Score:
    """Fraction of English words emitted in English spelling (target > 90%).

    This is the metric that catches the failure the architecture exists to
    avoid: if it drops, something is respelling `meeting` as `mitting`
    (SPELLING-SPEC §5.1, ADR-001, ADR-006).

    Counted over *types present in the reference*, case-insensitively, and
    scored as a preservation rate -- so `Score.errors` here is the number of
    English words that did NOT survive.
    """
    english = [
        token.text.lower()
        for token in normalizer.classify(tokenize(reference))
        if token.kind is TokenKind.ENGLISH
    ]
    if not english:
        return Score(0, 0)
    present = {word.lower() for word in tokenize(hypothesis)}
    lost = sum(1 for word in english if word not in present)
    return Score(lost, len(english))


def spelling_consistency(hypotheses: list[str], normalizer: Normalizer) -> Score:
    """Fraction of word types spelled identically everywhere (target > 98%).

    Two words count as the same type when the normalizer maps them to the same
    canonical form; the type is inconsistent if more than one surface spelling
    of it appears across the corpus. `Score.errors` is the number of
    inconsistent types.

    Only meaningful across a whole corpus -- a single utterance rarely repeats
    a word -- which is why this one takes a list.
    """
    surfaces: dict[str, set[str]] = {}
    for text in hypotheses:
        tokens = tokenize(text)
        for raw, token in zip(tokens, normalizer.classify(tokens), strict=True):
            if token.kind in (TokenKind.PUNCTUATION, TokenKind.NUMBER):
                continue
            surfaces.setdefault(token.text.lower(), set()).add(raw.lower())
    if not surfaces:
        return Score(0, 0)
    inconsistent = sum(1 for spellings in surfaces.values() if len(spellings) > 1)
    return Score(inconsistent, len(surfaces))


# --------------------------------------------------------------------------- #
# Corpus aggregation
# --------------------------------------------------------------------------- #
def aggregate(scores: list[Score]) -> Score:
    """Sum counts, then divide. See the module docstring on why not the mean."""
    return Score(sum(s.errors for s in scores), sum(s.total for s in scores))


# --------------------------------------------------------------------------- #
# §6.3 error breakdown
# --------------------------------------------------------------------------- #
def _align(reference: list[str], hypothesis: list[str]) -> list[tuple[str | None, str | None]]:
    """Word alignment via a Levenshtein backtrace.

    Returns (ref, hyp) pairs; `None` on either side marks an insertion or a
    deletion. Full matrix rather than two rows, because the backtrace needs it.
    """
    rows, cols = len(reference) + 1, len(hypothesis) + 1
    table = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        table[i][0] = i
    for j in range(cols):
        table[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            table[i][j] = min(table[i - 1][j - 1] + cost, table[i - 1][j] + 1, table[i][j - 1] + 1)

    pairs: list[tuple[str | None, str | None]] = []
    i, j = len(reference), len(hypothesis)
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            if table[i][j] == table[i - 1][j - 1] + cost:
                pairs.append((reference[i - 1], hypothesis[j - 1]))
                i, j = i - 1, j - 1
                continue
        if i > 0 and table[i][j] == table[i - 1][j] + 1:
            pairs.append((reference[i - 1], None))
            i -= 1
            continue
        pairs.append((None, hypothesis[j - 1]))
        j -= 1
    return list(reversed(pairs))


# Roman Urdu spelling variation is almost entirely vowel variation, because the
# Perso-Arabic source does not write short vowels and every writer guesses
# differently: `mein`/`men`, `rahi`/`rahee`, `karte`/`karate`, `sath`/`saath`.
# Consonants are the part the script *does* record, so a consonant difference
# means a different word.
#
# Edit distance alone cannot express that. `mein`->`men` and `aya`->`gaya` are
# both one character in four; the first is a respelling and the second is
# "came" against "went". Comparing consonant skeletons separates them, which a
# distance threshold provably cannot.
_VOWELS = "aeiouy"
# Pairs that are one Urdu letter written two ways, not two sounds.
_FOLD = str.maketrans({"v": "w", "q": "k"})


def _skeleton(word: str) -> str:
    """Consonants only, folded and de-doubled: `saath` and `sath` both give `sth`.

    `h` is the awkward one, and it has to be handled by position rather than as
    a letter. After a consonant it is aspiration -- `bh`, `ch`, `kh` are
    distinct sounds in Urdu and `bhai` is not `bai`. After a vowel it is a
    length or breathiness marker that writers add or drop freely: `pata` and
    `patah` are the same word, as are `jaga`/`jagah`, `waja`/`wajah`, `ye`/`yeh`
    and `nai`/`nahi`. Treating `h` as a consonant everywhere made all of those
    look like different words, which mislabelled ordinary spelling variation as
    acoustic error in the §6.3 breakdown.
    """
    folded = word.lower().translate(_FOLD)
    out: list[str] = []
    previous = ""
    for char in folded:
        if not char.isalpha() or char in _VOWELS:
            previous = char
            continue
        if char == "h" and previous in _VOWELS and previous:
            previous = char  # vowel-length marker, not a sound
            continue
        if not out or out[-1] != char:  # `achha`/`acha` -> `ch`
            out.append(char)
        previous = char
    return "".join(out)


def _is_spelling_variant(reference: str, hypothesis: str) -> bool:
    """Whether two spellings are the same word written differently.

    Same consonants, any vowels: a respelling. Different consonants: a
    different word. `hain`/`hai` lands on the acoustic side, which is the right
    call -- a dropped plural marker is a real error, not a spelling habit.
    """
    return _skeleton(reference) == _skeleton(hypothesis)


def classify_errors(
    references: list[str], hypotheses: list[str], normalizer: Normalizer
) -> ErrorBreakdown:
    """Sort every word error into the four §6.3 buckets.

    The rules, in order:

    - **orthographic** -- the normalizer maps both spellings to the same
      canonical form. The model heard the word correctly and spelled it a
      different legal way. Fix in the lexicon, not with training data.
    - **code-switch** -- the reference word is English and the hypothesis word
      is not the same string. This is the P0 failure, so it is separated from
      ordinary substitutions even though it is a strict subset of them.
    - **orthographic, by string similarity** -- the lexicon test above only
      catches variants the lexicon already knows, and the lexicon holds ~150
      rows. Measured on the first real baseline, that made the diagnostic
      useless: it reported 97.7% acoustic, while 36.8% of substitutions were
      pairs like `mein`/`men`, `rahi`/`rahee`, `karte`/`karate` -- plainly the
      same word spelled differently. Reporting those as acoustic would send the
      next month into collecting training data to fix a romanizer problem,
      which is the exact §6.3 failure this function exists to prevent. So a
      substitution whose two spellings are within `_ORTHOGRAPHIC_DISTANCE` of
      each other counts as orthographic even when the lexicon has no opinion.
    - **acoustic** -- everything else: a genuinely different word, plus every
      insertion and deletion.

    Timing is always 0.0 here. It cannot be derived from text and needs the
    aligner experiment (§4.4); `ErrorBreakdown` carries the field so the shape
    of the report does not change once that number exists.
    """
    acoustic = orthographic = code_switch = 0
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        ref_tokens, hyp_tokens = tokenize(reference), tokenize(hypothesis)
        kinds = {
            token_text.lower(): token.kind
            for token_text, token in zip(ref_tokens, normalizer.classify(ref_tokens), strict=True)
        }
        for ref_word, hyp_word in _align(
            script_normalize(" ".join(ref_tokens)).split(),
            script_normalize(" ".join(hyp_tokens)).split(),
        ):
            if ref_word == hyp_word:
                continue
            if ref_word is None or hyp_word is None:
                acoustic += 1
            elif normalizer.normalize([ref_word]) == normalizer.normalize([hyp_word]):
                orthographic += 1
            elif kinds.get(ref_word) is TokenKind.ENGLISH:
                code_switch += 1
            elif _is_spelling_variant(ref_word, hyp_word):
                orthographic += 1
            else:
                acoustic += 1

    total = acoustic + orthographic + code_switch
    if not total:
        return ErrorBreakdown(0.0, 0.0, 0.0, 0.0)
    return ErrorBreakdown(
        acoustic=acoustic / total,
        orthographic=orthographic / total,
        code_switch=code_switch / total,
        timing=0.0,
    )
