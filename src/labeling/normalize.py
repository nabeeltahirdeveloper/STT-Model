"""Canonical Roman Urdu spelling normalization.

This is the one module that is fully implemented in Phase 0, because every other
module depends on its contract. It implements SPELLING-SPEC §9:

    if token in english_lexicon:   emit unchanged        # §5
    elif token in acronym_list:    emit token.upper()    # §5.3
    elif token in canonical:       emit canonical[token] # §8
    elif token in variant_map:     emit variant[token]   # known misspelling
    else:                          emit apply_rules(token) + log_unknown(token)

Three properties matter more than the rules themselves:

*Deterministic* -- same input, same output, always. No randomness, no clock, no
context beyond the token list handed in. The model learns whatever the labels
say, so nondeterminism here becomes permanent inconsistency in the weights.

*Pure* -- no I/O in the transform. Lexicons are loaded once at construction.
The single exception is the unknown-token ledger (`unknown_tokens`), which
records misses for later lexicon growth; it never influences output, so
normalizing the same input twice still yields the same result.

*English-safe* -- English words pass through untouched. A code path that
phonetically re-spells an English word is a P0 bug, not a preference
(SPELLING-SPEC §5.1); it is the failure that killed the two-stage architecture.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from src.config import LexiconConfig
from src.labeling.lexicon import Lexicon
from src.types import Token, TokenKind

# Hesitation sounds are dropped; discourse fillers are kept (SPELLING-SPEC §7).
# `acha`, `matlab`, `yaani`, `bas` carry meaning in speech and belong in captions.
HESITATIONS: frozenset[str] = frozenset(
    {"um", "umm", "ummm", "uh", "uhh", "uhhh", "er", "err", "hmm", "hmmm", "aaa", "aah", "eee"}
)

# §4.4: word-final -ah collapses to -a, except where the h is really pronounced.
FINAL_H_KEPT: frozenset[str] = frozenset(
    {"subah", "wajah", "tarah", "jagah", "salah", "nigah", "gunah", "sharah"}
)

# Doubling a final consonant is a typing habit, not a distinction (§4.4: `wapas`).
_DOUBLE_FINAL = re.compile(r"([bcdfghjklmnpqrstvwxyz])\1$")
_FINAL_EE = re.compile(r"ee$")

_WORD = r"[A-Za-zÀ-ɏ']+"
_NUMBER = r"\d[\d,.]*"
_PUNCT = r"[.,!?;:—-]"
_TOKEN_RE = re.compile(rf"{_NUMBER}|{_WORD}|{_PUNCT}")

_SENTENCE_END = frozenset({".", "!", "?"})
_ATTACHING_PUNCT = frozenset({".", ",", "!", "?", ";", ":"})


def tokenize(text: str) -> list[str]:
    """Split text into word, number and punctuation tokens.

    Deliberately simple and lossy: anything that is not a word, a number or
    sentence punctuation is dropped, because it cannot survive into a caption
    anyway. Urdu (`۔`) and Arabic (`،`) punctuation never reach the output
    because they are not in the token pattern (§6).
    """
    return _TOKEN_RE.findall(text)


def detokenize(tokens: list[str]) -> str:
    """Rejoin tokens, attaching punctuation to the preceding word."""
    out: list[str] = []
    for token in tokens:
        if out and token in _ATTACHING_PUNCT:
            out[-1] += token
        else:
            out.append(token)
    return " ".join(out)


def apply_rules(token: str) -> str:
    """Rule fallback for tokens the lexicon has never seen (SPELLING-SPEC §3-§4).

    Conservative on purpose. These rules only remove notation the spec forbids;
    they never guess at a respelling, because a wrong guess here is
    indistinguishable from an ASR error downstream.
    """
    # §1.2 -- plain ASCII only. Strip diacritics rather than transliterating them.
    decomposed = unicodedata.normalize("NFD", token)
    ascii_only = "".join(ch for ch in decomposed if not unicodedata.combining(ch))

    # §3.4 -- ITRANS-style capitals mark retroflexes (`TamaTar`, `baRa`). The spec
    # collapses retroflexes to their dental counterparts and bans capitals as
    # phonetic markers, so internal capitals go. A *leading* capital is far more
    # likely to be a proper noun (§6) than a retroflex marker, so it survives.
    head, tail = ascii_only[:1], ascii_only[1:]
    word = head + tail.lower()

    if word.lower() in FINAL_H_KEPT:
        return word

    lowered = word.lower()
    if _FINAL_EE.search(lowered) and len(lowered) > 3:
        # §4.3 -- word-final long i is `i`, not `ee`: kabhee -> kabhi.
        return word[:-2] + "i"
    if lowered.endswith("ah") and len(lowered) > 3:
        # §4.4 -- word-final -ah -> -a: zyadah -> zyada.
        return word[:-1]
    if _DOUBLE_FINAL.search(lowered):
        # §4.4 -- a doubled final consonant is a typing habit: wapass -> wapas.
        return word[:-1]

    return word


class Normalizer:
    """Applies the canonical spelling spec to tokenized Roman Urdu text."""

    def __init__(self, lexicon: Lexicon) -> None:
        self._lexicon = lexicon
        self._unknown: set[str] = set()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @property
    def unknown_tokens(self) -> frozenset[str]:
        """Tokens that fell through to the rule fallback.

        Grows across calls. Weekly review of this set is how the lexicon grows
        (SPELLING-SPEC §9); a rising unknown rate is the early warning that the
        spec is drifting away from the data.
        """
        return frozenset(self._unknown)

    def reset_unknown(self) -> None:
        self._unknown.clear()

    def classify(self, tokens: list[str]) -> list[Token]:
        """Normalize and tag each token with where its spelling came from.

        The only cross-token dependency is the ±2 window used to resolve
        ambiguous tokens (§5.4). It is a fixed window rather than the whole
        utterance because `main` is English in `main road` but Urdu in
        `main road par tha` -- proximity is the signal, presence is not.
        """
        return [
            self._normalize_token(token, _window(tokens, index))
            for index, token in enumerate(tokens)
        ]

    def normalize(self, tokens: list[str]) -> list[str]:
        """Map each token to its canonical spelling. Pure with respect to output."""
        return [token.text for token in self.classify(tokens)]

    def normalize_text(self, text: str) -> str:
        """Full text pipeline: tokenize, drop disfluencies, normalize, case, rejoin.

        `normalize` handles spelling; this handles the surrounding rules that
        only make sense over a whole utterance -- §7 disfluencies and §6 casing.
        """
        tokens = _drop_disfluencies(tokenize(text))
        return detokenize(_apply_casing(self.classify(tokens)))

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _normalize_token(self, token: str, window: frozenset[str]) -> Token:
        lexicon = self._lexicon
        key = token.lower()

        if not token:
            return Token(token, TokenKind.PUNCTUATION)

        if token in _ATTACHING_PUNCT or all(not ch.isalnum() for ch in token):
            return Token(token, TokenKind.PUNCTUATION)
        if token[0].isdigit():
            return Token(token, TokenKind.NUMBER)

        # §5.4 -- a token that is both English and Urdu is decided by context
        # before anything else, because the two readings take different branches.
        entry = lexicon.ambiguous.get(key)
        if entry is not None and entry.resolve(window) == "urdu":
            return Token(lexicon.canonical_form(key) or apply_rules(token), TokenKind.URDU)

        # §5.1 -- English passes through untouched. This branch is first for a
        # reason: no later rule may ever get a chance to respell an English word.
        if key in lexicon.english:
            return Token(token, TokenKind.ENGLISH)
        if token.upper() in lexicon.acronyms:
            return Token(token.upper(), TokenKind.ACRONYM)  # §5.3
        if key in lexicon.canonical:
            return Token(lexicon.canonical[key], TokenKind.URDU)  # §8, exact match
        if key in lexicon.variants:
            return Token(lexicon.variants[key], TokenKind.URDU)  # known misspelling

        self._unknown.add(key)
        return Token(apply_rules(token), TokenKind.URDU)


_WINDOW = 2


def _window(tokens: list[str], index: int) -> frozenset[str]:
    """Lowercased tokens within ±2 positions of `index`, excluding it."""
    lo = max(0, index - _WINDOW)
    hi = min(len(tokens), index + _WINDOW + 1)
    return frozenset(tokens[position].lower() for position in range(lo, hi) if position != index)


def _drop_disfluencies(tokens: list[str]) -> list[str]:
    """§7 -- drop hesitation sounds and the punctuation trailing them.

    `umm... matlab kya hai` must become `Matlab kya hai`, so the ellipsis left
    behind by a dropped filler goes with it.
    """
    out: list[str] = []
    dropping = False
    for token in tokens:
        if token.lower() in HESITATIONS:
            dropping = True
            continue
        if dropping and token in _ATTACHING_PUNCT:
            continue
        dropping = False
        out.append(token)
    return out


def _apply_casing(tokens: list[Token]) -> list[str]:
    """§6 -- capital at sentence start, capital for proper nouns, lowercase elsewhere.

    Proper nouns arrive already capitalized from `canonical.tsv` (`Karachi`,
    `Pakistan`), and acronyms are already uppercase, so this only has to handle
    sentence starts and must not flatten what the lexicon decided.
    """
    out: list[str] = []
    at_sentence_start = True
    for token in tokens:
        text = token.text
        if token.kind is TokenKind.PUNCTUATION:
            out.append(text)
            if text in _SENTENCE_END:
                at_sentence_start = True
            continue
        if at_sentence_start and token.kind is not TokenKind.ACRONYM:
            text = text[:1].upper() + text[1:]
        out.append(text)
        at_sentence_start = False
    return out


@lru_cache(maxsize=1)
def _default_normalizer(root: str | None = None) -> Normalizer:
    """Process-wide normalizer built from the tracked lexicon files."""
    return Normalizer(Lexicon.load(LexiconConfig(), Path(root) if root else None))


def normalize(tokens: list[str]) -> list[str]:
    """Module-level convenience wrapper over the default lexicon.

    Prefer constructing a `Normalizer` directly in long-running code so the
    unknown-token ledger is scoped to the job rather than to the process.
    """
    return _default_normalizer().normalize(tokens)
