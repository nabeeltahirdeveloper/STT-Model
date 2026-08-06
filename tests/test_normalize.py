"""Contract tests for the normalizer.

Three properties, in order of how expensive they are to get wrong:

1. English words are never respelled. This is the P0 bug (SPELLING-SPEC §5.1) --
   it is the failure that killed the two-stage architecture, and it is invisible
   in aggregate metrics until someone reads the captions.
2. Determinism. Nondeterminism here becomes permanent inconsistency in the
   model weights, because these outputs are training labels (ADR-004).
3. Unknown tokens are recorded, because the unknown rate is the only early
   signal that the lexicon is drifting away from the data.
"""

from __future__ import annotations

import pytest

from src.labeling.lexicon import Lexicon
from src.labeling.normalize import Normalizer, apply_rules, detokenize, tokenize
from src.types import TokenKind

SENTENCE = "umm... main office se aa raha hun, meeting kal hai"


# --------------------------------------------------------------------------- #
# 1. English preservation
# --------------------------------------------------------------------------- #
def test_english_words_pass_through_untouched(normalizer: Normalizer, lexicon: Lexicon) -> None:
    """Every word in english.txt survives normalization byte-for-byte.

    Ambiguous tokens are excluded: they are declared precisely because they do
    not always take the English branch.
    """
    words = sorted(lexicon.english - set(lexicon.ambiguous))
    assert normalizer.normalize(words) == words


def test_english_words_are_tagged_english(normalizer: Normalizer) -> None:
    tokens = normalizer.classify(["meeting", "kal", "hai"])
    assert [token.kind for token in tokens] == [
        TokenKind.ENGLISH,
        TokenKind.URDU,
        TokenKind.URDU,
    ]


@pytest.mark.parametrize(
    ("english", "phonetic_respelling"),
    [
        ("meeting", "mitting"),
        ("school", "iskool"),
        ("problem", "prablam"),
        ("record", "rikaard"),
    ],
)
def test_the_p0_bug_does_not_happen(
    normalizer: Normalizer, english: str, phonetic_respelling: str
) -> None:
    """SPELLING-SPEC §5.1 — a code path that phonetically respells English is a bug."""
    assert normalizer.normalize([english]) == [english]
    assert normalizer.normalize([english]) != [phonetic_respelling]


def test_capitalised_english_keeps_its_case(normalizer: Normalizer) -> None:
    assert normalizer.normalize(["Meeting"]) == ["Meeting"]


# --------------------------------------------------------------------------- #
# 2. Determinism and purity
# --------------------------------------------------------------------------- #
def test_normalize_is_deterministic_over_1000_runs(normalizer: Normalizer) -> None:
    tokens = tokenize(SENTENCE)
    first = normalizer.normalize(tokens)
    for _ in range(1000):
        assert normalizer.normalize(tokens) == first


def test_normalize_text_is_deterministic_over_1000_runs(normalizer: Normalizer) -> None:
    first = normalizer.normalize_text(SENTENCE)
    assert all(normalizer.normalize_text(SENTENCE) == first for _ in range(1000))


def test_two_normalizers_agree(lexicon: Lexicon) -> None:
    """Output depends only on the lexicon, not on instance history."""
    warm = Normalizer(lexicon)
    warm.normalize(tokenize("kuch ajeeb alfaaz yahan"))  # populate the unknown ledger
    cold = Normalizer(lexicon)
    assert warm.normalize(tokenize(SENTENCE)) == cold.normalize(tokenize(SENTENCE))


def test_normalize_does_not_mutate_its_input(normalizer: Normalizer) -> None:
    tokens = tokenize(SENTENCE)
    snapshot = list(tokens)
    normalizer.normalize(tokens)
    assert tokens == snapshot


def test_normalization_is_idempotent(normalizer: Normalizer) -> None:
    """Normalizing already-canonical output must change nothing.

    Without this, spelling consistency depends on how many times text happened
    to pass through the pipeline.
    """
    once = normalizer.normalize(tokenize(SENTENCE))
    assert normalizer.normalize(once) == once


# --------------------------------------------------------------------------- #
# 3. Unknown-token ledger
# --------------------------------------------------------------------------- #
def test_unknown_tokens_are_logged(normalizer: Normalizer) -> None:
    normalizer.normalize(["zzqxwv", "hai"])
    assert "zzqxwv" in normalizer.unknown_tokens
    assert "hai" not in normalizer.unknown_tokens


def test_known_tokens_are_not_logged(normalizer: Normalizer) -> None:
    normalizer.normalize(["meeting", "TV", "hai", "nahi", "500", "."])
    assert normalizer.unknown_tokens == frozenset()


def test_unknown_ledger_does_not_affect_output(normalizer: Normalizer) -> None:
    first = normalizer.normalize(["zzqxwv"])
    assert normalizer.normalize(["zzqxwv"]) == first


def test_reset_unknown_clears_the_ledger(normalizer: Normalizer) -> None:
    normalizer.normalize(["zzqxwv"])
    normalizer.reset_unknown()
    assert normalizer.unknown_tokens == frozenset()


def test_seed_lexicon_covers_the_spec_examples(normalizer: Normalizer) -> None:
    """A sanity floor on lexicon coverage for the words §11 actually uses."""
    normalizer.normalize(tokenize("aaj meeting hai ye bohot acha hai matlab kya"))
    assert normalizer.unknown_tokens == frozenset()


# --------------------------------------------------------------------------- #
# Ambiguity resolution (§5.4)
# --------------------------------------------------------------------------- #
def test_ambiguous_token_defaults_to_urdu(normalizer: Normalizer) -> None:
    """`main ja raha hun` — no English context nearby, so میں wins."""
    assert normalizer.normalize(["main", "ja", "raha", "hun"])[0] == "mein"


def test_ambiguous_token_flips_on_context(normalizer: Normalizer) -> None:
    """`main road par` — `road` within two tokens forces the English reading."""
    assert normalizer.normalize(["main", "road", "par"])[0] == "main"


def test_ambiguity_context_window_is_bounded(normalizer: Normalizer) -> None:
    """`road` five tokens away must not reach back and flip the reading."""
    tokens = ["main", "kal", "shaam", "ko", "wahan", "road", "par"]
    assert normalizer.normalize(tokens)[0] == "mein"


# --------------------------------------------------------------------------- #
# Tokenizer, casing, disfluencies
# --------------------------------------------------------------------------- #
def test_tokenize_splits_punctuation() -> None:
    assert tokenize("hai, kya?") == ["hai", ",", "kya", "?"]


def test_tokenize_keeps_numbers_whole() -> None:
    assert tokenize("500 rupay") == ["500", "rupay"]


def test_detokenize_round_trips() -> None:
    assert detokenize(["hai", ",", "kya", "?"]) == "hai, kya?"


def test_sentence_start_is_capitalised(normalizer: Normalizer) -> None:
    assert (
        normalizer.normalize_text("ye acha hai. wo bhi acha hai") == "Ye acha hai. Wo bhi acha hai"
    )


def test_acronym_is_not_lowercased_at_sentence_start(normalizer: Normalizer) -> None:
    assert normalizer.normalize_text("tv dekh raha tha") == "TV dekh raha tha"


def test_proper_nouns_stay_capitalised_mid_sentence(normalizer: Normalizer) -> None:
    assert normalizer.normalize_text("wo karachi mein hai") == "Wo Karachi mein hai"


@pytest.mark.parametrize("filler", ["acha", "matlab", "yaani", "bas"])
def test_discourse_fillers_are_kept(normalizer: Normalizer, filler: str) -> None:
    """§7 — these carry meaning in speech and belong in the caption."""
    assert filler in normalizer.normalize_text(f"{filler} theek hai").lower()


@pytest.mark.parametrize("hesitation", ["umm", "uh", "hmm", "aaa"])
def test_hesitations_are_dropped(normalizer: Normalizer, hesitation: str) -> None:
    assert normalizer.normalize_text(f"{hesitation} theek hai") == "Theek hai"


def test_numbers_pass_through(normalizer: Normalizer) -> None:
    assert normalizer.normalize(["500", "rupay"]) == ["500", "rupay"]


# --------------------------------------------------------------------------- #
# Rule fallback
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("kabhee", "kabhi"),  # §4.3
        ("zindagee", "zindagi"),  # §4.3
        ("wapass", "wapas"),  # §4.4
        ("baRa", "bara"),  # §3.4
        ("ṭamāṭar", "tamatar"),  # §1.2
        ("subah", "subah"),  # §4.4 exception
    ],
)
def test_apply_rules(token: str, expected: str) -> None:
    assert apply_rules(token) == expected


def test_apply_rules_is_pure() -> None:
    assert apply_rules("kabhee") == apply_rules("kabhee")
