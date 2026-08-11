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

SENTENCE = "umm... main office se aa raha hoon, meeting kal hai"


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
    normalizer.normalize(tokenize("aaj meeting hai yeh bohat acha hai matlab kya"))
    assert normalizer.unknown_tokens == frozenset()


# --------------------------------------------------------------------------- #
# Ambiguity resolution (§5.4)
# --------------------------------------------------------------------------- #
def test_ambiguous_token_defaults_to_urdu(normalizer: Normalizer) -> None:
    """`main ja raha hun` — no English context nearby, so میں wins."""
    assert normalizer.normalize(["main", "ja", "raha", "hoon"])[0] == "mein"


def test_ambiguous_token_flips_on_context(normalizer: Normalizer) -> None:
    """`main road par` — `road` within two tokens forces the English reading."""
    assert normalizer.normalize(["main", "road", "par"])[0] == "main"


def test_ambiguity_context_window_is_bounded(normalizer: Normalizer) -> None:
    """`road` five tokens away must not reach back and flip the reading."""
    tokens = ["main", "kal", "shaam", "ko", "wahan", "road", "par"]
    assert normalizer.normalize(tokens)[0] == "mein"


def test_main_office_keeps_the_english_reading(normalizer: Normalizer) -> None:
    """Regression: `main office` was being respelled to `mein office`.

    Found in real eval data (ROADSIDE blocks 1 and 16). `main` defaults to the
    Urdu reading and only flips on a listed collocate; `road` was listed but
    `office` was not, so an ordinary English noun phrase came out as Roman
    Urdu. That is the constraint-2 failure this project exists to avoid, and
    the module's own SENTENCE fixture contained the phrase without asserting
    on it.
    """
    assert normalizer.normalize(["main", "office", "ke", "paas"])[0] == "main"


def test_main_still_defaults_to_urdu_without_english_context(
    normalizer: Normalizer,
) -> None:
    """Widening the collocate list must not flip the default reading."""
    assert normalizer.normalize(["main", "aa", "raha", "hoon"])[0] == "mein"


# --------------------------------------------------------------------------- #
# بڑا / بارے family (§3.4 retroflex, §4.3 final long i)
# --------------------------------------------------------------------------- #
def test_bari_collapses_the_doubled_r(normalizer: Normalizer) -> None:
    """بڑی → `bari`. §3.4 maps ڑ to a single `r`; `barri` doubles it."""
    assert normalizer.normalize(["barri", "baat"]) == ["barri", "baat"]


def test_baray_is_canonical_for_the_plural(normalizer: Normalizer) -> None:
    """بڑے → `baray`, and `barray` folds into it."""
    assert normalizer.normalize(["barray", "log"]) == ["baray", "log"]


def test_bare_is_left_alone_because_it_is_english(normalizer: Normalizer) -> None:
    """`bare` must NOT be mapped to `baray`.

    The plain §3.4/§4 output for بڑے is `bare`, which is an English word. The
    same override that makes تھے into `thay` rather than `the` applies: an
    Urdu spelling never gets to collide with an English one, so `bare` is
    absent from the variant list and passes through untouched.
    """
    assert normalizer.normalize(["bare", "minimum"]) == ["bare", "minimum"]


def test_baare_is_distinct_from_baray(normalizer: Normalizer) -> None:
    """بارے → `baare`, a different word from بڑے → `baray`."""
    assert normalizer.normalize(["ke", "baare", "mein"]) == ["ke", "baare", "mein"]


def test_baray_is_not_rewritten_to_baare(normalizer: Normalizer) -> None:
    """`ke baray mein` stays put — deliberately.

    بارے and بڑے are homographs in Roman. A word-level map cannot separate
    them, so mapping `baray` to `baare` would corrupt every "big" to fix every
    "about". These lines are resolved by hand instead.
    """
    assert normalizer.normalize(["ke", "baray", "mein"]) == ["ke", "baray", "mein"]


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
        normalizer.normalize_text("yeh acha hai. woh bhi acha hai")
        == "Yeh acha hai. Woh bhi acha hai"
    )


def test_acronym_is_not_lowercased_at_sentence_start(normalizer: Normalizer) -> None:
    assert normalizer.normalize_text("tv dekh raha tha") == "TV dekh raha tha"


def test_proper_nouns_stay_capitalised_mid_sentence(normalizer: Normalizer) -> None:
    assert normalizer.normalize_text("woh karachi mein hai") == "Woh Karachi mein hai"


def test_acronym_does_not_override_a_known_urdu_word(normalizer: Normalizer) -> None:
    """Regression: `isi` (اسی, "this very") was being upper-cased to `ISI`.

    The acronym branch matched case-insensitively and ran before the spelling
    lexicon, so an ordinary Urdu word became an intelligence agency in all four
    of its occurrences in the eval set — `isi tarah`, `roz isi time par`.
    """
    assert normalizer.normalize(["isi", "tarah"]) == ["isi", "tarah"]


def test_genuine_acronym_still_expands(normalizer: Normalizer) -> None:
    """The fix must not cost us §5.3: `tv` is not a Roman Urdu word."""
    assert normalizer.normalize_text("tv dekh raha tha") == "TV dekh raha tha"


# --------------------------------------------------------------------------- #
# Codes: model numbers, building codes, hyphenated compounds
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("code", ["JF-17", "C2", "S3H", "A-lines", "Maintain-ment"])
def test_codes_survive_tokenization_whole(code: str) -> None:
    """Regression: these were split into pieces and rejoined with spaces."""
    assert tokenize(code) == [code]


@pytest.mark.parametrize("code", ["JF-17", "C2", "S3H", "A-lines"])
def test_codes_are_not_respelled(code: str) -> None:
    """The rule fallback must not touch a code — `JF-17` was becoming `Jf-17`."""
    assert apply_rules(code) == code


def test_code_survives_the_full_pipeline(normalizer: Normalizer) -> None:
    assert normalizer.normalize_text("yeh JF-17 Thunder hai") == "Yeh JF-17 Thunder hai"


def test_ordinary_dash_is_still_punctuation() -> None:
    """Widening the token pattern must not swallow a real dash."""
    assert tokenize("hai - kya") == ["hai", "-", "kya"]


@pytest.mark.parametrize("filler", ["acha", "matlab", "yani", "bas"])
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
# ADR-009: respelling an unknown token is disabled. On the eval set these rules
# fired 28 times and were wrong 28 times, because "unknown" almost always meant
# "English we have not listed" rather than "Roman Urdu variant". The rules are
# still correct orthography and these cases document the target behaviour, so
# they stay as strict xfails rather than being deleted -- they will pass again
# once the fallback is gated on a "known Roman Urdu" check.
_RESPELLING_DISABLED = "ADR-009: respelling of unknown tokens disabled"


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        pytest.param(
            "kabhee",
            "kabhi",
            marks=pytest.mark.xfail(reason=_RESPELLING_DISABLED, strict=True),
        ),  # §4.3
        pytest.param(
            "zindagee",
            "zindagi",
            marks=pytest.mark.xfail(reason=_RESPELLING_DISABLED, strict=True),
        ),  # §4.3
        pytest.param(
            "wapass",
            "wapas",
            marks=pytest.mark.xfail(reason=_RESPELLING_DISABLED, strict=True),
        ),  # §4.4
        pytest.param(
            "baRa",
            "bara",
            marks=pytest.mark.xfail(reason=_RESPELLING_DISABLED, strict=True),
        ),  # §3.4
        ("ṭamāṭar", "tamatar"),  # §1.2 — still applied
        ("subah", "subah"),  # §4.4 exception
    ],
)
def test_apply_rules(token: str, expected: str) -> None:
    assert apply_rules(token) == expected


@pytest.mark.parametrize("word", ["he", "me", "say", "or", "no"])
def test_english_homographs_are_not_claimed_by_the_variant_map(
    normalizer: Normalizer, word: str
) -> None:
    """These were listed as rejected variants of Urdu words and respelled English.

    `he` -> `Hai`, `me` -> `mein`, `say` -> `se`, `or` -> `aur`, `no` -> `nau`.
    The last is the dangerous shape: English "no" became the Urdu numeral nine.
    `canonical.tsv` already documents this exclusion for `the`/`they`; these
    five were left in by oversight, and ADR-006 puts English orthography above
    the variant map.
    """
    assert normalizer.normalize([word]) == [word]


@pytest.mark.parametrize(
    "token", ["three", "fitness", "off", "Shah", "panah", "IPL", "YouTube", "PhD"]
)
def test_unknown_tokens_are_not_respelled(token: str) -> None:
    """ADR-009 — every one of these was corrupted before the fallback was gated.

    `three` -> `thri`, `fitness` -> `fitnes`, `off` -> `of`, `Shah` -> `Sha`.
    The `off` case is the worst kind: the output is a different valid word, so
    nothing downstream can tell that a substitution happened.
    """
    assert apply_rules(token) == token


def test_apply_rules_is_pure() -> None:
    assert apply_rules("kabhee") == apply_rules("kabhee")
