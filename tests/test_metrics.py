"""Evaluation metrics.

These decide what the project believes about its own quality, so the tests are
mostly about the metrics *disagreeing* in the right way: WER punishing a
spelling variant that CER barely notices, SN-WER forgiving punctuation, the
normalized variant forgiving a known variant that SN-WER still counts. A metric
suite where every number moves together cannot diagnose anything.
"""

from __future__ import annotations

import pytest

from src.eval.metrics import (
    ErrorBreakdown,
    Score,
    aggregate,
    cer,
    classify_errors,
    english_preservation,
    normalized_wer,
    script_normalize,
    sn_wer,
    spelling_consistency,
    wer,
)
from src.labeling.normalize import Normalizer


class TestScore:
    def test_rate(self) -> None:
        assert Score(3, 12).rate == 0.25

    def test_empty_reference_scores_zero_not_one(self) -> None:
        """Nothing to be wrong about. The corpus rate absorbs the insertions."""
        assert Score(4, 0).rate == 0.0


class TestScriptNormalize:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Hai, kya?", "hai kya"),
            ("  Ye   BOHOT  acha  ", "ye bohot acha"),
            ("ṭamāṭar", "tamatar"),
            ("JF-17 Thunder", "jf 17 thunder"),
        ],
    )
    def test_surface_differences_are_removed(self, text: str, expected: str) -> None:
        assert script_normalize(text) == expected

    def test_does_not_consult_the_lexicon(self) -> None:
        """`nahi` is a known variant of `nahin`; SN must NOT collapse it.

        That is `normalized_wer`'s job. Keeping them separate is what makes the
        gap between the two readable as "work the normalizer is doing".
        """
        assert script_normalize("nahi") != script_normalize("nahin")


class TestCer:
    def test_identical_is_zero(self) -> None:
        assert cer("ye bohot acha hai", "ye bohot acha hai").rate == 0.0

    def test_counts_characters_not_words(self) -> None:
        """`nahin` vs `nahi` is one character wrong, not one word wrong."""
        score = cer("wo nahin aya", "wo nahi aya")
        assert score.errors == 1
        assert score.rate < 0.1

    def test_is_symmetric_in_magnitude_but_scaled_by_reference(self) -> None:
        assert cer("abc", "abcdef").errors == 3
        assert cer("abc", "abcdef").total == 3


class TestWerFamily:
    def test_raw_wer_punishes_a_spelling_variant_fully(self) -> None:
        """The whole reason ADR-005 bans reporting this alone."""
        assert wer("wo nahin aya", "wo nahi aya").errors == 1

    def test_sn_wer_forgives_punctuation_and_case(self) -> None:
        assert sn_wer("Ye acha hai.", "ye acha hai").errors == 0

    def test_sn_wer_still_counts_a_spelling_variant(self) -> None:
        assert sn_wer("wo nahin aya", "wo nahi aya").errors == 1

    def test_normalized_wer_forgives_a_known_variant(self, normalizer: Normalizer) -> None:
        """`nahi` -> `nahin` is in the lexicon, so this is not an error."""
        assert normalized_wer("wo nahin aya", "wo nahi aya", normalizer).errors == 0

    def test_normalized_wer_still_counts_a_real_substitution(self, normalizer: Normalizer) -> None:
        assert normalized_wer("wo nahin aya", "wo nahin gaya", normalizer).errors == 1


class TestEnglishPreservation:
    def test_preserved_english_scores_no_errors(self, normalizer: Normalizer) -> None:
        score = english_preservation("kal meeting hai", "kal meeting hai", normalizer)
        assert score.errors == 0
        assert score.total == 1

    def test_respelled_english_is_caught(self, normalizer: Normalizer) -> None:
        """The P0 failure: `meeting` -> `mitting` (ADR-001, ADR-006)."""
        score = english_preservation("kal meeting hai", "kal mitting hai", normalizer)
        assert score.errors == 1

    def test_no_english_in_reference_is_not_a_perfect_score(self, normalizer: Normalizer) -> None:
        """An all-Urdu line must contribute nothing, not a free 100%."""
        assert english_preservation("ye acha hai", "ye acha hai", normalizer) == Score(0, 0)


class TestSpellingConsistency:
    def test_one_spelling_everywhere_is_consistent(self, normalizer: Normalizer) -> None:
        assert spelling_consistency(["wo nahin aya", "nahin"], normalizer).errors == 0

    def test_two_spellings_of_one_word_is_inconsistent(self, normalizer: Normalizer) -> None:
        """`nahin` and `nahi` normalize alike, so they are one type, spelled two ways."""
        score = spelling_consistency(["wo nahin aya", "wo nahi gaya"], normalizer)
        assert score.errors == 1

    def test_empty_corpus(self, normalizer: Normalizer) -> None:
        assert spelling_consistency([], normalizer) == Score(0, 0)


class TestAggregate:
    def test_sums_counts_before_dividing(self) -> None:
        """The bug this prevents: a short line outweighing a long one.

        One error in a 1-word line and one in a 99-word line is 2/100, not the
        mean of 100% and 1%.
        """
        assert aggregate([Score(1, 1), Score(1, 99)]).rate == pytest.approx(0.02)

    def test_empty(self) -> None:
        assert aggregate([]) == Score(0, 0)


class TestErrorBreakdown:
    def test_rejects_a_breakdown_that_does_not_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="must sum to 1.0"):
            ErrorBreakdown(0.5, 0.2, 0.1, 0.0)

    def test_all_zero_is_allowed(self) -> None:
        ErrorBreakdown(0.0, 0.0, 0.0, 0.0)

    def test_orthographic_error_is_not_counted_as_acoustic(self, normalizer: Normalizer) -> None:
        """`nahi` for `nahin` means the model heard right and spelled differently.

        Calling that acoustic would send the next month into collecting more
        training data to fix a lexicon problem — the exact §6.3 failure.
        """
        breakdown = classify_errors(["wo nahin aya"], ["wo nahi aya"], normalizer)
        assert breakdown.orthographic == 1.0
        assert breakdown.acoustic == 0.0

    def test_a_different_word_is_acoustic(self, normalizer: Normalizer) -> None:
        breakdown = classify_errors(["wo nahin aya"], ["wo nahin gaya"], normalizer)
        assert breakdown.acoustic == 1.0

    def test_mangled_english_is_code_switch(self, normalizer: Normalizer) -> None:
        breakdown = classify_errors(["kal meeting hai"], ["kal mitting hai"], normalizer)
        assert breakdown.code_switch == 1.0

    def test_deletion_counts_as_acoustic(self, normalizer: Normalizer) -> None:
        breakdown = classify_errors(["wo nahin aya"], ["wo aya"], normalizer)
        assert breakdown.acoustic == 1.0

    def test_a_perfect_transcript_has_no_breakdown(self, normalizer: Normalizer) -> None:
        breakdown = classify_errors(["ye acha hai"], ["ye acha hai"], normalizer)
        assert breakdown == ErrorBreakdown(0.0, 0.0, 0.0, 0.0)

    def test_timing_is_always_zero_until_the_aligner_lands(self, normalizer: Normalizer) -> None:
        """It cannot be derived from text; the field holds the report's shape."""
        breakdown = classify_errors(["wo nahin aya"], ["wo nahin gaya"], normalizer)
        assert breakdown.timing == 0.0

    def test_mixed_errors_sum_to_one(self, normalizer: Normalizer) -> None:
        breakdown = classify_errors(
            ["kal meeting hai", "wo nahin aya"],
            ["kal mitting hai", "wo nahi gaya"],
            normalizer,
        )
        total = (
            breakdown.acoustic + breakdown.orthographic + breakdown.code_switch + breakdown.timing
        )
        assert total == pytest.approx(1.0)


class TestSpellingVariantHeuristic:
    """ADR-009 aftermath: the lexicon is too small to carry the §6.3 split alone.

    On the first real baseline the lexicon test alone reported 97.7% acoustic
    while a third of substitutions were plainly the same word respelled. These
    pin the heuristic that closes that gap.
    """

    @pytest.mark.parametrize(
        ("reference", "hypothesis"),
        [("mein", "men"), ("rahi", "rahee"), ("karte", "karate"), ("sath", "saath")],
    )
    def test_respellings_count_as_orthographic(
        self, normalizer: Normalizer, reference: str, hypothesis: str
    ) -> None:
        breakdown = classify_errors([reference], [hypothesis], normalizer)
        assert breakdown.orthographic == 1.0

    @pytest.mark.parametrize(
        ("reference", "hypothesis"), [("nahin", "gaya"), ("aya", "khana"), ("baat", "log")]
    )
    def test_different_words_stay_acoustic(
        self, normalizer: Normalizer, reference: str, hypothesis: str
    ) -> None:
        breakdown = classify_errors([reference], [hypothesis], normalizer)
        assert breakdown.acoustic == 1.0

    def test_mangled_english_stays_code_switch_not_orthographic(
        self, normalizer: Normalizer
    ) -> None:
        """`meeting`/`mitting` is a near-identical pair, but the cause differs.

        Ordering matters here: the English test runs before the similarity one,
        so respelled English is still reported as the P0 failure it is rather
        than being absorbed into ordinary spelling drift.
        """
        breakdown = classify_errors(["kal meeting hai"], ["kal mitting hai"], normalizer)
        assert breakdown.code_switch == 1.0
        assert breakdown.orthographic == 0.0


class TestSkeletonHandlesHByPosition:
    """`h` is aspiration after a consonant and a length marker after a vowel.

    Treating it as a consonant everywhere made `pata`/`patah` and `ye`/`yeh`
    look like different words, which pushed ordinary spelling variation into
    the acoustic bucket of the §6.3 breakdown.
    """

    @pytest.mark.parametrize(
        ("reference", "hypothesis"),
        [("pata", "patah"), ("jagah", "jaga"), ("wajah", "waja"), ("yeh", "ye"), ("nahi", "nai")],
    )
    def test_h_after_a_vowel_is_not_a_sound(
        self, normalizer: Normalizer, reference: str, hypothesis: str
    ) -> None:
        breakdown = classify_errors([reference], [hypothesis], normalizer)
        assert breakdown.orthographic == 1.0

    @pytest.mark.parametrize(
        ("reference", "hypothesis"), [("bhai", "bai"), ("chota", "cota"), ("khana", "kana")]
    )
    def test_h_after_a_consonant_is_a_sound(
        self, normalizer: Normalizer, reference: str, hypothesis: str
    ) -> None:
        """Aspiration is phonemic in Urdu — `bhai` is not `bai`."""
        breakdown = classify_errors([reference], [hypothesis], normalizer)
        assert breakdown.acoustic == 1.0
