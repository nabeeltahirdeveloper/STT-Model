"""Tests for the OpenCut romanizer driver.

These cover the pure parts only -- script detection, run splitting, and the
punctuation/placeholder handling. Calling the model itself needs OpenCut's
environment and a 30-second load, which does not belong in the unit suite.
"""

from __future__ import annotations

from scripts.romanize_via_opencut import (
    _PLACEHOLDER,
    _PUNCT,
    script_of,
    split_runs,
)


class TestScriptOf:
    def test_urdu(self) -> None:
        assert script_of("بندہ") == "urdu"

    def test_devanagari(self) -> None:
        assert script_of("बंदा") == "devanagari"

    def test_latin(self) -> None:
        assert script_of("meeting") == "latin"

    def test_digits_are_not_letters(self) -> None:
        assert script_of("350") == "other"

    def test_majority_wins_on_mixed(self) -> None:
        # An Urdu word with a stray Latin character is still Urdu.
        assert script_of("بندہx") == "urdu"


class TestSplitRuns:
    def test_code_switch_makes_three_runs(self) -> None:
        assert split_runs("کہ image ہے") == [
            ("urdu", "کہ"),
            ("latin", "image"),
            ("urdu", "ہے"),
        ]

    def test_punctuation_attaches_to_preceding_run(self) -> None:
        # "other" tokens must not fragment a run, or the model loses context.
        assert split_runs("بندہ -- ہے") == [("urdu", "بندہ -- ہے")]

    def test_empty(self) -> None:
        assert split_runs("") == []


class TestPlaceholderLeak:
    """Regression: the model emits a literal placeholder for the Urdu full stop.

    The transliteration model was trained on data where "۔" had been replaced
    by the sentinel `thisishypenhere`, and it reproduces that sentinel as a
    word. It corrupted 45 of 269 clips (17%) in the first corpus draft run --
    silently, because the output is otherwise fluent Roman Urdu.
    """

    def test_urdu_full_stop_is_normalized_before_the_model_sees_it(self) -> None:
        assert "بات ہے۔ اچھا".translate(_PUNCT) == "بات ہے. اچھا"

    def test_other_urdu_punctuation_normalized(self) -> None:
        assert "کیا؟ ہاں، بس".translate(_PUNCT) == "کیا? ہاں, بس"

    def test_placeholder_is_stripped_from_output(self) -> None:
        leaked = "Nahin mein banti thi thisishypenhere acha thisishypenhere haan"
        assert _PLACEHOLDER.sub(". ", leaked).strip() == "Nahin mein banti thi. acha. haan"

    def test_placeholder_match_is_case_insensitive(self) -> None:
        assert _PLACEHOLDER.sub(". ", "bhi ThisIsHypenHere wo").strip() == "bhi. wo"

    def test_clean_text_is_untouched(self) -> None:
        clean = "Bohot rare hota hai lekin anesthesia walon ke baare mein"
        assert _PLACEHOLDER.sub(". ", clean).strip() == clean
