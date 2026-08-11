"""The dictionary romanizer (ADR-014).

It replaced a neural model that produced fluent wrong words at a rate that made
75% of generated labels unusable. So the tests are mostly about the two
properties a lookup table has and that model did not: English is untouched, and
an unknown word is *reported* rather than invented.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.labeling.transliterate import (
    Romanized,
    TransliterationError,
    load,
    romanize,
)

TABLE = {
    "یہ": "yeh",
    "بہت": "bohat",
    "اچھا": "acha",
    "ہے": "hai",
    "میرے": "mere",
    "دل": "dil",
}


class TestRomanize:
    def test_maps_urdu_words(self) -> None:
        assert romanize("یہ بہت اچھا ہے", TABLE).text == "yeh bohat acha hai"

    def test_english_is_never_touched(self) -> None:
        """§5.1 — the constraint the architecture exists to protect.

        Free here: a token with no Urdu characters is never looked up, so there
        is no code path that could respell it.
        """
        result = romanize("یہ meeting بہت acha hai", TABLE)
        assert "meeting" in result.text
        assert "acha hai" in result.text

    @pytest.mark.parametrize("token", ["JF-17", "C2", "PhD", "don't", "500"])
    def test_codes_numbers_and_contractions_pass_through(self, token: str) -> None:
        assert romanize(f"یہ {token} ہے", TABLE).text == f"yeh {token} hai"

    def test_punctuation_is_preserved_around_a_word(self) -> None:
        """`ہے۔` must find `ہے`, not miss and be reported unknown."""
        result = romanize("یہ اچھا ہے۔", TABLE)
        assert result.text == "yeh acha hai۔"
        assert result.complete

    def test_unknown_words_are_reported_not_invented(self) -> None:
        """The whole point of the pivot.

        The neural romanizer answered every input, including with `shayar` for
        `دل`. A confident wrong answer cannot be filtered downstream; a missing
        one can.
        """
        result = romanize("یہ زقنقنق ہے", TABLE)
        assert result.unknown == ("زقنقنق",)
        assert not result.complete
        assert "زقنقنق" in result.text  # left visible, not guessed at

    def test_complete_when_everything_resolves(self) -> None:
        assert romanize("میرے دل", TABLE).complete

    def test_empty_input(self) -> None:
        assert romanize("", TABLE) == Romanized("", ())


class TestRegressions:
    """The four substitutions a human reviewer found in real generated labels.

    Each was the neural model answering confidently and wrongly; each is a
    lookup away from correct.
    """

    @pytest.mark.parametrize(
        ("urdu", "expected", "was"),
        [("یہ", "yeh", "ki"), ("میرے", "mere", "ne"), ("دل", "dil", "shayar")],
    )
    def test_known_bad_substitutions(self, urdu: str, expected: str, was: str) -> None:
        assert romanize(urdu, TABLE).text == expected != was


class TestLoad:
    def test_missing_file_says_how_to_build_it(self, tmp_path: Path) -> None:
        with pytest.raises(TransliterationError, match="build_translit_dict"):
            load(tmp_path / "absent.tsv")

    def test_empty_file_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "t.tsv"
        path.write_text("# only a comment\nurdu\troman\n", encoding="utf-8")
        with pytest.raises(TransliterationError, match="no entries"):
            load(path)

    def test_parses_comments_and_header(self, tmp_path: Path) -> None:
        path = tmp_path / "t.tsv"
        path.write_text("# comment\nurdu\troman\nیہ\tyeh\n", encoding="utf-8")
        assert load(path) == {"یہ": "yeh"}


class TestShippedDictionary:
    """Guards on the real artifact, which is a data file people can edit."""

    def test_it_loads(self) -> None:
        assert len(load()) > 20_000

    @pytest.mark.parametrize(
        ("urdu", "expected"),
        [("یہ", "yeh"), ("میرے", "mere"), ("دل", "dil"), ("نہیں", "nahi"), ("میں", "mein")],
    )
    def test_common_words_are_right(self, urdu: str, expected: str) -> None:
        """If these regress, the dictionary was rebuilt from the misaligned corpus."""
        assert load().get(urdu) == expected

    def test_no_entry_maps_to_urdu_script(self) -> None:
        """A Roman side containing Urdu means the source rows were not parallel."""
        bad = [u for u, r in load().items() if any("؀" <= c <= "ۿ" for c in r)]
        assert not bad, f"{len(bad)} entries romanize to Urdu script, e.g. {bad[:5]}"


class TestLoanwords:
    """English borrowed into Urdu script (ADR-014).

    The corpus romanized these phonetically, so the dictionary alone emits
    `ayktrz` for `ایکٹرز`. §5.1 requires English orthography, and the hand-built
    map is what supplies it.
    """

    LOANS = {"ایکٹرز": "actors", "ہیروئین": "heroine", "وائس اوور": "voice over"}

    def test_loanword_beats_the_corpus_spelling(self) -> None:
        table = {"ایکٹرز": "ayktrz", "اچھے": "achay"}
        assert romanize("ایکٹرز اچھے", table, self.LOANS).text == "actors achay"

    def test_multi_word_loanword(self) -> None:
        """`وائس اوور` must be matched as a phrase, before tokenizing."""
        assert romanize("وائس اوور", {}, self.LOANS).text == "voice over"

    def test_english_casing_is_preserved(self) -> None:
        """`PhD` must not be lower-cased on the way through."""
        assert romanize("پی", {}, {"پی": "PhD"}).text == "PhD"

    def test_loanwords_are_optional(self) -> None:
        assert romanize("ایکٹرز", {"ایکٹرز": "ayktrz"}).text == "ayktrz"

    def test_a_loanword_is_not_reported_unknown(self) -> None:
        result = romanize("ایکٹرز", {}, self.LOANS)
        assert result.complete and not result.unknown


def test_shipped_loanwords_load() -> None:
    from src.labeling.transliterate import load_loanwords

    loans = load_loanwords()
    assert len(loans) >= 20
    assert all(
        not any("؀" <= c <= "ۿ" for c in v) for v in loans.values()
    ), "a loanword maps to Urdu script; the English column is wrong"
