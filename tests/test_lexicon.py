"""Lexicon integrity.

The lexicon files are hand-edited product artifacts, so the failure mode is a
bad row rather than a bad function. These tests are the guard on the data.
"""

from __future__ import annotations

import pytest

from src.labeling.lexicon import AmbiguousEntry, Lexicon, LexiconError

# docs/SPELLING-SPEC.md §8, illustrative table. Every row must be in the TSV.
SPEC_SECTION_8 = [
    ("hai", ["hei", "he", "hae"]),
    ("hain", ["hein", "hen"]),
    ("tha", ["thaa"]),
    ("thi", ["thee"]),
    ("nahin", ["nahi", "nai", "nhi"]),
    ("kya", ["kia", "keya"]),
    ("ki", ["kee"]),
    ("ke", ["kay", "keh", "k"]),
    ("se", ["say"]),
    ("par", ["per"]),
    ("aur", ["or", "aor"]),
    ("ye", ["yeh", "yh"]),
    ("wo", ["woh", "wh"]),
    ("bohot", ["bahut", "bohat", "buhat"]),
    ("acha", ["achha", "achcha"]),
    ("jana", ["jaana"]),
    ("ana", ["aana"]),
    ("dena", ["daina"]),
    ("lena", ["laina"]),
    ("abhi", ["abhee"]),
    ("kabhi", ["kabhee"]),
    ("zyada", ["ziyada", "zyadah"]),
    ("thora", ["thoda", "thodha"]),
    ("sirf", ["serf"]),
    ("matlab", ["matlub"]),
    ("bilkul", ["bilkool", "bilqul"]),
    ("shukriya", ["shukria"]),
]


def test_lexicon_loads_and_validates(lexicon: Lexicon) -> None:
    assert len(lexicon) > 200


@pytest.mark.parametrize(("canonical", "variants"), SPEC_SECTION_8)
def test_spec_section_8_row_is_present(
    lexicon: Lexicon, canonical: str, variants: list[str]
) -> None:
    assert lexicon.canonical[canonical] == canonical
    for variant in variants:
        assert lexicon.variants[variant] == canonical


def test_proper_nouns_keep_their_capital(lexicon: Lexicon) -> None:
    """§6 — proper nouns are capitalised in the lexicon, not by a casing rule."""
    assert lexicon.canonical["pakistan"] == "Pakistan"
    assert lexicon.canonical["karachi"] == "Karachi"
    assert lexicon.canonical["allah"] == "Allah"


def test_no_undeclared_collision_between_english_and_urdu(lexicon: Lexicon) -> None:
    """The check that keeps `main`-shaped bugs from ever shipping silently."""
    overlap = (lexicon.english & set(lexicon.canonical)) | (lexicon.english & set(lexicon.variants))
    assert overlap <= set(lexicon.ambiguous)


def test_no_token_is_both_canonical_and_a_variant(lexicon: Lexicon) -> None:
    assert not set(lexicon.variants) & set(lexicon.canonical)


def test_the_and_they_are_not_mapped_away_from_english(lexicon: Lexicon) -> None:
    """ADR-006 — SPELLING-SPEC §8 lists both as variants of تھے; English wins."""
    assert "the" not in lexicon.variants
    assert "they" not in lexicon.variants


def test_acronyms_are_uppercase(lexicon: Lexicon) -> None:
    assert all(acronym == acronym.upper() for acronym in lexicon.acronyms)
    assert "TV" in lexicon.acronyms


def test_english_lexicon_is_lowercase(lexicon: Lexicon) -> None:
    assert all(word == word.lower() for word in lexicon.english)


def test_canonical_form_lookup_is_case_insensitive(lexicon: Lexicon) -> None:
    assert lexicon.canonical_form("NAHI") == "nahin"
    assert lexicon.canonical_form("Yeh") == "ye"
    assert lexicon.canonical_form("zzqxwv") is None


def test_undeclared_collision_is_rejected() -> None:
    bad = Lexicon(
        canonical={"main": "main"},
        variants={},
        english=frozenset({"main"}),
        acronyms=frozenset(),
        ambiguous={},
    )
    with pytest.raises(LexiconError, match="ambiguous.tsv"):
        bad.validate()


def test_canonical_variant_conflict_is_rejected() -> None:
    bad = Lexicon(
        canonical={"ke": "ke"},
        variants={"ke": "kay"},
        english=frozenset(),
        acronyms=frozenset(),
        ambiguous={},
    )
    with pytest.raises(LexiconError, match="both canonical and"):
        bad.validate()


def test_missing_file_raises_a_useful_error(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from src.config import LexiconConfig

    with pytest.raises(LexiconError, match="not found"):
        Lexicon.load(LexiconConfig(), tmp_path)


@pytest.mark.parametrize(
    ("default", "context", "neighbours", "expected"),
    [
        ("urdu", {"road"}, {"road", "par"}, "english"),
        ("urdu", {"road"}, {"ja", "raha"}, "urdu"),
        ("english", set(), {"anything"}, "english"),
        ("english", {"ne"}, {"ne", "kaha"}, "urdu"),
    ],
)
def test_ambiguous_entry_resolution(
    default: str, context: set[str], neighbours: set[str], expected: str
) -> None:
    entry = AmbiguousEntry(
        token="x",
        default="urdu" if default == "urdu" else "english",
        context=frozenset(context),
    )
    assert entry.resolve(frozenset(neighbours)) == expected
