"""One test per row of docs/SPELLING-SPEC.md §11.

These are the acceptance tests for the spelling spec. Rows whose input is Urdu
script need the romanizer, which is Phase 1 work, so they are skipped rather
than deleted -- the test names are the TODO list.

Row 7 is a genuine contradiction inside the spec, not a bug in the code. It is
marked xfail(strict) so that resolving the spec makes the suite go red until the
test is updated too. See the module docstring in `src/labeling/normalize.py` and
the note in docs/DECISIONS.md ADR-006.
"""

from __future__ import annotations

import pytest

from src.labeling.normalize import Normalizer, apply_rules

PHASE_1 = "Phase 1: needs urdu_to_roman (romanizer)"

# ADR-009: the rule fallback no longer respells unknown tokens. These rules are
# still correct orthography; what was wrong was applying them to a token merely
# because the lexicon had no entry for it. Re-enable behind a "known Roman Urdu"
# check once english.txt covers real code-switched vocabulary.
RESPELLING_DISABLED = "ADR-009: respelling of unknown tokens disabled"


@pytest.mark.spec
@pytest.mark.skip(reason=PHASE_1)
def test_row_01_urdu_meeting_hai(normalizer: Normalizer) -> None:
    """آج میٹنگ ہے -> `Aaj meeting hai`. English loanword survives romanization."""
    assert normalizer.normalize_text("آج میٹنگ ہے") == "Aaj meeting hai"


@pytest.mark.spec
@pytest.mark.skip(reason=PHASE_1)
def test_row_02_mujhe_nahin_pata(normalizer: Normalizer) -> None:
    """مجھے نہیں پتہ -> `Mujhe nahi pata`. §3.5 nasalization: `nahin`, not `nahi`."""
    assert normalizer.normalize_text("مجھے نہیں پتہ") == "Mujhe nahi pata"


@pytest.mark.spec
@pytest.mark.skip(reason=PHASE_1)
def test_row_03_ye_bohot_acha_hai(normalizer: Normalizer) -> None:
    """یہ بہت اچھا ہے -> `Ye bohat acha hai`. Three contested §8 rows at once."""
    assert normalizer.normalize_text("یہ بہت اچھا ہے") == "Ye bohat acha hai"


@pytest.mark.spec
def test_row_04_record_kar_lo(normalizer: Normalizer) -> None:
    """`record kar lo` -> `Record kar lo`. English verb keeps English spelling (§5.1)."""
    assert normalizer.normalize_text("record kar lo") == "Record kar lo"


@pytest.mark.spec
@pytest.mark.skip(reason=PHASE_1)
def test_row_05_tamatar_500_rupay(normalizer: Normalizer) -> None:
    """ٹماٹر پانچ سو روپے -> `Tamatar 500 rupay`.

    Also needs number-word -> digit conversion for values above ten (§6), which
    is not implemented: the spec gives the threshold but no Roman Urdu numeral
    grammar (`paanch sau` -> 500).
    """
    assert normalizer.normalize_text("ٹماٹر پانچ سو روپے") == "Tamatar 500 rupay"


@pytest.mark.spec
def test_row_06_problem_ye_hai_ke(normalizer: Normalizer) -> None:
    """`problem yeh hai ke` -> `Problem yeh hai ke`. Code-switch at the sentence head."""
    assert normalizer.normalize_text("problem yeh hai ke") == "Problem yeh hai ke"


@pytest.mark.spec
@pytest.mark.xfail(
    strict=True,
    reason=(
        "SPELLING-SPEC contradicts itself: §8 makes `mein` canonical and `main` a "
        "rejected variant of میں, but §11 row 7 expects `Main`. Resolve against "
        "Roman-Urdu-Parl frequency before freeze (§2), then fix this test."
    ),
)
@pytest.mark.parametrize("text", ["main TV dekh raha tha"])
def test_row_07_main_tv_dekh_raha_tha(normalizer: Normalizer, text: str) -> None:
    """`main TV dekh raha tha` -> `Main TV dekh raha tha`. Acronym stays uppercase (§5.3)."""
    assert normalizer.normalize_text(text) == "Main TV dekh raha tha"


@pytest.mark.spec
def test_row_08_matlab_kya_hai(normalizer: Normalizer) -> None:
    """`umm... matlab kya hai` -> `Matlab kya hai`. Hesitation dropped, filler kept (§7)."""
    assert normalizer.normalize_text("umm... matlab kya hai") == "Matlab kya hai"


@pytest.mark.spec
def test_row_09_school_se_aa_raha_hun(normalizer: Normalizer) -> None:
    """`school se aa raha hun` -> `School se aa raha hun`. §5.2 loanword passthrough."""
    assert normalizer.normalize_text("school se aa raha hoon") == "School se aa raha hoon"


@pytest.mark.spec
def test_row_10_teen_baje_office_jana_hai(normalizer: Normalizer) -> None:
    """`teen bajay office jana hai` -> unchanged but capitalised. §6: 0-10 stay words."""
    expected = "Teen bajay office jana hai"
    assert normalizer.normalize_text("teen bajay office jana hai") == expected


# --------------------------------------------------------------------------- #
# Rule-level coverage for §3-§8, one real example pair per rule.
# --------------------------------------------------------------------------- #


@pytest.mark.spec
@pytest.mark.parametrize(
    ("token", "expected", "rule"),
    [
        ("kismat", "qismat", "§3.2 q stays distinct from k"),
        ("galat", "ghalat", "§3.2 gh"),
        ("bada", "bara", "§3.4 retroflex ڑ -> r"),
        ("chota", "chhota", "§3.3 aspirated چھ -> chh"),
        ("nahi", "nahi", "§3.5 nun ghunna written n"),
        ("kabhee", "kabhi", "§4.3 word-final long i is i, not ee"),
        ("zyadah", "ziyada", "§4.4 word-final -ah -> -a"),
        ("bahut", "bohat", "§8 canonical form"),
        ("achcha", "acha", "§8 canonical form"),
        ("yeh", "yeh", "§8 canonical form"),
        ("woh", "woh", "§8 canonical form"),
        ("hei", "hai", "§8 canonical form"),
    ],
)
def test_variant_maps_to_canonical(
    normalizer: Normalizer, token: str, expected: str, rule: str
) -> None:
    assert normalizer.normalize([token]) == [expected], rule


@pytest.mark.spec
@pytest.mark.parametrize("token", ["subah", "wajah", "tarah", "jagah"])
def test_final_h_is_kept_where_pronounced(normalizer: Normalizer, token: str) -> None:
    """§4.4 exception: the -ah rule does not apply where the h is really pronounced."""
    assert normalizer.normalize([token]) == [token]


@pytest.mark.spec
@pytest.mark.parametrize(
    ("token", "expected"),
    [("tv", "TV"), ("Tv", "TV"), ("cng", "CNG"), ("pti", "PTI"), ("nadra", "NADRA")],
)
def test_acronyms_are_uppercased(normalizer: Normalizer, token: str, expected: str) -> None:
    """§5.3 — uppercase, no periods."""
    assert normalizer.normalize([token]) == [expected]


@pytest.mark.spec
def test_ascii_only_no_diacritics(normalizer: Normalizer) -> None:
    """§1.2 — plain ASCII. `ṭamāṭar` is precise and forbidden."""
    assert normalizer.normalize(["ṭamāṭar"]) == ["tamatar"]


@pytest.mark.spec
def test_itrans_capitals_are_stripped_for_known_words(normalizer: Normalizer) -> None:
    """§3.4 — capitals are not phonetic markers, for anything the lexicon knows.

    The lexicon outranks the rule fallback, so a known word comes back in its
    canonical casing regardless of how it was typed. Sentence casing is applied
    afterwards, by `normalize_text`.
    """
    assert normalizer.normalize(["TamaTar"]) == ["tamatar"]
    assert normalizer.normalize_text("TamaTar acha hai") == "Tamatar acha hai"


@pytest.mark.spec
@pytest.mark.xfail(reason=RESPELLING_DISABLED, strict=True)
def test_itrans_capitals_are_stripped_for_unknown_words() -> None:
    """§3.4 applied to an *unknown* token — disabled by ADR-009.

    Capital-folding cannot tell `baRa` (ITRANS retroflex notation) from `IPL`,
    `PhD` or `YouTube`, and the latter three are far commoner in real
    code-switched audio. It corrupted all of them in the eval set.
    """
    assert apply_rules("baRa") == "bara"
    assert apply_rules("KaRachi") == "Karachi"


@pytest.mark.spec
def test_urdu_punctuation_never_survives(normalizer: Normalizer) -> None:
    """§6 — standard Latin punctuation only, never `۔` or `،`."""
    out = normalizer.normalize_text("yeh acha hai۔ woh bhi۔")
    assert "۔" not in out
    assert "،" not in out
