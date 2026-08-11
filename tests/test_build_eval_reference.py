"""The guards on promoting drafts into the eval set.

`data/eval/` is sacred (CLAUDE.md constraint 5) and this script is the only door
into it, so these tests are about the door being *shut* in the right cases. A
guard that has never been observed to fire is an assumption, not a guard.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_eval_reference import ReferenceError, Utterance, check, load, main


def _utterance(**overrides: object) -> Utterance:
    base = {
        "audio": "a.wav",
        "category": "NEWS",
        "duration_s": 3.0,
        "text": "Ye JF-17 Thunder hai",
        "corpus_transcript": "یہ JF-17 Thunder ہے",
        "corpus_disputed": False,
        "unchanged_from_draft": False,
    }
    base.update(overrides)
    return Utterance(**base)  # type: ignore[arg-type]


class TestGuards:
    def test_accepts_a_reviewed_set(self) -> None:
        check([_utterance(audio=f"{i}.wav") for i in range(10)])

    def test_rejects_an_empty_set(self) -> None:
        with pytest.raises(ReferenceError, match="no utterances"):
            check([])

    def test_rejects_an_empty_line(self) -> None:
        with pytest.raises(ReferenceError, match="empty reference"):
            check([_utterance(), _utterance(audio="b.wav", text="")])

    @pytest.mark.parametrize("text", ["Ye kaisa ہوتا hai", "wo बंदा hai"])
    def test_rejects_unromanized_script(self, text: str) -> None:
        """Constraint 1 — the output script is Latin. A stray Urdu or Devanagari
        word means the line was never romanized, which is exactly what happened
        once in ROADSIDE and was invisible until it was checked for."""
        with pytest.raises(ReferenceError, match="non-Latin"):
            check([_utterance(text=text)])

    def test_rejects_a_file_that_reads_as_unreviewed(self) -> None:
        """Over half the lines identical to the machine draft is not review."""
        rows = [_utterance(audio=f"{i}.wav", unchanged_from_draft=i < 6) for i in range(10)]
        with pytest.raises(ReferenceError, match="unreviewed"):
            check(rows)

    def test_allows_a_realistic_share_of_unchanged_lines(self) -> None:
        """26 of 269 real lines were correct as drafted. That must still pass."""
        rows = [_utterance(audio=f"{i}.wav", unchanged_from_draft=i < 3) for i in range(10)]
        check(rows)

    def test_unreviewed_guard_is_per_category(self) -> None:
        """One bad category must fail the run even if the whole set looks fine."""
        good = [_utterance(audio=f"g{i}.wav", category="NEWS") for i in range(20)]
        bad = [
            _utterance(audio=f"b{i}.wav", category="VLOG", unchanged_from_draft=True)
            for i in range(4)
        ]
        with pytest.raises(ReferenceError, match="VLOG"):
            check(good + bad)


class TestParsing:
    def test_flag_is_stripped_from_the_text_but_recorded(self, tmp_path: Path) -> None:
        """`??` marks a disputed corpus line. It is metadata, not speech."""
        drafts = tmp_path / "drafts"
        drafts.mkdir()
        (drafts / "DRAMA.txt").write_text(
            "# header\n\n[x.wav]\n#   اردو\nKaun se department hai?  ??\n",
            encoding="utf-8",
        )
        manifest = [
            {
                "audio": "x.wav",
                "category": "DRAMA",
                "duration_s": 2.0,
                "corpus_transcript": "اردو",
                "draft": "Kaun se departement hai?",
            }
        ]
        (utterance,) = load(drafts, manifest)
        assert utterance.text == "Kaun se department hai?"
        assert utterance.corpus_disputed is True

    def test_missing_manifest_row_is_an_error(self, tmp_path: Path) -> None:
        drafts = tmp_path / "drafts"
        drafts.mkdir()
        (drafts / "NEWS.txt").write_text("[ghost.wav]\n#   اردو\nText\n", encoding="utf-8")
        with pytest.raises(ReferenceError, match="no manifest row"):
            load(drafts, [])


def test_reference_and_manifest_stay_aligned(tmp_path: Path) -> None:
    """reference.txt line N must be manifest.jsonl row N, or every score is wrong."""
    drafts = tmp_path / "drafts"
    drafts.mkdir()
    # Written deliberately out of order: the script sorts by audio path.
    (drafts / "NEWS.txt").write_text(
        "[z.wav]\n#   ا\nZebra line\n\n[a.wav]\n#   ب\nAlpha line\n", encoding="utf-8"
    )
    rows = [
        {"audio": a, "category": "NEWS", "duration_s": 1.0, "corpus_transcript": "x", "draft": "d"}
        for a in ("z.wav", "a.wav")
    ]
    (drafts / "manifest.json").write_text(json.dumps(rows), encoding="utf-8")

    main(drafts=str(drafts), out_dir=str(tmp_path))

    reference = (tmp_path / "reference.txt").read_text(encoding="utf-8").splitlines()
    manifest = [
        json.loads(line)
        for line in (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert reference == ["Alpha line", "Zebra line"]
    assert [row["text"] for row in manifest] == reference
    assert [row["audio"] for row in manifest] == ["a.wav", "z.wav"]
