"""The scoring CLI.

The metrics themselves are tested in `test_metrics.py`. What is tested here is
the CLI refusing to produce a number it cannot stand behind -- a scorecard that
runs and prints something plausible against misaligned inputs is worse than one
that fails, because the number gets written down.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.score import ScoreError, main, score
from src.labeling.normalize import Normalizer


class TestGuards:
    def test_line_count_mismatch_is_refused(self, tmp_path: Path) -> None:
        """Predictions and references are matched by line number.

        One missing prediction line silently shifts every later utterance
        against the wrong reference, and the resulting score looks ordinary.
        """
        (tmp_path / "ref.txt").write_text("a\nb\nc\n", encoding="utf-8")
        (tmp_path / "pred.txt").write_text("a\nb\n", encoding="utf-8")
        with pytest.raises(ScoreError, match="2 predictions against 3 references"):
            main(pred=str(tmp_path / "pred.txt"), ref=str(tmp_path / "ref.txt"))

    def test_missing_file_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "ref.txt").write_text("a\n", encoding="utf-8")
        with pytest.raises(ScoreError, match="not found"):
            main(pred=str(tmp_path / "nope.txt"), ref=str(tmp_path / "ref.txt"))

    def test_unknown_metric_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "ref.txt").write_text("ye acha hai\n", encoding="utf-8")
        (tmp_path / "pred.txt").write_text("ye acha hai\n", encoding="utf-8")
        with pytest.raises(ScoreError, match="unknown metric"):
            main(
                pred=str(tmp_path / "pred.txt"),
                ref=str(tmp_path / "ref.txt"),
                metrics="cer,made-up",
                manifest=str(tmp_path / "absent.jsonl"),
            )


class TestScores:
    def test_identical_input_scores_perfectly(self, normalizer: Normalizer) -> None:
        lines = ["ye bohot acha hai", "kal meeting hai"]
        result = score(lines, lines, normalizer)
        assert result["cer"].errors == 0
        assert result["sn-wer"].errors == 0
        assert result["english-preservation"].errors == 0

    def test_normalized_wer_is_never_worse_than_sn_wer(self, normalizer: Normalizer) -> None:
        """Normalizing can only merge spellings, never invent a new mismatch.

        If this ever inverts, the normalizer is changing words rather than
        canonicalising them — the ADR-009 failure, caught as a number.
        """
        references = ["wo nahin aya", "ye bohot acha hai", "kal meeting hai"]
        hypotheses = ["wo nahi aya", "ye bahut achha hai", "kal meeting hai"]
        result = score(references, hypotheses, normalizer)
        assert result["normalized-wer"].errors <= result["sn-wer"].errors

    def test_cer_is_gentler_than_wer_on_a_spelling_variant(self, normalizer: Normalizer) -> None:
        """The premise of ADR-005, asserted rather than assumed."""
        result = score(["wo nahin aya"], ["wo nahi aya"], normalizer)
        assert result["cer"].rate < result["wer-raw"].rate
