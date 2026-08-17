"""The dev set must never overlap training, or its CER is meaningless.

A dev CER measured on clips the model has already trained on reports recall
rather than generalisation, and would read far better than the model deserves --
the same class of self-flattering metric that let three bad runs look healthy.
These tests are the guard on that.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_dev_set import select


def _rows(n: int, categories: tuple[str, ...] = ("A", "B", "C")) -> list[dict[str, object]]:
    return [
        {
            "audio": f"clip{i}.wav",
            "category": categories[i % len(categories)],
            "duration_s": 2.0 + (i % 5),
            "urdu": "ایک دو تین",
            "label": "aik do teen",
            "unknown": [],
        }
        for i in range(n)
    ]


def test_it_returns_the_requested_count() -> None:
    assert len(select(_rows(300), 150, 1.0, 12.0, 0.02)) == 150


def test_every_category_is_represented() -> None:
    """Round-robin, or a dev CER reports whichever genre sorts first."""
    chosen = select(_rows(300), 30, 1.0, 12.0, 0.02)
    assert {str(r["category"]) for r in chosen} == {"A", "B", "C"}


def test_categories_are_balanced() -> None:
    chosen = select(_rows(300), 30, 1.0, 12.0, 0.02)
    counts = [sum(1 for r in chosen if r["category"] == c) for c in "ABC"]
    assert max(counts) - min(counts) <= 1


def test_clips_outside_the_duration_window_are_skipped() -> None:
    rows = _rows(10)
    rows[0]["duration_s"] = 0.5
    rows[1]["duration_s"] = 99.0
    chosen = select(rows, 10, 1.0, 12.0, 0.02)
    assert "clip0.wav" not in {r["audio"] for r in chosen}
    assert "clip1.wav" not in {r["audio"] for r in chosen}


def test_rows_with_an_empty_label_are_skipped() -> None:
    rows = _rows(6)
    rows[0]["label"] = "  "
    assert "clip0.wav" not in {r["audio"] for r in select(rows, 6, 1.0, 12.0, 0.02)}


def test_rows_over_the_unknown_word_budget_are_skipped() -> None:
    rows = _rows(6)
    rows[0]["unknown"] = ["a", "b", "c"]  # 3 unknown of 3 words
    assert "clip0.wav" not in {r["audio"] for r in select(rows, 6, 1.0, 12.0, 0.02)}


def test_no_clip_is_selected_twice() -> None:
    chosen = select(_rows(60), 60, 1.0, 12.0, 0.02)
    audios = [r["audio"] for r in chosen]
    assert len(audios) == len(set(audios))


def test_asking_for_more_than_exists_returns_what_there_is() -> None:
    assert len(select(_rows(10), 500, 1.0, 12.0, 0.02)) == 10


@pytest.mark.skipif(
    not Path("data/labels/dev-set.jsonl").exists(),
    reason="needs a built dev set; generated in the Colab run",
)
def test_the_real_dev_set_does_not_touch_the_eval_set() -> None:
    """Constraint 5. The eval set is scored once, in a report -- never watched."""
    dev = {
        json.loads(line)["audio"]
        for line in Path("data/labels/dev-set.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    evalset = {
        json.loads(line)["audio"]
        for line in Path("data/eval/manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert not (dev & evalset)


@pytest.mark.skipif(
    not (
        Path("data/labels/dev-set.jsonl").exists()
        and Path("data/labels/train-subset.jsonl").exists()
    ),
    reason="needs both manifests built",
)
def test_the_real_dev_set_does_not_overlap_training() -> None:
    """Only meaningful when the subset was built with --exclude."""

    def audios(path: str) -> set[str]:
        return {
            json.loads(line)["audio"]
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    overlap = audios("data/labels/dev-set.jsonl") & audios("data/labels/train-subset.jsonl")
    assert not overlap, (
        f"{len(overlap)} dev clips are in training -- rebuild the subset with "
        "--exclude data/labels/dev-set.jsonl"
    )
