"""Phase 0 gate R1: is the UrduSpeech corpus actually usable?

    python -m scripts.verify_data

This is the first thing to run and the first thing that can kill the plan.
UrduSpeech is 156 hours from a single lab, published May 2026, and the entire
roadmap assumes it (PROJECT.md R1). Two things have to be true: the data is
really downloadable, and its licence permits commercial use. Neither is
confirmed. If this script fails, stop and re-plan -- do not start labelling.

The checks are deliberately shallow and offline: file presence, audio duration
by header, and a licence file being present. It cannot tell you whether the
licence *permits* what you want; it can only tell you whether anyone has read it.
"""

from __future__ import annotations

import contextlib
import wave
from dataclasses import dataclass
from pathlib import Path

# PROJECT.md §5.1. Hours are what the corpus paper claims; this script reports
# what is actually on disk, and the gap between the two is the finding.
EXPECTED_SUBSETS: dict[str, float] = {
    "us-cs": 89.4,  # code-switched conversational -- the primary training set
    "us-std": 59.2,  # standard Urdu acoustics
    "us-engpk": 7.3,  # Pakistani-accented English
    "benchmark": 9.0,  # human-verified eval
}

LICENCE_NAMES = ("LICENSE", "LICENSE.txt", "LICENCE", "LICENSE.md", "COPYING", "TERMS.md")
AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".m4a", ".opus")

# A subset with less than this fraction of its claimed hours is treated as an
# incomplete download rather than a small subset.
COMPLETENESS_THRESHOLD = 0.9


@dataclass(frozen=True, slots=True)
class SubsetReport:
    name: str
    expected_hours: float
    found_hours: float
    file_count: int
    unreadable: int
    present: bool

    @property
    def completeness(self) -> float:
        if not self.expected_hours:
            return 1.0
        return self.found_hours / self.expected_hours

    @property
    def ok(self) -> bool:
        return self.present and self.completeness >= COMPLETENESS_THRESHOLD


def wav_duration(path: Path) -> float | None:
    """Duration in seconds from the WAV header, or None if unreadable.

    Header-only on purpose: this runs over ~72,000 files and must not decode
    audio. Non-WAV formats return None and are counted, not measured -- an
    accurate count with an honest gap beats a slow guess.
    """
    if path.suffix.lower() != ".wav":
        return None
    with (
        contextlib.suppress(wave.Error, OSError, EOFError),
        wave.open(str(path), "rb") as handle,
    ):
        rate = handle.getframerate()
        if rate:
            return handle.getnframes() / float(rate)
    return None


def inspect_subset(root: Path, name: str, expected_hours: float) -> SubsetReport:
    directory = root / name
    if not directory.is_dir():
        return SubsetReport(name, expected_hours, 0.0, 0, 0, present=False)

    seconds = 0.0
    count = 0
    unreadable = 0
    for path in directory.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        count += 1
        duration = wav_duration(path)
        if duration is None:
            unreadable += 1
        else:
            seconds += duration

    return SubsetReport(
        name=name,
        expected_hours=expected_hours,
        found_hours=seconds / 3600.0,
        file_count=count,
        unreadable=unreadable,
        present=True,
    )


def find_licence(root: Path) -> Path | None:
    for name in LICENCE_NAMES:
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def render(reports: list[SubsetReport], licence: Path | None, root: Path) -> tuple[str, bool]:
    """Build the report text and decide whether the gate passes."""
    lines = [f"UrduSpeech verification — {root}", ""]
    lines.append(f"{'subset':<12}{'expected':>10}{'found':>10}{'files':>9}  status")
    lines.append("-" * 56)

    for report in reports:
        if not report.present:
            status = "MISSING"
        elif report.ok:
            status = "ok"
        else:
            status = f"INCOMPLETE ({report.completeness:.0%})"
        lines.append(
            f"{report.name:<12}{report.expected_hours:>9.1f}h{report.found_hours:>9.1f}h"
            f"{report.file_count:>9}  {status}"
        )

    unreadable = sum(report.unreadable for report in reports)
    if unreadable:
        lines.append("")
        lines.append(
            f"note: {unreadable} audio files were not WAV, so their duration is not counted. "
            "Found-hours is a lower bound."
        )

    lines.append("")
    if licence is None:
        lines.append("LICENCE: none found. This is the R1 gate — the corpus cannot be used")
        lines.append("         commercially until someone has read and recorded its terms in")
        lines.append("         docs/DECISIONS.md.")
    else:
        lines.append(f"LICENCE: {licence.name} present. Read it, then record the verdict in")
        lines.append("         docs/DECISIONS.md — presence is not permission.")

    blocking = [report.name for report in reports if not report.ok]
    lines.append("")
    if blocking:
        lines.append(f"GATE R1: FAIL — {', '.join(blocking)}")
        lines.append("Stop and re-plan. The whole roadmap assumes this corpus (PROJECT.md R1).")
    elif licence is None:
        lines.append("GATE R1: FAIL — audio is present but the licence is unverified.")
    else:
        lines.append("GATE R1: data present. Licence still needs a human verdict.")

    return "\n".join(lines), not blocking and licence is not None


def main(corpus: str = "data/raw/urduspeech") -> None:
    """Check the corpus and print a report. Exits non-zero if the gate fails."""
    root = Path(corpus)
    if not root.is_dir():
        raise SystemExit(
            f"corpus directory not found: {root}\n"
            "Download UrduSpeech first — this is PROJECT.md gate R1 and it blocks "
            "every downstream task."
        )

    reports = [inspect_subset(root, name, hours) for name, hours in EXPECTED_SUBSETS.items()]
    text, passed = render(reports, find_licence(root), root)
    print(text)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    import typer

    typer.run(main)
