"""Resolve PROJECT.md §4.4: does forced alignment work on Roman Urdu?

    python -m scripts.test_aligner --clips data/eval/alignment

THE HIGHEST-PRIORITY EXPERIMENT IN THE PROJECT, and it is roughly 30 minutes of
work. Qwen3-ForcedAligner supports 11 languages; neither Urdu nor Hindi is one
of them. Strategy 1 is to feed it Roman Urdu with `language="English"` on the
theory that the text is already Latin and the aligner matches graphemes to
acoustics -- it reports 34.2 ms on a cross-lingual benchmark. Nobody has checked
whether that holds here.

The answer decides the timestamp design:

    error < ~50 ms   -> strategy 1 (direct). Ship it.
    error < ~150 ms  -> usable, but test strategy 2 (Urdu-script bridge) before
                        committing; subtitle sync target is 200 ms.
    worse            -> strategy 2, then MMS/torchaudio CTC.

Input: a directory of clips, each `<name>.wav` with a `<name>.tsv` of
hand-marked word onsets (`token <TAB> start_seconds`). Twenty clips of ten
seconds is enough to answer the question -- do not build a corpus for this.

The scoring half is implemented and tested. The model call is not: the
`qwen-asr` package is not installed in the Phase 0 environment, and writing an
unverified API call would produce a script that looks runnable and is not.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path

from src.config import AlignerConfig
from src.types import TimedToken

# PROJECT.md §1.4 -- perceptual threshold for subtitle sync.
SYNC_THRESHOLD_MS = 200.0
# PROJECT.md §4.2 -- what the aligner reports on languages it does support.
REPORTED_BASELINE_MS = 32.4


@dataclass(frozen=True, slots=True)
class AlignmentScore:
    """Average absolute onset error, the aligner's standard metric."""

    clip: str
    token_count: int
    mean_abs_error_ms: float
    median_abs_error_ms: float
    worst_ms: float

    @property
    def within_sync_threshold(self) -> bool:
        return self.mean_abs_error_ms < SYNC_THRESHOLD_MS


def read_reference(path: Path) -> list[tuple[str, float]]:
    """Read `token <TAB> start_seconds` reference onsets."""
    onsets: list[tuple[str, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(f"{path}:{line_no}: expected 'token<TAB>start_seconds'")
            onsets.append((parts[0], float(parts[1])))
    return onsets


def score(
    clip: str, predicted: list[TimedToken], reference: list[tuple[str, float]]
) -> AlignmentScore:
    """Compare predicted onsets against hand-marked ones, token by token.

    Requires equal token counts. A length mismatch means the aligner dropped or
    invented a token, which is a different (and worse) failure than bad timing
    and must not be averaged away.
    """
    if len(predicted) != len(reference):
        raise ValueError(
            f"{clip}: aligner returned {len(predicted)} tokens, reference has "
            f"{len(reference)}. Token mismatch is a harder failure than timing error."
        )
    if not predicted:
        raise ValueError(f"{clip}: no tokens to score")

    errors = [
        abs(token.start - onset) * 1000.0
        for token, (_, onset) in zip(predicted, reference, strict=True)
    ]
    return AlignmentScore(
        clip=clip,
        token_count=len(errors),
        mean_abs_error_ms=statistics.fmean(errors),
        median_abs_error_ms=statistics.median(errors),
        worst_ms=max(errors),
    )


def verdict(scores: list[AlignmentScore]) -> str:
    """Turn the numbers into the §4.4 decision."""
    if not scores:
        return "no clips scored"

    overall = statistics.fmean(score.mean_abs_error_ms for score in scores)
    lines = [
        f"{'clip':<28}{'tokens':>8}{'mean':>10}{'median':>10}{'worst':>10}",
        "-" * 66,
    ]
    for entry in scores:
        lines.append(
            f"{entry.clip:<28}{entry.token_count:>8}{entry.mean_abs_error_ms:>9.1f}ms"
            f"{entry.median_abs_error_ms:>9.1f}ms{entry.worst_ms:>9.1f}ms"
        )
    lines += ["", f"overall mean absolute error: {overall:.1f} ms", ""]

    if overall < 50.0:
        lines.append(
            f"VERDICT: strategy 1 (direct) works. {overall:.1f} ms is in the same range as "
            f"the {REPORTED_BASELINE_MS} ms the aligner reports on supported languages. "
            "Set aligner.strategy: direct and move on."
        )
    elif overall < SYNC_THRESHOLD_MS:
        lines.append(
            f"VERDICT: usable but not comfortable. {overall:.1f} ms against a "
            f"{SYNC_THRESHOLD_MS:.0f} ms perceptual budget leaves no headroom for "
            "chunk-offset error. Test strategy 2 (Urdu-script bridge, PROJECT.md §4.4) "
            "before committing."
        )
    else:
        lines.append(
            f"VERDICT: strategy 1 fails at {overall:.1f} ms. Move to strategy 2 (bridge), then "
            "MMS/torchaudio CTC. Record the result in docs/DECISIONS.md either way."
        )
    return "\n".join(lines)


def run_aligner(audio: Path, tokens: list[str], config: AlignerConfig) -> list[TimedToken]:
    """Align `tokens` against `audio` with Qwen3-ForcedAligner.

    Import `qwen_asr` inside this function, not at module scope, so the scoring
    half of this script stays importable and testable without the model stack.

    Raises:
        NotImplementedError: needs `uv sync --all-extras` and a check of the
            installed `qwen-asr` aligner API. Pass
            `language="English"` (config.language) -- do NOT pass "Urdu", which
            the aligner does not support (PROJECT.md §4.3).
    """
    raise NotImplementedError(
        "install the model stack, then call Qwen3-ForcedAligner with "
        f"language={config.language!r} on {audio.name} for {len(tokens)} tokens"
    )


def main(clips: str = "data/eval/alignment", strategy: str = "direct") -> None:
    """Run the §4.4 experiment over every clip in `clips` and print the verdict."""
    root = Path(clips)
    if not root.is_dir():
        raise SystemExit(
            f"clip directory not found: {root}\n"
            "Hand-mark ~20 ten-second clips as <name>.wav + <name>.tsv "
            "(token<TAB>start_seconds). That is the whole input to this experiment."
        )

    config = AlignerConfig(strategy="direct" if strategy == "direct" else "bridge")
    scores: list[AlignmentScore] = []
    for audio in sorted(root.glob("*.wav")):
        reference = read_reference(audio.with_suffix(".tsv"))
        predicted = run_aligner(audio, [token for token, _ in reference], config)
        scores.append(score(audio.stem, predicted, reference))

    print(verdict(scores))


if __name__ == "__main__":  # pragma: no cover
    import typer

    typer.run(main)
