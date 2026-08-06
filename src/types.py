"""Data types shared across the pipeline.

These are frozen dataclasses rather than pydantic models because they sit on the
hot path between stages and carry no user input to validate -- validation happens
once, at the config boundary (`src.config`). Freezing them means a later stage
cannot silently mutate an earlier stage's output, which is the failure mode that
makes timestamp bugs so hard to trace.
"""

from dataclasses import dataclass, field
from enum import StrEnum


class TokenKind(StrEnum):
    """Where a token came from, which decides how the normalizer may touch it.

    The distinction exists because English words must survive untouched
    (SPELLING-SPEC §5) -- losing track of which tokens are English is how
    `meeting` becomes `mitting`.
    """

    URDU = "urdu"
    ENGLISH = "english"
    ACRONYM = "acronym"
    NUMBER = "number"
    PUNCTUATION = "punctuation"


@dataclass(frozen=True, slots=True)
class Segment:
    """A contiguous span of speech from one speaker.

    Produced by diarization, consumed by ASR. `start`/`end` are seconds from the
    start of the source media, not from the start of the chunk -- chunk-relative
    offsets are the classic source of drift on long audio (PROJECT.md §4.4).
    """

    start: float
    end: float
    text: str = ""
    speaker: str | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class Token:
    """A single word of transcript, before timing is attached."""

    text: str
    kind: TokenKind = TokenKind.URDU


@dataclass(frozen=True, slots=True)
class TimedToken:
    """A token with word-level timing from the forced aligner."""

    text: str
    start: float
    end: float
    kind: TokenKind = TokenKind.URDU
    confidence: float | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class Caption:
    """One subtitle cue: what is shown on screen, and when.

    `lines` is already line-broken -- the formatting rules in PROJECT.md §9
    (42 chars/line, max 2 lines, 17 chars/sec) are enforced when the Caption is
    built, not when it is written out, so the writers stay dumb and testable.
    """

    index: int
    start: float
    end: float
    lines: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    @property
    def chars_per_second(self) -> float:
        """Reading speed. Capped at 17 (PROJECT.md §9); above that, captions are unreadable."""
        if self.duration <= 0:
            return float("inf")
        return len(self.text.replace("\n", " ")) / self.duration
