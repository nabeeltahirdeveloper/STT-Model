"""Lexicon loading for the spelling normalizer.

Four files, all tracked in git because they are product artifacts rather than
build output (SPELLING-SPEC.md §12):

    data/lexicon/canonical.tsv   urdu <TAB> canonical <TAB> comma,separated,variants
    data/lexicon/english.txt     one lowercase English word per line
    data/lexicon/acronyms.txt    one uppercase acronym per line
    data/lexicon/ambiguous.tsv   token <TAB> default <TAB> comma,separated,context

Loading is the only I/O in the labeling path. The normalizer itself never reads
a file, so it stays pure and trivially testable (SPELLING-SPEC §9).

The collision check in `Lexicon.load` is the important part of this module.
`main` is both an English word and the dominant misspelling of میں; `sale` is
both English and Urdu. Any token that appears in both the English list and the
Roman Urdu list is a spelling ambiguity that will silently corrupt output if
left undeclared, so it is an error unless it is listed in `ambiguous.tsv`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.config import LexiconConfig

Resolution = Literal["english", "urdu"]

_COMMENT = "#"
_EMPTY_CELL = {"", "-", "--"}


class LexiconError(ValueError):
    """Raised when the lexicon files are internally inconsistent."""


@dataclass(frozen=True, slots=True)
class AmbiguousEntry:
    """A token that is both an English word and a Roman Urdu word.

    `default` is what the token means when nothing nearby disambiguates it.
    `context` lists neighbouring words that force the *other* reading -- e.g.
    `main` defaults to Urdu میں, but `road` or `gate` nearby makes it the English
    adjective. An empty context means the default always wins, which is a valid
    and deliberate answer: it records that the ambiguity was considered and
    resolved one way.
    """

    token: str
    default: Resolution
    context: frozenset[str]

    def resolve(self, neighbours: frozenset[str]) -> Resolution:
        if self.context & neighbours:
            return "english" if self.default == "urdu" else "urdu"
        return self.default


@dataclass(frozen=True, slots=True)
class Lexicon:
    """The complete spelling authority, loaded once and then read-only."""

    canonical: dict[str, str]
    variants: dict[str, str]
    english: frozenset[str]
    acronyms: frozenset[str]
    ambiguous: dict[str, AmbiguousEntry]

    @classmethod
    def load(cls, config: LexiconConfig | None = None, root: Path | None = None) -> Lexicon:
        """Read the lexicon files and validate them against each other.

        `root` prefixes relative paths, so tests can point at a fixture directory
        without mutating the working directory.
        """
        config = config or LexiconConfig()
        base = root or Path.cwd()

        def resolve(path: Path) -> Path:
            return path if path.is_absolute() else base / path

        canonical, variants = _read_canonical(resolve(config.canonical))
        english = _read_word_list(resolve(config.english), lower=True)
        acronyms = _read_word_list(resolve(config.acronyms), lower=False)
        ambiguous = _read_ambiguous(resolve(config.ambiguous))

        lexicon = cls(
            canonical=canonical,
            variants=variants,
            english=english,
            acronyms=frozenset(word.upper() for word in acronyms),
            ambiguous=ambiguous,
        )
        lexicon.validate()
        return lexicon

    def validate(self) -> None:
        """Fail loudly on the inconsistencies that produce silently wrong spellings."""
        overlap = (self.english & set(self.canonical)) | (self.english & set(self.variants))
        undeclared = sorted(overlap - set(self.ambiguous))
        if undeclared:
            raise LexiconError(
                "tokens appear in both english.txt and the Roman Urdu lexicon but are not "
                f"declared in ambiguous.tsv: {undeclared}. Either remove them from english.txt "
                "or add a context rule (SPELLING-SPEC §5.4)."
            )

        # A variant that is also a canonical form means one word has two spellings,
        # which is exactly what this project exists to prevent (SPELLING-SPEC §1.5).
        conflicting = sorted(set(self.variants) & set(self.canonical))
        if conflicting:
            raise LexiconError(
                f"tokens are listed as both canonical and as a rejected variant: {conflicting}"
            )

        for token, entry in self.ambiguous.items():
            if token != token.lower():
                raise LexiconError(f"ambiguous.tsv keys must be lowercase: {token!r}")
            if entry.default == "urdu" and token not in self.canonical | self.variants:
                raise LexiconError(
                    f"{token!r} defaults to the Urdu reading but has no entry in canonical.tsv"
                )

    def canonical_form(self, token: str) -> str | None:
        """Canonical spelling for a token, or None if it is not in the lexicon."""
        key = token.lower()
        return self.canonical.get(key) or self.variants.get(key)

    def __len__(self) -> int:
        return len(self.canonical) + len(self.variants)


def _iter_rows(path: Path) -> list[list[str]]:
    if not path.exists():
        raise LexiconError(f"lexicon file not found: {path}")
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith(_COMMENT):
                continue
            rows.append([cell.strip() for cell in line.split("\t")])
    return rows


def _read_canonical(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Parse canonical.tsv into (canonical form -> surface, variant -> surface)."""
    canonical: dict[str, str] = {}
    variants: dict[str, str] = {}
    rows = _iter_rows(path)
    for line_no, row in enumerate(rows, start=1):
        if row and row[0].lower() == "urdu":  # header
            continue
        if len(row) < 2:
            raise LexiconError(f"{path}:{line_no}: expected at least 2 tab-separated columns")
        surface = row[1]
        if not surface:
            raise LexiconError(f"{path}:{line_no}: empty canonical column")
        key = surface.lower()
        if key in canonical and canonical[key] != surface:
            raise LexiconError(
                f"{path}:{line_no}: {surface!r} already mapped to {canonical[key]!r}"
            )
        canonical[key] = surface

        raw_variants = row[2] if len(row) > 2 else ""
        if raw_variants in _EMPTY_CELL:
            continue
        for variant in raw_variants.split(","):
            token = variant.strip().lower()
            if not token:
                continue
            if token in variants and variants[token] != surface:
                raise LexiconError(
                    f"{path}:{line_no}: variant {token!r} maps to both "
                    f"{variants[token]!r} and {surface!r}"
                )
            variants[token] = surface
    return canonical, variants


def _read_word_list(path: Path, *, lower: bool) -> frozenset[str]:
    if not path.exists():
        raise LexiconError(f"lexicon file not found: {path}")
    words: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            word = raw.split(_COMMENT, 1)[0].strip()
            if not word:
                continue
            words.add(word.lower() if lower else word)
    return frozenset(words)


def _read_ambiguous(path: Path) -> dict[str, AmbiguousEntry]:
    entries: dict[str, AmbiguousEntry] = {}
    for line_no, row in enumerate(_iter_rows(path), start=1):
        if row and row[0].lower() == "token":  # header
            continue
        if len(row) < 2:
            raise LexiconError(f"{path}:{line_no}: expected at least 2 tab-separated columns")
        token, default = row[0].lower(), row[1].lower()
        if default not in ("english", "urdu"):
            raise LexiconError(f"{path}:{line_no}: default must be 'english' or 'urdu'")
        raw_context = row[2] if len(row) > 2 else ""
        context = frozenset(
            word.strip().lower()
            for word in raw_context.split(",")
            if word.strip() and raw_context not in _EMPTY_CELL
        )
        entries[token] = AmbiguousEntry(
            token=token,
            default="english" if default == "english" else "urdu",
            context=context,
        )
    return entries
