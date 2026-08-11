"""Build a frequency-ranked Roman Urdu vocabulary from Roman-Urdu-Parl.

    uv run python -m scripts.build_lexicon
    uv run python -m scripts.build_lexicon --top-n 5000

This is the gate on freezing the spelling spec. Every canonical spelling in
`data/lexicon/canonical.tsv` is currently a hypothesis; SPELLING-SPEC §1 says
frequency wins where the spec and real usage disagree, and §2 makes this command
the way to find out. Roman-Urdu-Parl was built to capture spelling variation
*with* realistic distribution -- 6.37M sentence pairs, 42,927 Roman types -- so
the dictionary should not be hand-built.

Three outputs, because counting is the easy part and deciding is the point:

1. **The frequency table** -- rank, token, count, share.
2. **A verdict per canonical row.** For each row in `canonical.tsv`, whether the
   corpus agrees with our canonical choice, prefers a variant we rejected, or is
   near-tied. §2's rule: disagreement means corpus wins; within 15% is a
   near-tie to be broken by the §3-§4 letter rules rather than by the count.
3. **The homograph report** (ADR-009). Tokens that are both English words and
   Roman Urdu spellings -- `he`, `me`, `or`, `say`, `no`. These currently have
   no principled default and block the freeze.

**On the homograph counts specifically:** this corpus is Roman Urdu text, and
Roman Urdu code-switches, so `no` appearing 40,000 times does not say whether it
was English "no" or Urdu نو. What the counts *can* settle is the relative claim:
if the unambiguous Urdu spelling of a word massively outnumbers the ambiguous
one, the ambiguous one is probably English when it appears. That is a weaker
inference than a raw count looks like, and the report says so per row rather
than presenting a number that invites over-reading.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src.labeling.lexicon import Lexicon
from src.labeling.normalize import tokenize

CORPUS = Path("data/raw/roman-urdu-parl/original_data/roman-urdu.txt")
OUT = Path("docs/lexicon-frequency.tsv")

# §2 -- counts closer than this are a near-tie and the letter rules decide.
NEAR_TIE = 0.15


def count_tokens(corpus: Path) -> Counter[str]:
    """Tokenize the Roman side of the corpus and count types.

    Uses the same tokenizer as the normalizer (`src.labeling.normalize`). A
    different tokenizer here would describe a different vocabulary than the one
    being normalized, and the comparison would be meaningless.
    """
    counts: Counter[str] = Counter()
    with corpus.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            for token in tokenize(line):
                if token.isalpha():
                    counts[token.lower()] += 1
    return counts


@dataclass(frozen=True, slots=True)
class Verdict:
    """What the corpus says about one canonical.tsv row."""

    canonical: str
    canonical_count: int
    rival: str
    rival_count: int

    @property
    def status(self) -> str:
        if not self.rival_count and not self.canonical_count:
            return "UNSEEN"
        leader = max(self.canonical_count, self.rival_count)
        if leader and abs(self.canonical_count - self.rival_count) / leader <= NEAR_TIE:
            return "NEAR-TIE"
        return "AGREES" if self.canonical_count > self.rival_count else "CORPUS DISAGREES"


def resolve(counts: Counter[str], lexicon: Lexicon) -> list[Verdict]:
    """Compare every canonical spelling against its most common rejected variant."""
    rivals: dict[str, list[str]] = {}
    for variant, canonical in lexicon.variants.items():
        rivals.setdefault(canonical, []).append(variant)

    verdicts: list[Verdict] = []
    for canonical, spellings in sorted(rivals.items()):
        best = max(spellings, key=lambda word: counts[word])
        verdicts.append(Verdict(canonical, counts[canonical.lower()], best, counts[best]))
    return verdicts


def main(
    corpus: str = str(CORPUS),
    top_n: int = 5000,
    out: str = str(OUT),
) -> None:
    """Emit the frequency table, the per-row verdicts, and the homograph report."""
    source = Path(corpus)
    if not source.exists():
        raise SystemExit(f"corpus not found: {source}")

    print(f"counting {source} ...", flush=True)
    counts = count_tokens(source)
    total = sum(counts.values())
    print(f"{total:,} tokens · {len(counts):,} distinct types\n")

    lexicon = Lexicon.load()
    destination = Path(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        handle.write("rank\ttoken\tcount\tshare\tours\n")
        for rank, (token, count) in enumerate(counts.most_common(top_n), start=1):
            ours = lexicon.canonical_form(token) or ""
            mark = "" if ours in ("", token) else ours
            handle.write(f"{rank}\t{token}\t{count}\t{count / total:.6f}\t{mark}\n")
    print(f"-> {destination}  (top {top_n})\n")

    verdicts = resolve(counts, lexicon)
    order = {"CORPUS DISAGREES": 0, "NEAR-TIE": 1, "UNSEEN": 2, "AGREES": 3}
    print(f"{'ours':<12}{'count':>11}   {'rival':<12}{'count':>11}   verdict")
    print("-" * 68)
    for verdict in sorted(verdicts, key=lambda v: (order[v.status], -v.rival_count)):
        print(
            f"{verdict.canonical:<12}{verdict.canonical_count:>11,}   "
            f"{verdict.rival:<12}{verdict.rival_count:>11,}   {verdict.status}"
        )
    print("-" * 68)
    tally = Counter(v.status for v in verdicts)
    print("  " + " · ".join(f"{status}: {n}" for status, n in sorted(tally.items())))

    # ADR-009 -- the tokens that block the freeze.
    print("\nEnglish / Roman Urdu homographs (ADR-009)")
    print("  A raw count cannot separate the two senses; see the module docstring.")
    print(f"\n  {'token':<10}{'count':>12}{'share':>10}   canonical rival")
    for token in sorted(lexicon.english & (set(lexicon.canonical) | set(lexicon.variants))):
        rival = lexicon.canonical_form(token) or "-"
        print(
            f"  {token:<10}{counts[token]:>12,}{counts[token] / total:>10.5f}   "
            f"{rival} ({counts.get(rival, 0):,})"
        )
    print("\nNext: resolve every ⚠️ row in SPELLING-SPEC §8 against this table (§2),")
    print("then freeze the spec. Amendments after Phase 2 cost a retrain (R8).")


if __name__ == "__main__":  # pragma: no cover
    import typer

    typer.run(main)
