"""Build an Urdu→Roman word dictionary from the aligned part of Roman-Urdu-Parl.

    uv run python -m scripts.build_translit_dict

**Most of this corpus is not aligned.** Its two files have identical line counts
— 6,365,808 each — which makes them look like a parallel corpus, but for the
first ~90% of the file row *i* on the Roman side is not the translation of row
*i* on the Urdu side. Building a dictionary from all of it produces confident
nonsense: `یہ` maps to `ki` 634,695 times against `yeh` 144,059, because a
shifted window lands on whatever word is common at that offset.

That is not a subtle failure. It was the cause of the 75% line error rate a
human reviewer found in the machine-generated labels, and it survived a
spot-check of four sentence pairs because those four happened to fall in the
aligned minority. Sampling successes proves nothing about a corpus; the probe
below scans the whole file instead.

The probe: `یہ` is common and unambiguous, so the share of its occurrences that
map to `yeh`/`ye`/`yah` measures alignment directly. Regions above
`ALIGNED_SHARE` are kept and the rest is discarded.

A dictionary cannot hallucinate. It only ever emits a spelling that a human
actually wrote for that word, which is the property the neural romanizer lacked
(`دل` -> `shayar`, `بچانا` -> `daman`).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

ROMAN = Path("data/raw/roman-urdu-parl/original_data/roman-urdu.txt")
URDU = Path("data/raw/roman-urdu-parl/original_data/urdu.txt")
OUT = Path("data/lexicon/transliteration.tsv")

# The alignment probe. Common, unambiguous, and its correct romanizations are
# known independently from SPELLING-SPEC §8.
PROBE = "یہ"
PROBE_ROMAN = {"yeh", "ye", "yah"}
# A region must romanize the probe correctly this often to be trusted.
ALIGNED_SHARE = 0.70
# Below this many sightings a region's estimate is noise, so it is skipped.
MIN_PROBE_HITS = 50
# A mapping seen once may be a typo in a single sentence.
MIN_PAIR_COUNT = 2
BUCKETS = 100


def find_aligned_region(roman: Path, urdu: Path) -> tuple[set[int], list[float]]:
    """Return the set of trusted buckets, and the per-bucket scores."""
    counts = [Counter() for _ in range(BUCKETS)]
    with (
        roman.open(encoding="utf-8", errors="replace") as left,
        urdu.open(encoding="utf-8", errors="replace") as right,
    ):
        for index, (roman_line, urdu_line) in enumerate(zip(left, right, strict=False)):
            roman_words, urdu_words = roman_line.split(), urdu_line.split()
            if len(roman_words) != len(urdu_words):
                continue
            for urdu_word, roman_word in zip(urdu_words, roman_words, strict=True):
                if urdu_word == PROBE:
                    counts[min(index * BUCKETS // 6_400_000, BUCKETS - 1)][roman_word.lower()] += 1

    scores: list[float] = []
    for bucket in counts:
        seen = sum(bucket.values())
        scores.append(
            sum(bucket[r] for r in PROBE_ROMAN) / seen if seen >= MIN_PROBE_HITS else -1.0
        )

    # Every trusted bucket, not just the last run of them. The corpus has two
    # aligned regions -- roughly 16-21% and 89-99% -- and taking only the tail
    # threw away a third of the usable data, which showed up as a dictionary
    # that did not know `دوستو` or `ہنسنا`.
    trusted = {index for index, score in enumerate(scores) if score >= ALIGNED_SHARE}
    if not trusted:
        raise SystemExit("no aligned region found — the corpus may have changed")
    return trusted, scores


def build(roman: Path, urdu: Path, trusted: set[int], lines: int) -> dict[str, str]:
    """Most common Roman spelling per Urdu word, over the aligned regions."""
    pairs: defaultdict[str, Counter[str]] = defaultdict(Counter)
    with (
        roman.open(encoding="utf-8", errors="replace") as left,
        urdu.open(encoding="utf-8", errors="replace") as right,
    ):
        for index, (roman_line, urdu_line) in enumerate(zip(left, right, strict=False)):
            if min(index * BUCKETS // lines, BUCKETS - 1) not in trusted:
                continue
            roman_words, urdu_words = roman_line.split(), urdu_line.split()
            if len(roman_words) != len(urdu_words):
                continue
            for urdu_word, roman_word in zip(urdu_words, roman_words, strict=True):
                pairs[urdu_word][roman_word.lower()] += 1

    return {
        urdu_word: counter.most_common(1)[0][0]
        for urdu_word, counter in pairs.items()
        if sum(counter.values()) >= MIN_PAIR_COUNT
    }


def main(roman: str = str(ROMAN), urdu: str = str(URDU), out: str = str(OUT)) -> None:
    """Detect the aligned region, build the dictionary, write it as TSV."""
    left, right = Path(roman), Path(urdu)
    for path in (left, right):
        if not path.exists():
            raise SystemExit(f"corpus not found: {path}")

    print("probing alignment across the file ...", flush=True)
    trusted, scores = find_aligned_region(left, right)
    runs, current = [], []
    for index in range(BUCKETS):
        if index in trusted:
            current.append(index)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    print(
        f"  {len(trusted)}/{BUCKETS} regions aligned, in {len(runs)} run(s): "
        + ", ".join(f"{r[0]}-{r[-1]}%" for r in runs)
    )
    print(f"  discarding {1 - len(trusted) / BUCKETS:.0%} of the corpus as misaligned")

    table = build(left, right, trusted, 6_400_000)
    destination = Path(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        handle.write("# Urdu word -> Roman spelling, from the aligned region of\n")
        handle.write("# Roman-Urdu-Parl. Built by scripts/build_translit_dict.py (ADR-014).\n")
        handle.write("# Do not rebuild from the whole corpus: 90% of it is misaligned.\n")
        handle.write("urdu\troman\n")
        for urdu_word, roman_word in sorted(table.items()):
            handle.write(f"{urdu_word}\t{roman_word}\n")
    print(f"\n{len(table):,} entries -> {destination}")


if __name__ == "__main__":  # pragma: no cover
    import typer

    typer.run(main)
