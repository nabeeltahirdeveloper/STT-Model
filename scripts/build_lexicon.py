"""Build a frequency-ranked Roman Urdu vocabulary from Roman-Urdu-Parl.

    python -m scripts.build_lexicon \
        --corpus data/raw/roman-urdu-parl \
        --top-n 5000 \
        --out docs/lexicon-frequency.tsv

STUB -- follow-up prompt 2, but this is the gate on freezing the spelling spec.

Every canonical spelling in `data/lexicon/canonical.tsv` is currently a
hypothesis. SPELLING-SPEC §1 says frequency wins where the spec and real usage
disagree, and §2 makes this exact command the way to find out. Roman-Urdu-Parl
was built to capture spelling variation *with* realistic distribution -- 6.37M
sentence pairs, 42,927 Roman types -- so the dictionary does not need to be
hand-built and should not be.

Output columns: rank, token, count, share, and the canonical form this project
currently uses, so disagreements are visible in a diff rather than in prose.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path


def count_tokens(corpus: Path) -> Counter[str]:
    """Tokenize the Roman side of the corpus and count types.

    Must use the same tokenizer as the normalizer (`src.labeling.normalize`),
    or the frequencies describe a different vocabulary than the one being
    normalized.

    Raises:
        NotImplementedError: prompt 2.
    """
    raise NotImplementedError("prompt 2: read Roman-Urdu-Parl, tokenize with normalize.tokenize")


def main(
    corpus: str = "data/raw/roman-urdu-parl",
    top_n: int = 5000,
    out: str = "docs/lexicon-frequency.tsv",
) -> None:
    """Emit the top-N Roman Urdu tokens ranked by frequency.

    Raises:
        NotImplementedError: prompt 2.
    """
    _ = Path(corpus), top_n, Path(out)
    raise NotImplementedError(
        "prompt 2: then compare every ⚠️ row in SPELLING-SPEC §8 against these "
        "counts and recommend a resolution for each (§2 tie-break: within 15% "
        "is a near-tie, decide with §3-§4)"
    )


if __name__ == "__main__":  # pragma: no cover
    import typer

    typer.run(main)
