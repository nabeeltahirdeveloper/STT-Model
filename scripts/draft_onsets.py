"""Draft word onsets for the alignment experiment, for a human to correct.

    uv run python -m scripts.draft_onsets

Marking 200 word onsets by hand is slow enough that it does not get done, and a
half-done job produces worse ground truth than none. So this does what the
subtitle drafts did: a machine proposes, a human corrects, and only the
corrected version counts.

Uses `torchaudio`'s MMS forced aligner -- PROJECT.md §4.4's own fallback 3,
chosen here over a paid API because it runs locally and because its vocabulary
is the bare Latin alphabet, meaning it aligns *romanized* text natively. That is
the exact shape of our data.

**This is a different aligner from the one under test.** Qwen3-ForcedAligner is
what §4.4 strategy 1 proposes to ship; MMS is an independent second opinion. Two
machines agreeing proves nothing on its own, which is why the output here is a
draft for review and not a reference.

Output is Audacity label format (`start<TAB>end<TAB>word`) written next to each
clip, so the timings can be *seen* against the waveform rather than read as
numbers. Fix what is wrong, export, and re-run `prepare_alignment_clips
--from-audacity` to convert.
"""

from __future__ import annotations

import re
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path

ALIGNMENT = Path("data/eval/alignment")

# MMS_FA's vocabulary is a-z plus apostrophe. Digits and punctuation have no
# acoustic token, so they are stripped for alignment while the original word is
# kept for the label -- the reference must stay word-for-word with the .tsv.
_CLEAN = re.compile(r"[^a-z']")


@dataclass(slots=True)
class Clip:
    name: str
    wav: Path
    words: list[str]


def load_clips(directory: Path) -> list[Clip]:
    """Read each `<name>.wav` and the word list from its `<name>.tsv`."""
    clips: list[Clip] = []
    for tsv in sorted(directory.glob("*.tsv")):
        wav = tsv.with_suffix(".wav")
        if not wav.exists():
            continue
        # Punctuation is not a word and has no onset. Left in the reference it
        # blocks the whole clip, since a token with no letters cannot be aligned
        # and dropping it silently would shift every later onset by one.
        words = [
            token
            for token in (
                line.split("\t")[0]
                for line in tsv.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith("#")
            )
            if any(char.isalnum() for char in token)
        ]
        if words:
            clips.append(Clip(tsv.stem, wav, words))
    return clips


def main(directory: str = str(ALIGNMENT), backup: bool = True) -> None:
    """Align every clip and write Audacity labels beside it."""
    warnings.filterwarnings("ignore")
    import torch
    import torchaudio
    from torchaudio.pipelines import MMS_FA as BUNDLE

    root = Path(directory)
    clips = load_clips(root)
    if not clips:
        raise SystemExit(f"no clip/tsv pairs in {root}")

    print("loading MMS forced aligner ...", flush=True)
    model = BUNDLE.get_model()
    tokenizer = BUNDLE.get_tokenizer()
    aligner = BUNDLE.get_aligner()

    ok, skipped = 0, []
    for clip in clips:
        cleaned = [_CLEAN.sub("", word.lower()) for word in clip.words]
        if any(not word for word in cleaned):
            # A word with no alignable letters (a bare number) would shift every
            # later onset by one. Better to skip the clip than to mis-align it.
            skipped.append((clip.name, "contains a word with no letters"))
            continue

        waveform, sample_rate = torchaudio.load(clip.wav)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != BUNDLE.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, BUNDLE.sample_rate)

        with torch.inference_mode():
            emission, _ = model(waveform)
            spans = aligner(emission[0], tokenizer(cleaned))

        ratio = waveform.shape[1] / emission.shape[1] / BUNDLE.sample_rate
        if backup:
            label_file = root / f"{clip.name}.txt"
            if label_file.exists() and not (root / f"{clip.name}.txt.synthetic").exists():
                shutil.copy2(label_file, root / f"{clip.name}.txt.synthetic")

        lines = []
        for word, span in zip(clip.words, spans, strict=True):
            start = span[0].start * ratio
            end = span[-1].end * ratio
            lines.append(f"{start:.3f}\t{end:.3f}\t{word}")
        (root / f"{clip.name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        ok += 1
        print(f"  {clip.name}  {len(clip.words)} words", flush=True)

    print(f"\n{ok} clips aligned, {len(skipped)} skipped")
    for name, why in skipped:
        print(f"  skipped {name}: {why}")
    print(f"\n-> {root}/*.txt  (Audacity label format)")
    print("Open the wav in Audacity, File > Import > Labels, check them against the")
    print("waveform, drag what is wrong, then export and run:")
    print("  uv run python -m scripts.prepare_alignment_clips --from-audacity " + str(root))


if __name__ == "__main__":  # pragma: no cover
    import typer

    typer.run(main)
