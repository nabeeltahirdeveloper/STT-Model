"""Romanize ASR output by driving OpenCut's transliteration service.

    uv run python -m scripts.romanize_via_opencut

Romanization lives in OpenCut and is not reimplemented here. This script only
drives it, then applies this project's canonical spelling on top. The split is
deliberate: OpenCut owns "which letters", `src/labeling/normalize.py` owns
"which spelling", and neither should learn the other's job.

Three things measured on the OpenCut benchmark shape this file.

*AssemblyAI's script choice is unstable.* Asked for `language_code: "ur"` it
returns Urdu script for some clips, Devanagari for others, and mixes them
mid-sentence. PROJECT.md §2.2 predicted it. Both scripts must be handled.

*Devanagari romanizes better than Urdu script.* Devanagari writes short vowels;
Urdu script does not. OpenCut's deterministic maps turn Devanagari into
`banda agar aapka nasha karata hai` while the same maps turn Urdu script into
`bndh agr aap kw nsha krta he`. So Devanagari goes through the maps and Urdu
script goes through the neural model.

*The neural model needs sentence context.* OpenCut's endpoint romanizes one
word at a time, and on isolated words m2m100 hallucinates (`بندہ` -> `shayar`),
which its own `has_consonant_consistency` guard then rejects, falling back to
the rule map. Feeding whole segments avoids both. That is why this script
batches by segment rather than by word.

The output is a DRAFT. It is not the eval set and must never be used as one --
a machine-generated reference measures how well one model imitates another.
A human corrects every line first (data/eval/drafts/README.md).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Literal

Script = Literal["urdu", "devanagari", "latin", "other"]

OPENCUT_API = Path("/Users/mubeen-dev/Documents/quantum/opencut/apps/python-api")

# Runs inside OpenCut's environment. Kept as source text rather than an import
# because the two projects have separate lockfiles and must not share a venv.
_WORKER = """
import json, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from api.services import transliteration_service as ts

payload = json.load(sys.stdin)
out = {}

# Devanagari: OpenCut's deterministic maps. They keep the vowels, so the neural
# model buys nothing here and costs a hallucination risk.
if payload["devanagari"]:
    out["devanagari"] = ts.transliterate_batch(payload["devanagari"])

# Urdu script: the neural model, at segment level. Calling the batch helper
# per-word is what makes it fall back to the rule map.
if payload["urdu"]:
    runtime = ts._load_runtime()
    tok, model = runtime["tokenizer"], runtime["model"]
    import torch
    tok.src_lang = ts.SOURCE_LANG
    # Chunked so hundreds of segments fit in memory. The model is loaded once
    # for the whole run -- loading costs ~30s, so one process, many batches.
    size = 24
    decoded = []
    for i in range(0, len(payload["urdu"]), size):
        chunk = payload["urdu"][i:i + size]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                forced_bos_token_id=tok.get_lang_id(ts.TARGET_LANG),
                num_beams=ts.DEFAULT_NUM_BEAMS,
                max_new_tokens=192,
            )
        decoded += [t.strip() for t in tok.batch_decode(gen, skip_special_tokens=True)]
        print(f"  urdu {i + len(chunk)}/{len(payload['urdu'])}", file=sys.stderr)
    out["urdu"] = decoded

json.dump(out, sys.stdout, ensure_ascii=False)
"""


def script_of(word: str) -> Script:
    """Which writing system a word is in, by majority of its letters."""
    counts: dict[Script, int] = {"urdu": 0, "devanagari": 0, "latin": 0, "other": 0}
    for char in word:
        code = ord(char)
        if 0x0900 <= code <= 0x097F:
            counts["devanagari"] += 1
        elif 0x0600 <= code <= 0x06FF or 0x0750 <= code <= 0x077F:
            counts["urdu"] += 1
        elif char.isascii() and char.isalpha():
            counts["latin"] += 1
    best = max(counts, key=lambda key: counts[key])
    return best if counts[best] else "other"


def split_runs(text: str) -> list[tuple[Script, str]]:
    """Break a segment into consecutive same-script runs.

    Punctuation and digits attach to the run they follow, so `کہ image ہے`
    stays three runs rather than fragmenting on every symbol.
    """
    runs: list[tuple[Script, list[str]]] = []
    for word in text.split():
        kind = script_of(word)
        if kind == "other" and runs:
            kind = runs[-1][0]
        if runs and runs[-1][0] == kind:
            runs[-1][1].append(word)
        else:
            runs.append((kind, [word]))
    return [(kind, " ".join(words)) for kind, words in runs]


# The transliteration model emits the literal string "thisishypenhere" when it
# meets an Urdu full stop -- a preprocessing placeholder that leaked into its
# training data and now comes back out as a word. It corrupted 17% of the first
# corpus run. Feeding it Latin punctuation instead avoids the trigger; the
# post-filter below catches any that still get through.
_PUNCT = str.maketrans({"۔": ".", "،": ",", "؟": "?", "؛": ";", "٫": ".", "٪": "%"})
_PLACEHOLDER = re.compile(r"\s*thisishypenhere\s*", re.I)


def romanize(segments: list[str], opencut_api: Path = OPENCUT_API) -> list[str]:
    """Romanize segments by calling OpenCut's service in its own environment.

    Raises:
        RuntimeError: if OpenCut's environment cannot run the worker. Failing
            loudly is deliberate -- OpenCut's own service degrades silently to
            the rule map, and that is precisely how its quality problem stayed
            invisible.
    """
    segments = [segment.translate(_PUNCT) for segment in segments]
    plan: list[list[tuple[Script, str]]] = [split_runs(segment) for segment in segments]

    request: dict[str, list[str]] = {"urdu": [], "devanagari": []}
    for runs in plan:
        for kind, text in runs:
            if kind in ("urdu", "devanagari"):
                request[kind].append(text)

    if not request["urdu"] and not request["devanagari"]:
        return list(segments)

    result = subprocess.run(  # noqa: S603
        ["uv", "run", "--group", "tts", "python", "-c", _WORKER],
        cwd=opencut_api,
        input=json.dumps(request),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"OpenCut romanizer failed (exit {result.returncode}).\n"
            f"cwd={opencut_api}\n{result.stderr[-1500:]}"
        )

    romanized = json.loads(result.stdout)
    queues = {kind: iter(values) for kind, values in romanized.items()}

    output: list[str] = []
    for runs in plan:
        parts = [
            next(queues[kind]) if kind in queues and kind in ("urdu", "devanagari") else text
            for kind, text in runs
        ]
        joined = " ".join(part for part in parts if part)
        output.append(_PLACEHOLDER.sub(". ", joined).strip())
    return output


def main(
    text: str = "",
    infile: str = "",
) -> None:
    """Romanize text from --text or a file of one segment per line."""
    if text:
        segments = [text]
    elif infile:
        segments = [line.strip() for line in Path(infile).read_text(encoding="utf-8").splitlines()]
        segments = [line for line in segments if line]
    else:
        segments = [line.strip() for line in sys.stdin.read().splitlines() if line.strip()]

    for source, roman in zip(segments, romanize(segments), strict=True):
        print(f"{source}\n  -> {roman}")


if __name__ == "__main__":  # pragma: no cover
    import typer

    typer.run(main)
