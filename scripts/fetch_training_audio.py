"""Download exactly the audio clips named in a training manifest.

    uv run python -m scripts.fetch_training_audio --manifest data/labels/train-subset.jsonl

Not a directory download. `--include "corpus/US-CS/short/drama/audio/*"` fetches
every drama clip, which is most of the 56 GB the subset exists to avoid. This
resolves each row's `audio` path to its repository path and fetches those files
and no others.

Resumable and safe to re-run: a file already on disk is skipped, so an
interrupted download costs only what it had not yet reached.
"""

from __future__ import annotations

import json
from pathlib import Path

MANIFEST = Path("data/labels/train-subset.jsonl")
LOCAL = Path("data/raw/urduspeech")


def main(
    manifest: str = str(MANIFEST),
    local_dir: str = str(LOCAL),
    repo: str = "ASLP-lab/UrduSpeech",
) -> None:
    """Fetch every clip the manifest references that is not already local."""
    import shutil

    from huggingface_hub import hf_hub_download

    rows = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines()]
    root = Path(local_dir)

    wanted: list[str] = []
    for row in rows:
        path = Path(str(row["audio"]))
        if path.exists():
            continue
        wanted.append(str(path.relative_to(root)))
    wanted = sorted(set(wanted))

    hours = sum(float(r.get("duration_s") or 0) for r in rows) / 3600
    print(f"{len(rows):,} clips in the manifest ({hours:.1f} h)")
    print(f"{len(wanted):,} to download; the rest are already on disk\n")
    if not wanted:
        print("nothing to do")
        return

    failed: list[str] = []
    for index, name in enumerate(wanted, start=1):
        try:
            hf_hub_download(repo, name, repo_type="dataset", local_dir=str(root))
        except Exception as error:  # noqa: BLE001 - one bad clip must not end the run
            failed.append(f"{name}: {type(error).__name__}")
        if index % 250 == 0 or index == len(wanted):
            print(f"  {index:,}/{len(wanted):,}", flush=True)

    # local_dir keeps a .cache of metadata beside the files; it is not needed
    # once the download is complete and it doubles the disk cost.
    shutil.rmtree(root / ".cache", ignore_errors=True)

    print(f"\ndone. {len(wanted) - len(failed):,} fetched, {len(failed)} failed")
    for line in failed[:10]:
        print(f"  {line}")
    if failed:
        print("Re-run to retry the failures; existing files are skipped.")


if __name__ == "__main__":  # pragma: no cover
    import typer

    typer.run(main)
