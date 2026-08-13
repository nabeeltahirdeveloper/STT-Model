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
import random
import time
from pathlib import Path

MANIFEST = Path("data/labels/train-subset.jsonl")
LOCAL = Path("data/raw/urduspeech")

# The Hub throttles by request rate, and a clip is a single small file, so
# concurrency buys throughput right up to the point where it stops buying
# anything: 16 workers over 13,470 clips returned 11,910 HTTP failures in 169
# seconds -- roughly 80 requests a second, answered with 429s. Eight is below
# the point where the Hub pushes back, and the retry below absorbs the rest.
WORKERS = 8
RETRIES = 5


def _fetch_with_retry(
    download: object,
    repo: str,
    name: str,
    root: Path,
    retries: int,
    sleep: object = time.sleep,
    jitter: object = random.random,
) -> str | None:
    """Download one clip, backing off on failure. Returns an error, or None.

    Rate limiting is transient by definition, so a failed clip is retried rather
    than recorded: the earlier version treated a 429 as permanent and lost 88%
    of the download to it. Backoff is exponential with jitter, because 8 threads
    retrying in lockstep reproduce the burst that caused the throttling.
    """
    delay = 1.0
    for attempt in range(retries):
        try:
            download(repo, name, repo_type="dataset", local_dir=str(root))  # type: ignore[operator]
        except Exception as error:  # noqa: BLE001 - one bad clip must not end the run
            if attempt == retries - 1:
                # The status code is the whole diagnosis -- 429 means slow down,
                # 401 means the token is wrong, 404 means the manifest is stale.
                # Recording only the exception type hid that distinction once.
                return f"{name}: {type(error).__name__}: {str(error).splitlines()[0][:160]}"
            sleep(delay + jitter())  # type: ignore[operator]
            delay *= 2
        else:
            return None
    return None


def main(
    manifest: str = str(MANIFEST),
    local_dir: str = str(LOCAL),
    repo: str = "ASLP-lab/UrduSpeech",
    workers: int = WORKERS,
    retries: int = RETRIES,
) -> None:
    """Fetch every clip the manifest references that is not already local."""
    import concurrent.futures
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
    print(f"{len(wanted):,} to download; the rest are already on disk")
    print(f"{workers} workers, {retries} attempts each\n")
    if not wanted:
        print("nothing to do")
        return

    failed: list[str] = []
    done_count = 0
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_fetch_with_retry, hf_hub_download, repo, name, root, retries)
            for name in wanted
        ]
        for future in concurrent.futures.as_completed(futures):
            done_count += 1
            if done_count % 250 == 0 or done_count == len(wanted):
                rate = done_count / max(time.time() - started, 1e-9)
                left = (len(wanted) - done_count) / max(rate, 1e-9) / 60
                print(
                    f"  {done_count:,}/{len(wanted):,}"
                    f" · {len(failed):,} failed · {left:.0f} min left",
                    flush=True,
                )
            err = future.result()
            if err:
                failed.append(err)

    # local_dir keeps a .cache of metadata beside the files; it is not needed
    # once the download is complete and it doubles the disk cost.
    shutil.rmtree(root / ".cache", ignore_errors=True)

    fetched = len(wanted) - len(failed)
    print(f"\ndone. {fetched:,} fetched, {len(failed):,} failed")
    for line in failed[:10]:
        print(f"  {line}")

    if failed:
        print("\nRe-run to retry the failures; existing files are skipped.")
        # A partial download used to exit 0, so the run continued and the
        # shortfall surfaced later as an unexplained error from the training
        # script. Exit non-zero so the failure is attributed where it happened.
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    import typer

    typer.run(main)
