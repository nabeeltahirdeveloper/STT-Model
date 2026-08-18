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

# The corpus has `PODCAST/` and `Podcast/` as separate directories, which cannot
# both exist on a case-insensitive filesystem. The eval set was built on macOS,
# where the second one was written as `Podcast2/`, and the manifest recorded
# that -- so the local path is right and the *repository* path is not. 36 of the
# 269 eval clips 404 on Linux without this. The manifest is not rewritten
# because data/eval/ is frozen (constraint 5); the divergence is resolved here,
# where it arises.
REPO_ALIASES = {"/Podcast2/": "/Podcast/"}


def repo_path(local_relative: str) -> str:
    """Map a local path to the path the Hub actually serves."""
    for local, remote in REPO_ALIASES.items():
        if local in local_relative:
            return local_relative.replace(local, remote)
    return local_relative


# The Hub throttles by request rate, and a clip is a single small file, so
# concurrency buys throughput right up to the point where it stops buying
# anything: 16 workers over 13,470 clips returned 11,910 HTTP failures in 169
# seconds -- roughly 80 requests a second, answered with 429s. Eight is below
# the point where the Hub pushes back, and the retry below absorbs the rest.
WORKERS = 8
RETRIES = 5


def fetch_archive(repo: str, root: Path, member_prefix: str = "urduspeech") -> int:
    """Pull the whole corpus as one tar and extract it. Returns clips extracted.

    The Hub throttles by request count, not bytes: ~1,000 requests per 5
    minutes, and one clip is one request. 13,470 clips therefore have a floor
    near 35 minutes however many threads ask, and that cost was paid three
    times in one week because /content is wiped between sessions.

    One archive is one request. It arrives at CDN speed -- roughly two minutes
    for 12 GB inside Google's network -- and the rate limit never applies.

    Returns 0 rather than raising if the archive is absent or unreadable, so
    the caller can fall back to fetching clip by clip. A missing optimisation
    must not be a failure.
    """
    import tarfile

    from huggingface_hub import hf_hub_download

    try:
        print(f"archive: trying {repo} ...", flush=True)
        local = hf_hub_download(
            repo, "urduspeech-audio.tar", repo_type="dataset", local_dir=str(root.parent)
        )
    except Exception as error:  # noqa: BLE001 - absence is a fallback, not a failure
        print(f"archive unavailable ({type(error).__name__}); falling back to per-clip")
        return 0

    print(f"extracting {local} ...", flush=True)
    extracted = 0
    with tarfile.open(local) as archive:
        for member in archive:
            # Refuse paths that escape the destination. The archive is ours, but
            # a tar that writes outside its root is the one bug in this pattern
            # worth never having.
            if member.name.startswith(("/", "..")) or ".." in Path(member.name).parts:
                continue
            if not member.name.startswith(member_prefix):
                continue
            archive.extract(member, path=root.parent, filter="data")
            if member.isfile():
                extracted += 1
    Path(local).unlink(missing_ok=True)
    print(f"archive: {extracted:,} files extracted")
    return extracted


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
    remote = repo_path(name)
    delay = 1.0
    for attempt in range(retries):
        try:
            download(repo, remote, repo_type="dataset", local_dir=str(root))  # type: ignore[operator]
            # local_dir writes the file under its *repository* path. Where that
            # differs from the manifest's path, move it to where the manifest
            # says, or every later `audio.exists()` check fails on a file that
            # was in fact downloaded.
            if remote != name:
                destination = root / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                (root / remote).replace(destination)
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
    archive: str = "",
) -> None:
    """Fetch every clip the manifest references that is not already local.

    Args:
        archive: a dataset repo holding `urduspeech-audio.tar`. Tried first;
            one request instead of thousands. Falls back to per-clip fetching
            if it is missing, so the flag is safe to leave set.
    """
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

    # One request beats thousands. Only worth it when a lot is missing --
    # below that, pulling 12 GB to obtain a handful of clips is the slower path.
    if archive and len(wanted) > 500 and fetch_archive(archive, root):
        still = [name for name in wanted if not (root / name).exists()]
        print(f"{len(wanted) - len(still):,} of {len(wanted):,} clips came from the archive")
        wanted = still
        if not wanted:
            print("nothing left to fetch")
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
