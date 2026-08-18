"""Retry behaviour for the training-audio fetcher.

A Colab run lost 11,910 of 13,470 clips to HTTP failures and reported exit 0,
so the shortfall surfaced two cells later as an unexplained error from the
training script. These tests cover the two things that went wrong: a transient
failure was treated as permanent, and a mass failure was treated as success.
"""

from __future__ import annotations

import pytest

from scripts.fetch_training_audio import _fetch_with_retry


class _Flaky:
    """Fails `failures` times, then succeeds."""

    def __init__(self, failures: int, error: Exception | None = None) -> None:
        self.failures = failures
        self.calls = 0
        self.error = error or RuntimeError("429 Client Error: Too Many Requests")

    def __call__(self, *args: object, **kwargs: object) -> None:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error


def _run(download: object, retries: int = 5) -> tuple[str | None, list[float]]:
    slept: list[float] = []
    from pathlib import Path

    err = _fetch_with_retry(
        download,
        "repo",
        "clip.wav",
        Path("/tmp"),
        retries,
        sleep=slept.append,
        jitter=lambda: 0.0,
    )
    return err, slept


def test_a_clip_that_succeeds_first_time_is_not_retried() -> None:
    download = _Flaky(failures=0)
    err, slept = _run(download)
    assert err is None
    assert download.calls == 1
    assert slept == []


@pytest.mark.parametrize("failures", [1, 2, 4])
def test_a_transient_failure_is_retried_until_it_succeeds(failures: int) -> None:
    """The regression: a 429 is rate limiting, not a verdict on the file."""
    download = _Flaky(failures=failures)
    err, _ = _run(download)
    assert err is None
    assert download.calls == failures + 1


def test_backoff_is_exponential() -> None:
    """8 threads retrying in lockstep recreate the burst that caused the 429."""
    _, slept = _run(_Flaky(failures=3))
    assert slept == [1.0, 2.0, 4.0]


def test_a_permanent_failure_reports_the_status_line_not_just_the_type() -> None:
    """`HfHubHTTPError` alone cannot distinguish 429 from 401 from 404."""
    download = _Flaky(failures=99, error=RuntimeError("401 Client Error: Unauthorized"))
    err, _ = _run(download, retries=2)
    assert err is not None
    assert "clip.wav" in err
    assert "401" in err
    assert "Unauthorized" in err


def test_retries_are_bounded() -> None:
    download = _Flaky(failures=99)
    err, slept = _run(download, retries=3)
    assert err is not None
    assert download.calls == 3
    assert len(slept) == 2  # no sleep after the final attempt


def test_a_multiline_error_is_collapsed_to_one_line() -> None:
    """Hub errors carry a request id and a URL on following lines."""
    download = _Flaky(
        failures=99, error=RuntimeError("429 Too Many Requests\nrequest id: abc\nurl")
    )
    err, _ = _run(download, retries=1)
    assert err is not None
    assert "\n" not in err
    assert "429" in err


class TestRepoPathAliases:
    """`PODCAST/` and `Podcast/` cannot coexist on a case-insensitive filesystem.

    The eval set was built on macOS, where the second was written as
    `Podcast2/`, and the manifest recorded that. The local path is therefore
    correct and the repository path is not: 36 of the 269 eval clips 404 on
    Linux without the alias. data/eval/ is frozen (constraint 5), so the
    divergence is resolved in the fetcher rather than by rewriting the manifest.
    """

    def test_podcast2_maps_to_the_directory_the_hub_serves(self) -> None:
        from scripts.fetch_training_audio import repo_path

        local = "benchmark/US-benchmark-CS/short/Podcast2/audio/SPEAKER_002_Podcast_0167.wav"
        assert repo_path(local) == (
            "benchmark/US-benchmark-CS/short/Podcast/audio/SPEAKER_002_Podcast_0167.wav"
        )

    def test_unrelated_paths_are_untouched(self) -> None:
        from scripts.fetch_training_audio import repo_path

        path = "corpus/US-CS/short/drama/audio/SPEAKER_1216_DRAMA_011503.wav"
        assert repo_path(path) == path

    def test_the_uppercase_podcast_directory_is_not_rewritten(self) -> None:
        """`PODCAST/` is a real, distinct directory. Only `Podcast2/` is local."""
        from scripts.fetch_training_audio import repo_path

        path = "benchmark/US-benchmark-CS/long/PODCAST/audio/SPEAKER_000_PODCAST_0002.WAV"
        assert repo_path(path) == path

    def test_a_downloaded_alias_is_moved_to_the_manifest_path(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Downloading is not enough: local_dir writes under the *repo* path."""
        from scripts.fetch_training_audio import _fetch_with_retry

        local = "benchmark/US-benchmark-CS/short/Podcast2/audio/a.wav"
        remote = "benchmark/US-benchmark-CS/short/Podcast/audio/a.wav"

        def fake_download(repo, name, repo_type, local_dir):  # type: ignore[no-untyped-def]
            assert name == remote, f"asked the Hub for {name}"
            written = tmp_path / name
            written.parent.mkdir(parents=True, exist_ok=True)
            written.write_bytes(b"audio")

        assert _fetch_with_retry(fake_download, "r", local, tmp_path, 1) is None
        assert (tmp_path / local).exists(), "file must land where the manifest expects it"
        assert not (tmp_path / remote).exists(), "and must not be left at the repo path"


class TestArchiveFallback:
    """One request beats thousands, but a missing archive must not be fatal.

    The Hub throttles by request count -- ~1,000 per 5 minutes, one clip one
    request -- so 13,470 clips have a ~35 minute floor whatever the thread
    count. That was paid three times in a week. A single tar is one request at
    CDN speed. The optimisation is opportunistic, so its absence falls back.
    """

    def test_a_missing_archive_returns_zero_rather_than_raising(
        self, tmp_path, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        import scripts.fetch_training_audio as module

        def boom(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise OSError("404")

        monkeypatch.setattr("huggingface_hub.hf_hub_download", boom)
        assert module.fetch_archive("nope/nope", tmp_path / "urduspeech") == 0

    def test_it_extracts_and_reports_the_count(self, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        import tarfile

        import scripts.fetch_training_audio as module

        payload = tmp_path / "src" / "urduspeech" / "corpus"
        payload.mkdir(parents=True)
        for name in ("a.wav", "b.wav"):
            (payload / name).write_bytes(b"audio")
        tar_path = tmp_path / "urduspeech-audio.tar"
        with tarfile.open(tar_path, "w") as archive:
            archive.add(tmp_path / "src" / "urduspeech", arcname="urduspeech")

        monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *a, **k: str(tar_path))
        root = tmp_path / "out" / "urduspeech"
        root.parent.mkdir(parents=True, exist_ok=True)
        assert module.fetch_archive("repo", root) == 2
        assert (root / "corpus" / "a.wav").exists()

    def test_members_escaping_the_root_are_skipped(self, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        """The archive is ours, but a tar writing outside its root is worth never having."""
        import tarfile

        import scripts.fetch_training_audio as module

        evil = tmp_path / "evil.wav"
        evil.write_bytes(b"x")
        tar_path = tmp_path / "urduspeech-audio.tar"
        with tarfile.open(tar_path, "w") as archive:
            archive.add(evil, arcname="../escaped.wav")

        monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *a, **k: str(tar_path))
        root = tmp_path / "out" / "urduspeech"
        root.parent.mkdir(parents=True, exist_ok=True)
        assert module.fetch_archive("repo", root) == 0
        assert not (tmp_path / "out" / "escaped.wav").exists()
        assert not (tmp_path / "escaped.wav").exists()
