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
