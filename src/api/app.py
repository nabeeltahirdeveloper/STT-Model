"""FastAPI service.

STUB -- Phase 3. The routes and response shapes are defined now so the contract
is reviewable before there is anything behind it.

Captioning is asynchronous by design: PROJECT.md §1.4 targets 0.3x realtime, so
a 40-minute upload is a 12-minute job. Holding an HTTP connection open for that
is not viable, hence submit-then-poll.
"""

from __future__ import annotations

import uuid
from enum import StrEnum
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

app = FastAPI(
    title="Roman Urdu Captions",
    version="0.1.0",
    description="Roman Urdu subtitles for code-switched Urdu-English video.",
)


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class JobCreated(BaseModel):
    job_id: str
    status: JobStatus


class Job(BaseModel):
    job_id: str
    status: JobStatus
    stage: str | None = None
    srt: str | None = None
    vtt: str | None = None
    transcript: str | None = None
    error: str | None = None


# Placeholder for the real queue. A dict is wrong for anything multi-process and
# is here only so the route signatures typecheck.
_JOBS: dict[str, Job] = {}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/transcribe", response_model=JobCreated, status_code=202)
async def transcribe(file: Annotated[UploadFile, File()]) -> JobCreated:
    """Accept a video upload and queue a captioning job.

    Raises:
        NotImplementedError: Phase 3 work.
    """
    job_id = str(uuid.uuid4())
    raise NotImplementedError(f"Phase 3: enqueue {job_id} for {file.filename}")


@app.get("/jobs/{job_id}", response_model=Job)
def get_job(job_id: str) -> Job:
    """Poll a job. Returns output paths once `status` is `done`."""
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"no such job: {job_id}")
    return job
