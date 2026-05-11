"""HTTP endpoints for the Gemmacademy API."""

from __future__ import annotations

import json
import os
import shutil
import socket
import uuid
from datetime import datetime, timezone

import requests
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from sqlmodel import select

from . import jobs, storage
from .db import get_session
from .models import (
    Class,
    ClassDetail,
    ClassSummary,
    CreateClassResponse,
    HealthResponse,
    Job,
    JobStatus,
    QAPair,
)
from .pipeline import is_demo_mode

router = APIRouter()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        vllm_available=_check_vllm(),
        gpu_count=_count_gpus(),
        active_jobs=jobs.active_job_count(),
        demo_mode=is_demo_mode(),
    )


def _check_vllm() -> bool:
    url = os.environ.get("VLLM_URL", "http://localhost:8000/v1/models")
    try:
        r = requests.get(url, timeout=1.5)
        return r.status_code == 200
    except (requests.RequestException, socket.error):
        return False


def _count_gpus() -> int:
    try:
        out = shutil.which("nvidia-smi")
        if not out:
            return 0
        import subprocess

        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=2,
        )
        if r.returncode != 0:
            return 0
        return sum(1 for line in r.stdout.splitlines() if line.strip())
    except Exception:  # noqa: BLE001
        return 0


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------


@router.get("/classes", response_model=list[ClassSummary])
def list_classes() -> list[ClassSummary]:
    with get_session() as session:
        rows = session.exec(
            select(Class).where(Class.status != "deleted").order_by(Class.created_at.desc())
        ).all()
        return [ClassSummary.model_validate(row, from_attributes=True) for row in rows]


@router.get("/classes/{class_id}", response_model=ClassDetail)
def get_class(class_id: str) -> ClassDetail:
    with get_session() as session:
        klass = session.get(Class, class_id)
        if klass is None or klass.status == "deleted":
            raise HTTPException(status_code=404, detail="Class not found")
        return ClassDetail.model_validate(klass, from_attributes=True)


@router.delete("/classes/{class_id}")
def delete_class(class_id: str) -> dict:
    with get_session() as session:
        klass = session.get(Class, class_id)
        if klass is None:
            return {"success": True}  # idempotent
        klass.status = "deleted"
        session.add(klass)
        session.commit()
    return {"success": True}


@router.post("/classes", response_model=CreateClassResponse)
async def create_class(
    name: str = Form(...),
    grade: str = Form(...),
    subject: str = Form(...),
    files: list[UploadFile] = File(...),
) -> CreateClassResponse:
    if not name.strip():
        raise HTTPException(status_code=422, detail="name is required")
    if not files:
        raise HTTPException(status_code=422, detail="At least one file is required")
    if len(files) > storage.MAX_FILES:
        raise HTTPException(status_code=422, detail=f"Too many files (max {storage.MAX_FILES})")

    class_id = uuid.uuid4().hex
    job_id = uuid.uuid4().hex
    upload_dir = storage.upload_dir_for(job_id)

    total_bytes = 0
    for f in files:
        original = f.filename or "upload.bin"
        ext = "." + original.rsplit(".", 1)[-1].lower() if "." in original else ""
        if ext not in storage.ALLOWED_EXTENSIONS:
            storage.remove_dir(upload_dir)
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported file type: {original} ({ext or 'no extension'})",
            )
        dest = upload_dir / storage.safe_filename(original)
        size = 0
        with dest.open("wb") as out:
            while chunk := await f.read(1024 * 1024):
                size += len(chunk)
                total_bytes += len(chunk)
                if total_bytes > storage.MAX_TOTAL_BYTES:
                    storage.remove_dir(upload_dir)
                    raise HTTPException(
                        status_code=422,
                        detail=f"Total upload exceeds {storage.MAX_TOTAL_BYTES // (1024*1024)} MB",
                    )
                out.write(chunk)

    now = datetime.now(timezone.utc)
    with get_session() as session:
        klass = Class(
            id=class_id,
            name=name.strip(),
            grade=grade.strip(),
            subject=subject.strip(),
            created_at=now,
            status="training",
        )
        job = Job(
            id=job_id,
            class_id=class_id,
            status="queued",
            started_at=now,
            current_stage="queued",
            stage_progress=0.0,
        )
        session.add(klass)
        session.add(job)
        session.commit()

    await jobs.start_worker()
    return CreateClassResponse(id=class_id, job_id=job_id)


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


@router.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str) -> JobStatus:
    with get_session() as session:
        job = session.get(Job, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")

        sample_qa: list[QAPair] = []
        if job.sample_qa_json:
            try:
                raw = json.loads(job.sample_qa_json)
                sample_qa = [QAPair(**p) for p in raw if "q" in p and "a" in p][:3]
            except (json.JSONDecodeError, TypeError):
                sample_qa = []

        return JobStatus(
            id=job.id,
            class_id=job.class_id,
            status=job.status,
            current_stage=job.current_stage,
            stage_progress=job.stage_progress,
            started_at=job.started_at,
            completed_at=job.completed_at,
            questions_generated=job.questions_generated,
            questions_target=job.questions_target,
            train_loss=job.train_loss,
            sample_qa=sample_qa,
            error_message=job.error_message,
        )
