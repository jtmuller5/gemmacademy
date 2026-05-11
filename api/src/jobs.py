"""Single-worker job runner.

One asyncio task picks the next queued job from sqlite, runs the pipeline,
writes Job + Class state back to sqlite. No multi-worker / queue / Redis —
the rig can only train one job at a time anyway (GPU memory ceiling).
"""

from __future__ import annotations

import asyncio
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlmodel import select

from . import pipeline, storage
from .db import get_session
from .models import Class, Job

_worker_task: Optional[asyncio.Task] = None
_worker_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


async def start_worker() -> None:
    global _worker_task
    async with _worker_lock:
        if _worker_task is None or _worker_task.done():
            _worker_task = asyncio.create_task(_worker_loop(), name="job-worker")


def active_job_count() -> int:
    with get_session() as session:
        rows = session.exec(select(Job).where(Job.status == "running")).all()
        return len(rows)


# ---------------------------------------------------------------------------
# Worker loop
# ---------------------------------------------------------------------------


async def _worker_loop() -> None:
    print("[worker] started", flush=True)
    while True:
        job = _pick_next_queued_job()
        if job is None:
            await asyncio.sleep(2)
            continue
        print(f"[worker] picked up job {job.id} for class {job.class_id}", flush=True)
        try:
            await _run_job(job)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            _mark_job_failed(job.id, str(e))
            _mark_class_failed(job.class_id, str(e))


def _pick_next_queued_job() -> Optional[Job]:
    with get_session() as session:
        return session.exec(
            select(Job).where(Job.status == "queued").order_by(Job.started_at)
        ).first()


# ---------------------------------------------------------------------------
# Single-job execution
# ---------------------------------------------------------------------------


async def _run_job(job: Job) -> None:
    job_id = job.id
    class_id = job.class_id

    _update_job(job_id, status="running", current_stage="reading", stage_progress=0.0)

    upload_dir = storage.upload_dir_for(job_id)
    output_dir = storage.output_dir_for(class_id)

    async def on_stage(
        *,
        stage: str,
        progress: float,
        questions_generated: Optional[int] = None,
        questions_target: Optional[int] = None,
        train_loss: Optional[float] = None,
        sample_qa: Optional[list[dict]] = None,
    ) -> None:
        fields: dict = {
            "current_stage": stage,
            "stage_progress": float(progress),
        }
        if questions_generated is not None:
            fields["questions_generated"] = questions_generated
        if questions_target is not None:
            fields["questions_target"] = questions_target
        if train_loss is not None:
            fields["train_loss"] = float(train_loss)
        if sample_qa is not None:
            fields["sample_qa_json"] = json.dumps(sample_qa)
        _update_job(job_id, **fields)

    async def on_complete(
        *,
        model_url: str,
        model_size_bytes: Optional[int],
        training_examples: Optional[int],
    ) -> None:
        _mark_class_ready(
            class_id,
            model_url=model_url,
            model_size_bytes=model_size_bytes,
            training_examples=training_examples,
        )
        _update_job(
            job_id,
            status="complete",
            current_stage="ready",
            stage_progress=1.0,
            completed_at=datetime.now(timezone.utc),
        )

    await pipeline.run_pipeline(
        job_id=job_id,
        class_id=class_id,
        upload_dir=upload_dir,
        output_dir=output_dir,
        on_stage=on_stage,
        on_complete=on_complete,
    )


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def _update_job(job_id: str, **fields) -> None:
    with get_session() as session:
        job = session.get(Job, job_id)
        if job is None:
            return
        for k, v in fields.items():
            setattr(job, k, v)
        session.add(job)
        session.commit()


def _mark_job_failed(job_id: str, message: str) -> None:
    _update_job(
        job_id,
        status="failed",
        error_message=message[:1000],
        completed_at=datetime.now(timezone.utc),
    )


def _mark_class_failed(class_id: str, message: str) -> None:
    with get_session() as session:
        klass = session.get(Class, class_id)
        if klass is None:
            return
        klass.status = "failed"
        klass.error_message = message[:1000]
        session.add(klass)
        session.commit()


def _mark_class_ready(
    class_id: str,
    *,
    model_url: str,
    model_size_bytes: Optional[int],
    training_examples: Optional[int],
) -> None:
    with get_session() as session:
        klass = session.get(Class, class_id)
        if klass is None:
            return
        klass.status = "ready"
        klass.model_url = model_url
        if model_size_bytes is not None:
            klass.model_size_bytes = int(model_size_bytes)
        if training_examples is not None:
            klass.training_examples = int(training_examples)
        klass.error_message = None
        session.add(klass)
        session.commit()
