"""SQLModel tables and request/response shapes for the Gemmacademy API."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel
from sqlmodel import Field, SQLModel


class CamelModel(BaseModel):
    """Base for API response models — serializes field names as camelCase."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
    )


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# DB tables
# ---------------------------------------------------------------------------


class Class(SQLModel, table=True):
    id: str = Field(primary_key=True)
    name: str
    grade: str
    subject: str
    created_at: datetime = Field(default_factory=utcnow)
    status: str = "training"  # "training" | "ready" | "failed" | "deleted"
    model_url: Optional[str] = None
    model_size_bytes: Optional[int] = None
    training_examples: Optional[int] = None
    error_message: Optional[str] = None


class Job(SQLModel, table=True):
    id: str = Field(primary_key=True)
    class_id: str = Field(foreign_key="class.id", index=True)
    status: str = "queued"  # "queued" | "running" | "complete" | "failed"
    started_at: datetime = Field(default_factory=utcnow)
    completed_at: Optional[datetime] = None
    current_stage: str = "queued"  # reading | generating | training | packaging | ready
    stage_progress: float = 0.0
    questions_generated: Optional[int] = None
    questions_target: Optional[int] = None
    train_loss: Optional[float] = None
    sample_qa_json: Optional[str] = None
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# API response shapes — keep in sync with the dashboard tRPC contract
# ---------------------------------------------------------------------------


class ClassSummary(CamelModel):
    id: str
    name: str
    grade: str
    subject: str
    created_at: datetime
    status: str
    model_url: Optional[str] = None


class ClassDetail(ClassSummary):
    model_size_bytes: Optional[int] = None
    training_examples: Optional[int] = None
    error_message: Optional[str] = None


class QAPair(CamelModel):
    q: str
    a: str


class JobStatus(CamelModel):
    id: str
    class_id: str
    status: str
    current_stage: str
    stage_progress: float
    started_at: datetime
    completed_at: Optional[datetime] = None
    questions_generated: Optional[int] = None
    questions_target: Optional[int] = None
    train_loss: Optional[float] = None
    sample_qa: list[QAPair] = []
    error_message: Optional[str] = None


class CreateClassResponse(CamelModel):
    id: str
    job_id: str


class HealthResponse(CamelModel):
    status: str
    vllm_available: bool
    gpu_count: int
    active_jobs: int
    demo_mode: bool
