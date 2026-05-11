"""SQLite engine + session helpers."""

from __future__ import annotations

import os
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine

DB_PATH = Path(os.environ.get("DB_PATH", "./jobs.db")).resolve()
_engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False},
    echo=False,
)


def init_db() -> None:
    # Importing here so the SQLModel metadata is registered before create_all.
    from . import models  # noqa: F401

    SQLModel.metadata.create_all(_engine)


def get_session() -> Session:
    return Session(_engine)
