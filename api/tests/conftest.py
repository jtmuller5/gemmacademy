"""Test fixtures.

Each test runs against a fresh sqlite DB and a temp upload/output dir, with
DEMO_MODE on so we never try to call vLLM, train Gemma, or push to HF.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

API_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(API_ROOT))


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.setenv("UPLOADS_ROOT", str(tmp_path / "uploads"))
    monkeypatch.setenv("OUTPUTS_ROOT", str(tmp_path / "outputs"))
    # Cwd matters because storage uses relative paths off the project root
    monkeypatch.chdir(tmp_path)

    # Force a fresh import of modules whose module-level state depends on
    # env vars (DB engine, upload/output roots, demo-mode flag). Don't
    # reload models.py — re-declaring the SQLModel tables would conflict
    # with the existing ones in SQLModel.metadata.
    for mod in [
        "src.db",
        "src.storage",
        "src.pipeline",
        "src.jobs",
        "src.routes",
        "src.main",
    ]:
        if mod in sys.modules:
            importlib.reload(sys.modules[mod])
    yield
