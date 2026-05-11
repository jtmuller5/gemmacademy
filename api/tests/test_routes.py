"""Happy-path integration: upload -> queued -> running -> complete.

Runs in DEMO_MODE so no GPU / vLLM / HF is needed. Sleep ticks are sped up via
DEMO_TICK_SECONDS so the test finishes in a few seconds.
"""

from __future__ import annotations

import time


def test_demo_pipeline_end_to_end(monkeypatch):
    monkeypatch.setenv("DEMO_TICK_SECONDS", "0.05")

    from fastapi.testclient import TestClient

    from src.main import app

    with TestClient(app) as client:
        # Health
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["demo_mode"] is True

        # Empty list to start
        r = client.get("/classes")
        assert r.status_code == 200
        assert r.json() == []

        # Upload
        files = [
            ("files", ("lesson.txt", b"Mrs. Henderson teaches the Pizza Method for fractions.", "text/plain")),
        ]
        r = client.post(
            "/classes",
            data={"name": "4th Grade Math", "grade": "4", "subject": "Math"},
            files=files,
        )
        assert r.status_code == 200, r.text
        body = r.json()
        class_id = body["id"]
        job_id = body["job_id"]

        # Class exists
        r = client.get(f"/classes/{class_id}")
        assert r.status_code == 200
        assert r.json()["status"] in {"training", "ready"}

        # Poll the job
        deadline = time.time() + 25
        last = None
        while time.time() < deadline:
            r = client.get(f"/jobs/{job_id}")
            assert r.status_code == 200
            last = r.json()
            if last["status"] == "complete":
                break
            time.sleep(0.2)

        assert last is not None
        assert last["status"] == "complete", last
        assert last["current_stage"] == "ready"
        assert last["stage_progress"] == 1.0

        # Class is ready and points at a model
        r = client.get(f"/classes/{class_id}")
        detail = r.json()
        assert detail["status"] == "ready"
        assert detail["model_url"].startswith("https://huggingface.co/")
        assert detail["training_examples"] is not None

        # Soft-delete works
        r = client.delete(f"/classes/{class_id}")
        assert r.status_code == 200
        assert r.json() == {"success": True}
        r = client.get("/classes")
        assert r.json() == []


def test_validation_rejects_unsupported_extension():
    from fastapi.testclient import TestClient

    from src.main import app

    with TestClient(app) as client:
        files = [("files", ("payload.exe", b"\x00\x01", "application/octet-stream"))]
        r = client.post(
            "/classes",
            data={"name": "x", "grade": "4", "subject": "Math"},
            files=files,
        )
        assert r.status_code == 422
