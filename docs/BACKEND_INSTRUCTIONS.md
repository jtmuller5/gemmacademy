# Gemmacademy — Backend API Instructions

> Build a FastAPI service that wraps the existing Phase 2 training pipeline and serves the dashboard's tRPC contract. The dashboard agent is building against a mock; your job is to make a real backend that returns the same data shapes so the mock can be swapped out.
>
> **Scope: a single Python service on the rig. One sqlite jobs table. One worker process.** No Celery, no Redis, no Kubernetes, no nothing. We are running this on one machine that also does training. Operational simplicity is a feature.

---

## Context (read this first)

The Gemmacademy project has three components in motion:

1. **Phase 2 pipeline (already built — DON'T rebuild):** training scripts, Q&A generator, conversion, HF upload. Lives in `~/projects/gemmacademy/training/`. Working end-to-end. The shipped model is up at `https://huggingface.co/jtmuller/gemmacademy-fractions-v1`.
2. **Dashboard (Next.js, on the MacBook):** built against a mock tRPC backend. See `FRONTEND_INSTRUCTIONS.md` for the contract.
3. **Android student app (Kotlin, on the MacBook):** downloads the model from HF and runs it offline.

You are building the bridge between #2 and #1. The dashboard needs a real backend that:
- Accepts file uploads from a teacher
- Runs the Phase 2 pipeline (PDF extract → vLLM Q&A gen → Unsloth fine-tune → litert-torch convert → HF push)
- Returns job status with progress per stage
- Returns a download URL when the job is done

**Read these files before writing any code:**
- `~/projects/gemmacademy/NOTES.md` — hard-won engineering gotchas
- `~/projects/gemmacademy/MORNING_SUMMARY.md` — the Phase 2 result
- `~/projects/gemmacademy/FRONTEND_INSTRUCTIONS.md` — the tRPC contract you must match
- `~/projects/gemmacademy/training/train.py` — the working trainer
- `~/projects/gemmacademy/training/generate_qa.py` — the working data generator
- `~/projects/gemmacademy/training/convert_real.sh` — the working conversion command

---

## Stack — non-negotiable

- **Python 3.12** (matches the existing rig env)
- **FastAPI** for the HTTP API
- **uvicorn** as the ASGI server
- **sqlite + sqlmodel** for the jobs table (file at `./jobs.db`, single file, easy to inspect)
- **Pydantic** for request/response models (comes with FastAPI)
- **`uv`** for package management — new project at `~/projects/gemmacademy/api/`

**Do not add:** Celery, RQ, Redis, RabbitMQ, Kafka, Postgres, MongoDB, SQLAlchemy ORM patterns, Alembic migrations, Pydantic Settings, structlog, OpenTelemetry. None of this is needed for a single-machine demo backend.

**Do not add:** authentication, OAuth, API keys, rate limiting. The service is open. The dashboard hits it directly. We address auth if and only if this becomes a real product.

**Do not add:** Docker, Kubernetes manifests, systemd units, nginx config. Run with `uv run uvicorn main:app --host 0.0.0.0 --port 8001` in a tmux session. That's it.

If you find yourself wanting to add a dependency, ask first.

---

## Project structure

Create at `~/projects/gemmacademy/api/`. Single Python project, single uv environment.

```
api/
├── pyproject.toml
├── README.md
├── jobs.db                   # sqlite, gitignored
├── uploads/                  # teacher-uploaded files; subdir per job
├── outputs/                  # symlinks/copies of completed model artifacts
├── src/
│   ├── main.py               # FastAPI app entry
│   ├── routes.py             # all HTTP endpoints
│   ├── models.py             # Pydantic request/response + sqlmodel tables
│   ├── jobs.py               # job runner: the worker loop
│   ├── pipeline.py           # wraps Phase 2 scripts as Python functions
│   ├── storage.py            # filesystem operations (upload dirs, output dirs)
│   └── hf.py                 # Hugging Face Hub upload wrapper
└── tests/
    └── test_routes.py        # one happy-path integration test
```

---

## The data model (sqlmodel tables)

Two tables. Mirror the dashboard's data contract.

### `Class` table

```python
class Class(SQLModel, table=True):
    id: str = Field(primary_key=True)              # uuid4 hex
    name: str
    grade: str                                      # "K","1",...,"8"
    subject: str                                    # "Math","Science","ELA","Social Studies","Other"
    created_at: datetime
    status: str                                     # "training" | "ready" | "failed"
    model_url: str | None = None                    # HF Hub URL when ready
    model_size_bytes: int | None = None
    training_examples: int | None = None
    error_message: str | None = None
```

### `Job` table

```python
class Job(SQLModel, table=True):
    id: str = Field(primary_key=True)               # uuid4 hex
    class_id: str = Field(foreign_key="class.id")
    status: str                                     # "queued" | "running" | "complete" | "failed"
    started_at: datetime
    completed_at: datetime | None = None
    current_stage: str                              # see stage progression below
    stage_progress: float = 0.0                     # 0..1 within current stage
    questions_generated: int | None = None
    questions_target: int | None = None
    train_loss: float | None = None
    sample_qa_json: str | None = None               # JSON-serialized list of {q, a}
    error_message: str | None = None
```

### Stage progression

Stages match the dashboard's expectations exactly:

```
"reading"     → extract text from uploaded PDFs/DOCX/TXT/MD
"generating"  → call vLLM, produce qa-fractions.jsonl
"training"    → run Unsloth fine-tune
"packaging"   → litert-torch export_hf, optionally push to HF
"ready"       → done, model_url is set on the Class
```

Failure at any stage sets `status=failed` and writes `error_message`.

---

## HTTP endpoints

These match the dashboard's tRPC procedure shapes. The dashboard calls these via fetch() from inside its tRPC resolvers; you don't need to speak tRPC's wire format.

### `GET /classes`
Returns `list[ClassSummary]` — list of all classes.

### `GET /classes/{class_id}`
Returns `ClassDetail` — full class record including model_url etc.

### `POST /classes`
Multipart form data:
- `name: str`
- `grade: str`
- `subject: str`
- `files: list[UploadFile]`

Validates inputs (max 10 files, total <50 MB, allowed extensions: pdf/docx/txt/md). Creates Class row with status=`training`, creates Job row with status=`queued`, writes uploaded files to `uploads/{job_id}/`, kicks off the worker, returns `{"id": class_id, "job_id": job_id}`.

**Why multipart and not the base64 the dashboard mock uses:** base64 was a mock convenience to keep things in the tRPC mutation. Real backends use multipart for file uploads. The dashboard's tRPC resolver should accept the base64 input, decode it, and POST as multipart to this endpoint. (Mention this in your README so whoever wires the dashboard knows.)

### `DELETE /classes/{class_id}`
Marks the Class as deleted (soft delete is fine — we never need this for the demo, just don't 500). Returns `{"success": true}`.

### `GET /jobs/{job_id}`
Returns `JobStatus`. The dashboard polls this every 2s during training. Make sure it's fast (<50ms). Return all fields the dashboard expects (current_stage, stage_progress, questions_generated, questions_target, train_loss, sample_qa, error_message).

### Health check: `GET /health`
Returns `{"status": "ok", "vllm_available": bool, "gpu_count": int, "active_jobs": int}` for sanity-checking. Doesn't need to be authenticated; it's diagnostic.

---

## CORS

The dashboard runs on `localhost:3000` (or wherever Next.js dev server is). The API runs on `localhost:8001` on the rig, accessible to the MacBook over Tailscale. CORS will block requests by default.

Allow:
- `http://localhost:3000`
- `http://localhost:3001`
- The Tailscale hostname of the MacBook (e.g., `http://macbook.tail-scale.ts.net:3000`)
- Any `*.ngrok.io` if the user demos via ngrok

Use FastAPI's `CORSMiddleware`. Don't try to be clever; just allow these origins explicitly.

---

## The job runner

This is the most important part of the service. The model is: a single async worker loop that processes one job at a time.

### Why one job at a time

The pipeline uses 30+ GB of GPU memory (vLLM serving 26B + Unsloth training E2B + litert-torch convert at different stages). Running two jobs concurrently will OOM. One job at a time is the right answer.

### Worker pattern

In `src/jobs.py`:

```python
import asyncio

_worker_task: asyncio.Task | None = None

async def start_worker():
    global _worker_task
    if _worker_task is None or _worker_task.done():
        _worker_task = asyncio.create_task(_worker_loop())

async def _worker_loop():
    while True:
        job = pick_next_queued_job()  # SELECT WHERE status='queued' ORDER BY started_at
        if job is None:
            await asyncio.sleep(2)
            continue
        try:
            await run_job(job)
        except Exception as e:
            mark_job_failed(job.id, str(e))
            mark_class_failed(job.class_id, str(e))
```

Start the worker on FastAPI app startup (`@app.on_event("startup")`). Don't bother with multiple workers, queues, or fanciness — sleep-poll is fine.

### `run_job(job)` calls into `pipeline.py`

The pipeline functions are thin wrappers around the existing Phase 2 scripts. **Do not duplicate logic. Call the existing scripts.**

In `src/pipeline.py`:

```python
from pathlib import Path
import subprocess
import json

TRAINING_DIR = Path("/home/joemuller/projects/gemmacademy/training")

async def run_pipeline(job_id: str, class_id: str, upload_dir: Path):
    # Stage: reading
    update_stage(job_id, "reading", 0.0)
    lesson_text = extract_text_from_uploads(upload_dir)
    save_lesson_text(class_id, lesson_text)
    update_stage(job_id, "reading", 1.0)

    # Stage: generating
    update_stage(job_id, "generating", 0.0)
    qa_jsonl_path = await run_qa_generator(class_id, lesson_text, on_progress=...)
    update_stage(job_id, "generating", 1.0)

    # Stage: training
    update_stage(job_id, "training", 0.0)
    merged_model_path = await run_training(class_id, qa_jsonl_path, on_progress=...)
    update_stage(job_id, "training", 1.0)

    # Stage: packaging
    update_stage(job_id, "packaging", 0.0)
    litertlm_path = run_conversion(merged_model_path, class_id)
    model_url = upload_to_hf(litertlm_path, class_id)
    update_stage(job_id, "packaging", 1.0)

    # Stage: ready
    mark_class_ready(class_id, model_url=model_url, model_size_bytes=..., training_examples=...)
    mark_job_complete(job_id)
```

**For the demo / hackathon, you have two implementation choices for the heavy stages:**

**Option A: Real pipeline.** Each stage actually runs the Phase 2 work. Generating Q&A takes ~3.5 min, training takes ~80 sec, conversion takes ~5 min. Total: 10-15 min per job. **This is the demo-correct option.**

**Option B: Demo-mode shortcut.** A flag in env that, when set, skips the actual training and points the model_url at the existing pre-trained `https://huggingface.co/jtmuller/gemmacademy-fractions-v1`. Stages still run with realistic timing (sleeps + simulated progress + sample Q&A pulled from `qa-fractions.jsonl`). Total: 30 sec per job. **This is the safe-for-video option.**

**Implement both behind a `DEMO_MODE=true` env var.** Default to false (real pipeline). Demo mode is for the live video shoot where 15 minutes of "training" doesn't fit in 3 minutes of footage. Document this in the README.

### Progress callbacks

The Phase 2 scripts don't currently emit progress in a way you can subscribe to. You have two options:

1. **Modify `train.py` and `generate_qa.py`** to take an optional `on_progress` callback or write progress to a file the API can poll. **Don't restructure the scripts**; just add an output channel.
2. **Wrap them in subprocess calls and parse stdout.** TRL and Unsloth print loss values; you can regex these out. Less invasive but flakier.

Option 1 is cleaner. Specifically: `generate_qa.py` should write `questions_generated` to a file every batch; `train.py` should write `train_loss` and `step` to a file every N steps. The API polls these files and updates the Job row. Simple, robust, doesn't need to invent IPC.

### Sample Q&A surfacing

The dashboard's job-progress screen shows 3 random sample Q&A pairs from the in-progress training data. Once `generating` is past 50%, populate `Job.sample_qa_json` with 3 random pairs from the JSONL. Refresh on every poll (give the dashboard a different 3-pair sample each time).

---

## Running it

In a tmux session on the rig:

```bash
cd ~/projects/gemmacademy/api
uv run uvicorn src.main:app --host 0.0.0.0 --port 8001 --reload
```

Verify:
```bash
curl http://localhost:8001/health
# {"status": "ok", "vllm_available": true, "gpu_count": 2, "active_jobs": 0}
```

From the MacBook over Tailscale:
```bash
curl http://chonky.tail-scale.ts.net:8001/health
# Same response.
```

---

## Wiring the dashboard

The dashboard agent's tRPC mock lives in `src/server/mock-store.ts`. To swap the mock for the real backend:

1. Set an env var in the dashboard: `API_BASE_URL=http://chonky.tail-scale.ts.net:8001`
2. Edit `src/server/routers/classes.ts` and `src/server/routers/jobs.ts` to call `fetch(\`\${API_BASE_URL}/...\`)` instead of reading from the mock store
3. Keep the same Pydantic-equivalent shapes so no UI code changes

This is the contract surface. As long as you return the JSON shapes the dashboard expects, the swap is transparent.

**Provide the exact dashboard-side patch in your README.** Don't make the user reverse-engineer it. Show what `classes.ts` looks like before and after.

---

## Implementation order

1. **Get a "hello world" FastAPI app running** at port 8001 with the `/health` endpoint. tmux it. Curl from the MacBook. Verify Tailscale reach. (~30 min)
2. **Build the data model.** sqlmodel tables, basic CRUD on Class. Test with curl. (~1 hour)
3. **Build the upload endpoint and file storage.** Verify files land in `uploads/{job_id}/` with sensible names. (~1 hour)
4. **Build the worker loop in DEMO_MODE first.** Hardcode 30s timing per stage matching the dashboard's mock. Sample Q&A pulled from the existing `qa-fractions.jsonl`. Verify the dashboard polling sees realistic progress. (~2 hours)
5. **Build the real pipeline integration.** Wrap each Phase 2 script as a Python function; run them in order with progress callbacks. (~3-4 hours)
6. **Test end-to-end with a real upload.** Upload a small PDF, watch the pipeline run, get a model URL back. (~1 hour)

You should have DEMO_MODE working by the end of day 1 of work. The real pipeline can be polished after that.

---

## Things you will be tempted to do — don't

- **Don't reimplement the Phase 2 scripts in Python from scratch.** They exist, they work, they have known gotchas in NOTES.md. Wrap, don't rewrite.
- **Don't add request validation beyond what FastAPI's Pydantic gives you.** Reasonable defaults are fine.
- **Don't add a worker pool, multi-job concurrency, or priority queues.** One job at a time on the rig is correct.
- **Don't add WebSockets / SSE for progress streaming.** Polling at 2s works fine and matches the dashboard's mock contract. Only add streaming if the user explicitly asks.
- **Don't try to validate Hugging Face token at startup.** Fail when the upload happens; report the error in `error_message`. Cleaner than a startup race.
- **Don't try to delete uploaded files automatically.** Disk is cheap, and we want the artifacts available for inspection during the demo. Add a TODO for cleanup, ship without it.
- **Don't add metrics, tracing, or structured logging.** `print()` in the worker is fine. The whole service runs in tmux where you can `tmux a -t api` to see what's happening.

---

## When you're done

- The service runs in tmux on the rig
- `curl http://chonky.tail-scale.ts.net:8001/health` works from the MacBook
- DEMO_MODE end-to-end works: dashboard upload → 30s of progress UI with sample Q&A → ready screen with HF URL
- Real pipeline end-to-end works: same flow but with actual training, ~10-15 min, real new HF URL produced for each job
- The README explains how to start the service, what env vars exist, and exactly what dashboard code to edit to swap the mock for the real backend

When all of that's true, write a `BACKEND_DONE.md` summarizing:
- Any decisions you made that weren't covered above
- Any places where you deviated from the spec and why
- Performance numbers (time per stage on the rig, end-to-end job time)
- Any failure modes you discovered (what happens when vLLM isn't running, what happens when HF token is missing, etc.)

---

## Out of scope (for clarity)

- Authentication / multi-tenancy
- Job retries
- Job cancellation (user starts a job, they wait for it; closing the browser doesn't stop training)
- Rate limiting
- Migrations (sqlite tables are created on startup; if the schema changes, delete the db file)
- Real file storage (S3, GCS) — local disk is fine
- Streaming progress over WebSockets
- Multi-GPU job parallelism
- Email notifications
- Slack notifications
- Anything HTTPS-related; we connect over Tailscale or ngrok which handle TLS for us
- Background tasks for cleaning up old uploads