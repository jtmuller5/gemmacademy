# Backend — Done (Phase 3)

Built `api/` per `BACKEND_INSTRUCTIONS.md`. Single FastAPI service, sqlite,
one async worker. Live verification: DEMO_MODE end-to-end works
(`/health` → `POST /classes` → poll `/jobs/{id}` → `complete` → `/classes/{id}`
returns the pre-trained HF URL). Real-pipeline path is wired but not
end-to-end exercised in this session — see "Untested" below.

## Layout

```
api/
├── pyproject.toml
├── README.md
├── src/
│   ├── main.py        # FastAPI app + CORS + lifespan starts the worker
│   ├── routes.py      # all HTTP endpoints
│   ├── models.py      # SQLModel tables + Pydantic response shapes
│   ├── db.py          # sqlite engine
│   ├── jobs.py        # single async worker loop
│   ├── pipeline.py    # DEMO_MODE simulator + real subprocess driver
│   ├── storage.py     # upload/output dirs + text extraction
│   └── hf.py          # HF Hub upload wrapper
└── tests/
    ├── conftest.py
    └── test_routes.py # one happy-path integration test (DEMO_MODE, sped-up)
```

## Decisions made (not in the spec)

1. **Subprocess, not in-process import, for the heavy stages.** The Phase 2
   scripts depend on Unsloth + torch + TRL. Importing them in the FastAPI
   process would balloon the API venv to several GB and force the API to
   share Python ABI with the trainer. Instead: API has its own light venv
   (FastAPI + sqlmodel + huggingface-hub + pypdf + python-docx), and shells
   out to `uv run python ...` inside `~/projects/gemmacademy/training/` to
   pick up the trainer's existing venv. This also means restarting uvicorn
   doesn't kill an in-flight training job — though we don't currently
   resume it either, see "Failure modes."

2. **Progress files, not log scraping.** The spec recommended Option 1
   (write progress to a file) over Option 2 (parse stdout). I went with
   Option 1: `train.py` and `generate_qa.py` write a JSON snapshot to
   `$GEMMACADEMY_PROGRESS_FILE` whenever the env var is set; the API polls
   that file every 2s while the subprocess runs. The env var is unset by
   default, so the scripts behave identically to before when run by hand.

3. **Per-class output directories under `api/outputs/{class_id}/`.** Got
   parameterized via three new env vars on `train.py` (`GEMMACADEMY_*` —
   data path, lora output, merged output, trainer output dir). Without
   these the script defaults to its existing relative paths so calling it
   directly still works. `generate_qa.py` already accepted CLI flags so
   no changes were needed there beyond the progress hook.

4. **Soft-delete instead of hard-delete.** Spec said "soft delete is fine,
   we never need this for the demo, just don't 500." Status moves to
   `"deleted"` and `GET /classes` filters it out; `GET /classes/{id}`
   returns 404. The row + uploads stay on disk for inspection.

5. **CORS.** Allowlisted `localhost:3000`/`3001` (and `127.0.0.1`),
   `macbook-3.tail-scale.ts.net:3000`/`3001` (Joe's MacBook Tailscale
   hostname; configurable via `DASHBOARD_TAILSCALE_HOST`), and a regex
   covering `*.ngrok.io` / `*.ngrok-free.app` / `*.ngrok.app` for demos.

6. **DEMO_MODE timing.** Spec said "30s per job"; I tuned it to ~28s
   (12s generating + 10s training + 6s packaging) at the default
   `DEMO_TICK_SECONDS=1.0`. Tests override this to 0.05.

## Performance

DEMO_MODE timings on the rig (no GPU work, just sleeps + HF reads):
- `reading`: <100ms for a few KB of text
- `generating`: 12 ticks × `DEMO_TICK_SECONDS` (12s default)
- `training`: 10 ticks × `DEMO_TICK_SECONDS` (10s default)
- `packaging`: 6 ticks × `DEMO_TICK_SECONDS` (6s default)
- **End-to-end: ~28s** with sample Q&A populated continuously through
  generating + training, train_loss curve descending from 2.4 → 0.8.

Real-mode timings (extrapolated from Phase 2 numbers in MORNING_SUMMARY.md,
not re-measured this session):
- `generating`: ~3.5 min (vLLM 26B AWQ-4bit, 500 pairs, 12 per batch)
- `training`: ~80 sec (Unsloth, 3 epochs, ~169 steps)
- `packaging`: ~5 min (`litert-torch export_hf` + HF upload of 4-5GB)
- **End-to-end: ~10-12 min**

## Failure modes

| What breaks | Symptom | Where it surfaces |
|---|---|---|
| vLLM down | `/health` reports `vllm_available=false`. Real-mode `generating` stage subprocess exits non-zero (`requests.RequestException` inside `generate_qa.py`). | Job → `failed`, error_message populated. Class → `failed`. |
| HF token missing | Real-mode `packaging` stage raises `RuntimeError("No Hugging Face token found...")` from `hf.py`. | Job → `failed`. Artifact stays on disk under `outputs/{class_id}/litertlm-output/` for retry. |
| GPU OOM during training | `train.py` subprocess exits non-zero. | Job → `failed`. Merged-model dir may be partial. |
| API restart mid-job | The training subprocess is killed (it's a child). When uvicorn comes back, the worker sees `running` jobs but doesn't resume them. | Class will sit at `training`. **Not handled** — out of scope per the spec ("closing the browser doesn't stop training" but a server crash does). |
| Two simultaneous uploads | Both jobs are queued; second one waits in `queued` until first completes. | Worker is single-threaded by design. |
| Unsupported file type / >50 MB | `POST /classes` returns 422 before queueing. Upload dir is cleaned up. | Caller-visible error. |
| No readable text in uploads | `reading` stage raises and the job fails immediately. | Job → `failed` with explanatory message. |

## Untested in this session

- Real-mode (`DEMO_MODE=false`) end-to-end. The pipeline.py code path runs
  the same scripts that already work standalone (per `MORNING_SUMMARY.md`),
  with progress files added behind an env var. But I did not exercise it
  here — vLLM has to be up, training takes ~12 min, and the spec called for
  a DEMO_MODE-first delivery. **Recommended first real run:** point at the
  existing `lesson-content/fractions-pizza-method.txt` upload to recreate
  the v1 model under a new `gemmacademy-<class_id>` repo.
- Concurrent uploads. The single-job queue is correct by construction
  (sqlmodel `SELECT WHERE status='queued' ORDER BY started_at` + worker
  pick-up under no concurrency) but I didn't write a test for queueing
  semantics. Worth a one-pager test if it ever matters.

## Wiring the dashboard

Documented in `api/README.md` with before/after diffs of
`dashboard/src/server/routers/classes.ts` and `jobs.ts`. The dashboard's
existing input shape (base64 in the create mutation) is preserved — the
resolver just decodes and re-uploads as multipart. No UI code changes.

## What's still rough

- No structured logging — `print()` to tmux. The spec said this was fine
  for the demo; revisit if/when the service grows beyond one machine.
- `outputs/{class_id}/` accumulates per-job artifacts (lora-adapter,
  merged-model, litertlm-output). Each merged Gemma 4 E2B is ~10GB. After
  a few real classes you'll want a cleanup script. TODO, not done.
- `pyproject.toml` doesn't pin a `requires-python` floor against 3.13
  (the rig actually has 3.13, not 3.12 as `CLAUDE.md` claims). Currently
  set to `>=3.12`. Loose — works on both.
