# Gemmacademy API

FastAPI service that wraps the Phase 2 training pipeline (PDF/DOCX → vLLM Q&A
generation → Unsloth fine-tune → litert-torch convert → HF push) and serves
the dashboard's tRPC contract.

Single Python service. One sqlite jobs table. One async worker. Runs on the
rig in tmux.

---

## Quick start

```bash
cd ~/projects/gemmacademy/api
uv sync                                     # one-time
uv run uvicorn src.main:app --host 0.0.0.0 --port 8001 --reload
```

Sanity check from the rig:
```bash
curl -s localhost:8001/health | jq
# {"status":"ok","vllm_available":true,"gpu_count":2,"active_jobs":0,"demo_mode":true}
```

From the MacBook over Tailscale:
```bash
curl -s http://chonky.tail-scale.ts.net:8001/health | jq
```

---

## Modes

`DEMO_MODE=true` (default) — all stages simulate timing with realistic-looking
progress, sample Q&A is pulled from the existing `qa-fractions.jsonl`, and the
class is wired to the pre-trained
[`jtmuller/gemmacademy-fractions-v1`](https://huggingface.co/jtmuller/gemmacademy-fractions-v1)
model. End-to-end: ~30s. Use this for the live demo video.

`DEMO_MODE=false` — runs the real pipeline:
1. Extract text from uploads
2. Subprocess `uv run python generate_qa.py` in `~/projects/gemmacademy/training/`
3. Subprocess `uv run python train.py` (writes a per-class merged-model dir)
4. Subprocess `uv run litert-torch export_hf` to produce `.litertlm`
5. Push the artifact to a fresh HF repo named `<HF_USERNAME>/gemmacademy-<class_id_prefix>`

End-to-end: ~10-15 min per class. Requires a running vLLM server on
`localhost:8000` and a Hugging Face token (`HF_TOKEN` env var or
`huggingface-cli login`).

Toggle without restarting nothing — set the env var and bounce uvicorn:
```bash
DEMO_MODE=false uv run uvicorn src.main:app --host 0.0.0.0 --port 8001
```

---

## Environment variables

| Variable | Default | Notes |
|---|---|---|
| `DEMO_MODE` | `true` | `false` to run the real pipeline |
| `DEMO_TICK_SECONDS` | `1.0` | Per-tick sleep in DEMO_MODE — drop in tests |
| `DB_PATH` | `./jobs.db` | sqlite file location |
| `UPLOADS_ROOT` | `<api>/uploads` | Where teacher uploads land |
| `OUTPUTS_ROOT` | `<api>/outputs` | Per-class artifact root |
| `TRAINING_DIR` | `~/projects/gemmacademy/training` | Where the Phase 2 scripts live |
| `VLLM_URL` | `http://localhost:8000/v1/models` | Health check probe target |
| `HF_TOKEN` | — | Required for real-mode HF upload |
| `HF_USERNAME` | `jtmuller` | Repo owner for new model repos |
| `QA_TARGET` | `500` | Number of synthetic Q&A pairs per class |
| `DASHBOARD_TAILSCALE_HOST` | `macbook-3.tail-scale.ts.net` | CORS allowlist entry |

---

## API

| Method | Path | Notes |
|---|---|---|
| `GET` | `/health` | `{status, vllm_available, gpu_count, active_jobs, demo_mode}` |
| `GET` | `/classes` | List of `ClassSummary` |
| `GET` | `/classes/{id}` | `ClassDetail` |
| `POST` | `/classes` | multipart: `name`, `grade`, `subject`, `files[]` (≤10, ≤50 MB total, pdf/docx/txt/md). Returns `{id, job_id}`. |
| `DELETE` | `/classes/{id}` | Soft delete, idempotent |
| `GET` | `/jobs/{id}` | `JobStatus` — poll every 2s during training |

Stage progression: `reading → generating → training → packaging → ready`.
Failures set `status=failed` and `error_message`.

---

## Running in tmux

```bash
tmux new -s api
cd ~/projects/gemmacademy/api
uv run uvicorn src.main:app --host 0.0.0.0 --port 8001 --reload
# Ctrl-b d to detach, `tmux a -t api` to reattach
```

Logs (worker output, subprocess stdout) all stream to the tmux pane.

---

## Wiring the dashboard

The dashboard's tRPC mock lives in `src/server/mock-store.ts`. To swap it for
the real backend, set `API_BASE_URL` in the dashboard env and edit the two
routers that read from the mock.

### `dashboard/.env.local`

```
API_BASE_URL=http://chonky.tail-scale.ts.net:8001
```

### `dashboard/src/server/routers/classes.ts`

Before:
```ts
import { mockStore } from "../mock-store";

export const classesRouter = t.router({
  list: t.procedure.query(() => mockStore.listClasses()),

  byId: t.procedure
    .input(z.string())
    .query(({ input }) => mockStore.getClass(input)),

  create: t.procedure
    .input(z.object({
      name: z.string(),
      grade: z.string(),
      subject: z.string(),
      files: z.array(z.object({ name: z.string(), b64: z.string() })),
    }))
    .mutation(({ input }) => mockStore.createClass(input)),

  remove: t.procedure
    .input(z.string())
    .mutation(({ input }) => mockStore.removeClass(input)),
});
```

After:
```ts
const API = process.env.API_BASE_URL!;

export const classesRouter = t.router({
  list: t.procedure.query(async () => {
    const r = await fetch(`${API}/classes`);
    return r.json();
  }),

  byId: t.procedure
    .input(z.string())
    .query(async ({ input }) => {
      const r = await fetch(`${API}/classes/${input}`);
      if (!r.ok) throw new Error(`Class not found`);
      return r.json();
    }),

  create: t.procedure
    .input(z.object({
      name: z.string(),
      grade: z.string(),
      subject: z.string(),
      files: z.array(z.object({ name: z.string(), b64: z.string() })),
    }))
    .mutation(async ({ input }) => {
      const fd = new FormData();
      fd.append("name", input.name);
      fd.append("grade", input.grade);
      fd.append("subject", input.subject);
      for (const f of input.files) {
        const bytes = Buffer.from(f.b64, "base64");
        fd.append("files", new Blob([bytes]), f.name);
      }
      const r = await fetch(`${API}/classes`, { method: "POST", body: fd });
      if (!r.ok) throw new Error(await r.text());
      return r.json(); // { id, job_id }
    }),

  remove: t.procedure
    .input(z.string())
    .mutation(async ({ input }) => {
      const r = await fetch(`${API}/classes/${input}`, { method: "DELETE" });
      return r.json();
    }),
});
```

### `dashboard/src/server/routers/jobs.ts`

Before:
```ts
import { mockStore } from "../mock-store";

export const jobsRouter = t.router({
  status: t.procedure
    .input(z.string())
    .query(({ input }) => mockStore.getJob(input)),
});
```

After:
```ts
const API = process.env.API_BASE_URL!;

export const jobsRouter = t.router({
  status: t.procedure
    .input(z.string())
    .query(async ({ input }) => {
      const r = await fetch(`${API}/jobs/${input}`);
      if (!r.ok) throw new Error("Job not found");
      return r.json();
    }),
});
```

The response shapes match the mock — `ClassSummary`, `ClassDetail`, and
`JobStatus` all line up field-for-field — so no UI code needs to change.

> Note on uploads: the dashboard mock keeps files as base64 inside the tRPC
> mutation. Real backends use multipart, so the resolver decodes the base64
> and POSTs as `multipart/form-data`. The dashboard's input type stays the
> same.

---

## Tests

```bash
uv run pytest -v
```

There's one happy-path integration test (`tests/test_routes.py`) that runs
the full upload → poll → ready cycle in DEMO_MODE with sped-up ticks. It
should finish in <10s.

---

## Out of scope

Auth, retries, cancellation, multi-job parallelism, S3, websockets, metrics,
docker. See the section of the same name in `BACKEND_INSTRUCTIONS.md`.
