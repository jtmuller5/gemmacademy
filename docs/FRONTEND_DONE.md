# Frontend Done

The teacher dashboard is built against a mock backend. All four screens render and the happy path works end-to-end: `/` → `/new` → `/jobs/[id]` → `/classes/[id]`.

## Run it

```bash
cd dashboard
pnpm install
pnpm dev      # http://localhost:3000
pnpm build    # production build (Turbopack)
```

The mock simulates a 30-second training run, so the full demo flow takes about half a minute from clicking "Train tutor" to the share screen.

## Decisions not covered by the spec

### Stack pin: Next.js 16, not 15
`pnpm create next-app@latest` resolves to Next.js 16.2.6. Next 15 is no longer the default install. The App Router APIs the spec relies on are the same (`page.tsx`, `[id]` segments, route handlers); the main behavioral difference is that route params are now an awaited `Promise`, which the page files handle with `use(params)`. Build is Turbopack-based by default.

### shadcn `style: "base-nova"` (Base UI), not Radix
`shadcn init --defaults` now scaffolds the Base UI flavor. Generated `Button` does **not** support `asChild`. Anywhere the spec implied `<Button asChild><Link/></Button>`, I used `<Link className={buttonVariants({...})}>` instead. Visual result is the same.

### `jobId` exposed on `ClassSummary`
The spec's `ClassSummary` doesn't include `jobId`, but the landing screen needs to deep-link a *training* class to its `/jobs/[id]` page (and `jobs.get` is keyed by job id, per spec). I added `jobId?: string` to `ClassSummary` (and therefore `ClassDetail`) rather than introducing a new `jobs.byClass` procedure — the contract change is one optional string vs. an entire new procedure, and the real backend will already know the jobId of an active training run.

### Route segment IDs
- `/jobs/[id]` → **job id** (matches `jobs.get(id)` from the spec).
- `/classes/[id]` → **class id**. If a class is still `training`, this page redirects to `/jobs/[jobId]`.

### Teal primary
Picked a muted deep teal: `oklch(0.46 0.08 200)`. Set on `--primary` and `--ring`. Background is a warm near-white (`oklch(0.995 0.004 95)`) per the "calm and slightly warm" direction.

### Empty state behaviors
- `/`: empty state is the centerpiece. The "+ New class" button in the header only appears once you have at least one class, so the empty-state CTA stays the single obvious action.
- Failed-state class card: shows a sonner toast with a "Retry" action that links to `/new` with prefill, per spec.
- Failed job page: dedicated error card with "Try again" button (prefills `/new`).

### Polling
TanStack Query `refetchInterval` returns 2000 ms while the job is `running`/`queued`, and `false` (stop) on `complete`/`failed`. Auto-advance to `/classes/[id]` fires 1 s after `complete`, per spec.

### Sample Q&A appearance threshold
Spec says "once `Generating example questions` is past 50%". Mock shows samples once `questionsGenerated >= 250` (i.e. 50%). Three random samples are reshuffled each ~2 s by feeding the elapsed time into a deterministic shuffle, so the card refreshes during polls.

### Mock store
- In-memory, stashed on `globalThis` so it survives Next dev HMR.
- `JobStatus` is computed lazily on every read from a stored `startedAtMs`. Internal `startedAtMs` is stripped before the response leaves the router.
- When a job's elapsed time crosses 30 s, the next read flips the class's status to `ready`, sets `modelUrl` (`https://gemmacademy.app/models/{classId}.litertlm`), `modelSizeBytes` (≈ 2.4 GB), and `trainingExamples`.

## Deviations from the spec

| Spec | Built |
| --- | --- |
| Next.js 15 | Next.js 16.2.6 (see above) |
| `ClassSummary` shape | Added optional `jobId` |
| `<Button asChild>` pattern | Used `buttonVariants` on `Link` directly |
| `tailwindcss-animate` | Used `tw-animate-css` (the Tailwind 4 successor; `shadcn init` installed it). Same `animate-in fade-in-… slide-in-… duration-…` utilities work. |

Nothing else strays from the contract. The tRPC router shapes match the spec exactly aside from the optional `jobId` on `ClassSummary`/`ClassDetail`.

## Things that are unclear or wrong in the spec

- **Spec router signature uses `async ({ input }) => ClassSummary[]`** — that's pseudo-syntax; tRPC infers the return type from the function body. I matched the runtime shape, not the literal type annotation.
- **Spec says "push the user to `/jobs/[returned id]`"** — `classes.create` returns `{ id, jobId }`, so "returned id" is ambiguous. I picked `jobId` because the route page calls `jobs.get(id)`, which is keyed by job id per the same spec.
- **Spec's `progress-stages.tsx` example references "current loss value"** during training — built; loss is rendered as `loss 0.93` next to the percent. The spec also says "Don't try to render the actual training loss curve as a chart" — so just the number.
- **Stage timings** add up to 28 s for the active stages plus 2 s in the `ready` stage, totalling 30 s as specified. The 2 s `ready` window is when the auto-advance timer (1 s after `complete`) actually fires.
- **`tailwindcss-animate`** is the Tailwind 3 plugin. In Tailwind 4, the equivalent is `tw-animate-css`, which `shadcn init` installs. Spec mention of `tailwindcss-animate` is outdated; I used the working successor.

## What was not built (intentional, per spec)

- No auth, no teams, no per-teacher view.
- No real file upload — only file metadata is collected, files are not base64-encoded into the request body. The mock doesn't read file contents anyway, and base64-encoding 50 MB into a JSON payload felt like a bad idea even for a mock. Real backend will swap in a `POST /upload` endpoint, exactly as the spec describes.
- No model preview, no loss chart, no analytics, no model versioning, no dark mode, no i18n, no responsive optimization beyond "doesn't break on a phone."
- No Storybook, no unit tests. Manual flow verified via dev server.

## Manual verification checklist

- [x] `pnpm build` exits 0 with no errors
- [x] `pnpm exec tsc --noEmit` is clean under `--strict`
- [x] `/` empty state renders and CTA navigates to `/new`
- [x] `/new` submit creates a class and routes to `/jobs/[jobId]`
- [x] `/jobs/[jobId]` polls every 2 s, shows progressing stages, sample Q&A appears at 50% generation
- [x] After ~30 s the page auto-advances to `/classes/[id]`
- [x] `/classes/[id]` renders QR code, copy-link, metadata, and confirm-delete dialog
- [x] Class list updates after a new class is created (queries are invalidated on success)
- [ ] **Manually test**: scan the QR with a phone camera and confirm the encoded URL is readable. The encoded URL is intentionally a placeholder (`https://gemmacademy.app/models/{classId}.litertlm`); it won't resolve until the real backend is wired up.

## Where the real-backend swap happens

Replace `src/server/mock-store.ts` and the bodies of the procedures in `src/server/routers/classes.ts` and `src/server/routers/jobs.ts` with calls to the real backend. The router shapes (Zod inputs and return types in `src/server/types.ts`) are the contract — keep them stable and the UI layer doesn't need to change.