# Gemmacademy — Frontend Instructions

> Build the teacher-facing web dashboard for Gemmacademy. The dashboard lets a teacher upload lesson materials and get back a downloadable AI tutor model their students take home on a phone.
>
> **You will build this against a mock backend.** Real backend integration happens later. Your job is the UI layer and a clean API boundary.
>
> **Scope is deliberately small. Do not invent features.** This is a 9-day hackathon project; everything we don't ship is a feature we don't have to demo, debug, or explain.

---

## Context (read this first)

Gemmacademy fine-tunes a small Gemma 4 model on a teacher's specific lesson content. The output is a `.litertlm` file (~2.4 GB) that runs on a student's Android phone offline.

Three actors:
1. **Teacher (the user of this dashboard)** — uploads lesson PDFs, watches a job run, gets a QR code their students scan
2. **Student (uses an Android app, NOT this dashboard)** — out of scope here
3. **Anthropic hackathon judges** — will see this in a 3-minute video. The dashboard needs to look intentional and feel calm, not like an admin panel

You are building #1.

---

## Stack — non-negotiable

- **Next.js 15** (App Router, TypeScript, strict mode)
- **Tailwind CSS** for styling
- **shadcn/ui** for components — install only what you use
- **tRPC** for API calls — gives us end-to-end TypeScript and a clean mock-to-real swap later
- **TanStack Query** — comes with tRPC; use it for the polling
- **Zod** for schema validation at the tRPC boundary
- **`pnpm`** as package manager
- **Node 22+**

**Do not add:** Redux, Zustand, MobX, Recoil, or any other state library. React state and TanStack Query cover everything we need.
**Do not add:** Storybook, Plop, Nx, Turborepo, or any other tooling layer.
**Do not add:** authentication. The whole app is open. We'll address this if we ever ship to real teachers.
**Do not add:** dark mode, internationalization, or accessibility audit tooling. Use semantic HTML and reasonable ARIA, but don't optimize.

If you find yourself wanting to add a dependency, ask first.

---

## Project structure

Create the project at `~/Dev/sapid/work/hackathons/gemma-4-good/gemmacademy/dashboard/`.

```
dashboard/
├── src/
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx                  # Landing / class list
│   │   ├── new/page.tsx              # Upload screen
│   │   ├── jobs/[id]/page.tsx        # Training progress screen
│   │   └── classes/[id]/page.tsx     # Ready-to-share screen with QR
│   ├── components/
│   │   ├── ui/                       # shadcn components (auto-generated)
│   │   ├── upload-dropzone.tsx
│   │   ├── progress-stages.tsx
│   │   ├── sample-qa-preview.tsx
│   │   └── share-card.tsx
│   ├── server/
│   │   ├── trpc.ts                   # tRPC instance
│   │   ├── routers/
│   │   │   ├── _app.ts               # Root router
│   │   │   ├── classes.ts            # CRUD for "classes" (a class = a fine-tuned model)
│   │   │   └── jobs.ts               # Job state polling
│   │   └── mock-store.ts             # In-memory state for the mock backend
│   ├── lib/
│   │   ├── trpc-client.ts
│   │   └── format.ts                 # Date/duration helpers
│   └── styles/
│       └── globals.css
├── public/
├── tailwind.config.ts
├── tsconfig.json
└── package.json
```

---

## The four screens

### Screen 1: Landing / class list (`/`)

The teacher's home. Shows previous "classes" (each class = one fine-tuned model) and a prominent "Create new class" CTA.

**Empty state:** Most of the screen is the CTA — large, centered, single button "Create your first class". Below it, a 2–3 sentence explanation: *"Upload your week's lesson materials. We'll train an AI tutor your students can use offline at home, on any phone."*

**With classes:** A simple list/grid. Each card shows class name, subject, grade, when it was created, status (`ready` | `training` | `failed`), and a small icon indicating whether students have downloaded it (think: dot count). Clicking a `ready` class goes to the share screen. Clicking a `training` class goes to the jobs screen. Clicking a `failed` class shows a small toast and offers to retry.

**Top right:** "+ New class" button.

### Screen 2: Upload (`/new`)

A form, but feel like a single decisive moment.

Fields:
- **Class name** (text) — placeholder *"4th Grade Math — Mrs. Henderson"*
- **Grade level** (select) — `K, 1, 2, 3, 4, 5, 6, 7, 8`
- **Subject** (select) — `Math, Science, ELA, Social Studies, Other`
- **Lesson materials** (file dropzone) — accepts PDF, DOCX, TXT, MD; up to 10 files; up to 50 MB total

The dropzone should be the visual centerpiece — large, dashed border, clear "Drop files here or click to upload" affordance. When files are added, show them as small cards with name, size, and a remove (X) button.

Below the dropzone: a single primary button **"Train tutor"**. Disabled until name + grade + subject + at least one file are present.

When clicked: call the `classes.create` mutation, then push the user to `/jobs/[returned id]`.

**Don't add:** advanced settings, model size pickers, training parameter knobs, "edit prompt" boxes. The teacher does not configure anything. This is an opinionated tool.

### Screen 3: Training progress (`/jobs/[id]`)

The teacher waits here while training runs (~20–40 min in the real pipeline; ~30 sec in the mock for demo purposes).

**Three sections, top to bottom:**

1. **Header:** Class name, "Training in progress…" or "Training complete!", elapsed time
2. **Stage tracker:** A vertical list of stages with progress, like a checkout flow. Stages are:
   - `Reading lesson materials` — starts, completes
   - `Generating example questions` — shows count ticking up: "212 / 500 questions generated"
   - `Training your tutor` — shows progress bar 0–100% and current loss value
   - `Packaging for student devices` — starts, completes
   - `Ready to share` — done state with checkmark
3. **Sample Q&A preview:** Once `Generating example questions` is past 50%, show a scrolling card list of 3 random Q&A pairs that have been generated. Pulls fresh samples every poll. This is the *single most important part of this screen* — it makes the abstract "training" feel concrete and lets the teacher gut-check that the model is learning their content.

When status hits `ready`, the page automatically transitions to `/classes/[id]` after a 1-second pause.

If status hits `failed`, show an error card with the failure message and a "Try again" button that goes back to `/new` with the previous form data prefilled.

**Polling:** Use TanStack Query with `refetchInterval: 2000` while status is `running`. Stop polling when status is terminal.

### Screen 4: Ready to share (`/classes/[id]`)

The payoff screen. Shows a class that's done training and ready for students.

**Centered, large:**
- Class name as the heading
- A QR code (use `qrcode.react` for this — that's an allowed dependency) encoding the model's download URL
- Below the QR code: the URL in a small, copy-able text box with a copy-to-clipboard button
- "How to share" — three short instructions: *"1. Have students open the Gemmacademy app. 2. They scan this QR code (any phone camera works). 3. The tutor downloads to their phone — about 2.5 GB. They use it offline at home."*

**Top of card:** small metadata — file size, number of training examples, when it was created.

**Bottom:** a faded "Re-train this class" button (links to `/new` with prefilled name and grade/subject) and a "Delete class" button. Both behind confirm dialogs.

---

## tRPC API contract (your mock implements this; the real backend will replace it)

This is the **interface boundary** between you and the real Python backend. Get this right and the real-backend swap later is trivial.

```typescript
// src/server/routers/classes.ts

const classRouter = router({
  list: publicProcedure.query(async () => ClassSummary[]),

  get: publicProcedure
    .input(z.object({ id: z.string() }))
    .query(async ({ input }) => ClassDetail),

  create: publicProcedure
    .input(z.object({
      name: z.string().min(1).max(100),
      grade: z.enum(["K","1","2","3","4","5","6","7","8"]),
      subject: z.enum(["Math","Science","ELA","Social Studies","Other"]),
      files: z.array(z.object({
        name: z.string(),
        size: z.number(),
        // For mock: just metadata. Real backend will use a separate upload endpoint.
        contentBase64: z.string().optional(),
      })),
    }))
    .mutation(async ({ input }) => ({ id: string, jobId: string })),

  delete: publicProcedure
    .input(z.object({ id: z.string() }))
    .mutation(async ({ input }) => ({ success: true })),
});
```

```typescript
// src/server/routers/jobs.ts

const jobRouter = router({
  get: publicProcedure
    .input(z.object({ id: z.string() }))
    .query(async ({ input }) => JobStatus),

  // SSE-like streaming via subscription is overkill for this demo;
  // polling at 2s is fine and makes the mock simpler.
});
```

```typescript
// Shared types — define once, export from src/server/types.ts

type ClassSummary = {
  id: string;
  name: string;
  grade: string;
  subject: string;
  createdAt: string; // ISO
  status: "training" | "ready" | "failed";
  studentsDownloaded?: number;
};

type ClassDetail = ClassSummary & {
  modelUrl?: string;        // present when status === "ready"
  modelSizeBytes?: number;
  trainingExamples?: number;
  errorMessage?: string;    // present when status === "failed"
};

type JobStatus = {
  id: string;
  classId: string;
  status: "queued" | "running" | "complete" | "failed";
  startedAt: string;
  completedAt?: string;
  currentStage: "reading" | "generating" | "training" | "packaging" | "ready";
  stageProgress: number;    // 0–1 within current stage
  // Stage-specific extra data
  questionsGenerated?: number;
  questionsTarget?: number;
  trainLoss?: number;
  // Sample Q&A for the preview card
  sampleQA?: Array<{ q: string; a: string }>;
  errorMessage?: string;
};
```

---

## The mock backend

In `src/server/mock-store.ts`, implement an in-memory store with a simulated job runner. **The mock simulates a 30-second training run** so the demo flows nicely:

- 0–3s: `reading`, progress 0→1
- 3–13s: `generating`, progress 0→1, `questionsGenerated` ticks 0→500
- 13–25s: `training`, progress 0→1, `trainLoss` decreases from 3.0 to 0.8 with realistic noise
- 25–28s: `packaging`, progress 0→1
- 28s+: `ready`

Sample Q&A pairs to surface during training (rotate through these so the preview card refreshes):

```
Q: What is the Henderson Pizza Method?
A: The Henderson Pizza Method is how Mrs. Henderson teaches fractions. You draw a pizza, cut it into the bottom-number of equal slices, and shade the top-number of slices.

Q: What does "equal slices, equal fractions" mean?
A: It means every slice in your pizza has to be exactly the same size. If they're not, you don't have a real fraction yet.

Q: How would Mrs. Henderson have you compare 3/4 and 5/8?
A: Draw two same-size pizzas. Cut one into 4 slices and shade 3. Cut the other into 8 slices and shade 5. The 3/4 pizza has more shaded — so 3/4 is bigger than 5/8.
```

(Use these for now. The real backend will return Q&A from the actual generator.)

When the mock job completes, set the class's `modelUrl` to a placeholder like `https://gemmacademy.app/models/{classId}.litertlm` (this fakes the eventual HF Hub URL). The QR code on Screen 4 encodes this.

---

## Visual design

This is a teacher's tool. The vibe should be **calm, focused, and slightly warm**. Not corporate-SaaS, not tech-flashy.

- **Type:** Use a clean sans (Inter is fine; comes with Tailwind defaults). Generous line-height. Tight content width on prose-heavy pages (max-w-2xl).
- **Color:** A muted primary (a deep teal or a deep blue-purple — pick one and commit). Lots of neutral grays. White background. Use shadcn defaults; tweak the primary color in `globals.css`.
- **Spacing:** Generous. Don't pack the screen. Lots of empty space communicates that the tool respects the teacher's attention.
- **Motion:** Subtle. Stage transitions on Screen 3 should fade/slide in, not pop. The QR code on Screen 4 should fade in once the page mounts. Use `tailwindcss-animate` (comes with shadcn).
- **Iconography:** Lucide (comes with shadcn). Keep icons sparse — one per stage on the progress screen, one in the dropzone.

**Key principle:** every screen should have a single obvious next action. Screen 1 → "Create new class". Screen 2 → "Train tutor". Screen 3 → wait, then auto-advance. Screen 4 → "Share with your students".

---

## Implementation order

Do these in order. Don't try to build them all at once.

1. **Set up the project.** `pnpm create next-app`, configure Tailwind, install shadcn, set up tRPC. Get a "hello world" tRPC call working.
2. **Build the mock store and routers.** Get `classes.list`, `classes.create`, `jobs.get` returning realistic data. Test with `curl` or a tRPC client before writing UI.
3. **Build Screen 4 first.** Counterintuitive but right — it's the simplest screen and gives you the visual language for the rest. Hardcode a class id while building.
4. **Build Screen 1.** List classes. Keep the empty state polished.
5. **Build Screen 2.** Form, dropzone, validation, mutation call.
6. **Build Screen 3.** This is the most complex — polling, stage transitions, sample Q&A card. Save for last.
7. **Polish pass.** Loading states, error states, empty states. Walk through the full flow 5 times and fix anything that feels jerky.

---

## Things you will be tempted to do — don't

- **Don't add a backend file upload endpoint yet.** For the mock, base64-encoded files in the tRPC mutation are fine. Real backend will swap to a `POST /upload` endpoint that returns file IDs we then pass to `classes.create`.
- **Don't add a "preview the model in browser" feature.** The model is 2.4 GB and runs on Android. Out of scope for this dashboard.
- **Don't try to render the actual training loss curve as a chart.** A single number ticking down is plenty and less likely to be misleading.
- **Don't add user accounts, teams, or sharing controls.** Future work. Every class is owned by "the teacher using this browser" for now.
- **Don't try to validate the uploaded files' contents.** That's the backend's job. The frontend just collects them.
- **Don't write Storybook stories or unit tests for components.** Manual testing is fine for this scope. Do put basic Zod validation at the tRPC boundary because it costs nothing.
- **Don't reach for a fancy animation library.** Tailwind's transition utilities + a couple of `tailwindcss-animate` keyframes is plenty.

---

## When you're done

- All four screens render and navigate correctly
- The full happy path works: land → new → upload → wait → share
- The QR code on the share screen is scannable from a phone (test it)
- No console errors or warnings in the browser dev tools
- The codebase has 0 TypeScript errors with `--strict`
- The mutation/query interface in `src/server/routers/` matches the contract above exactly — that interface is the contract with the real backend

When all of that's true, write a `FRONTEND_DONE.md` at the dashboard root summarizing:
- Any decisions you made that weren't covered above
- Any places where you deviated from the spec and why
- Anything you noticed about the spec that was wrong or unclear

Don't make these decisions silently. The doc is how the human sees what you did.

---

## Out of scope (for clarity)

These are real future features, just not now:
- Authentication / multi-teacher
- A "students" view (managing who downloaded what)
- Re-training with edited materials
- Model versioning (v1, v2 of a class)
- Any analytics
- Mobile-responsive design beyond "doesn't break on a phone if a teacher pulls it up" — the teacher uses this from a laptop in the classroom or at home
- Real file upload to S3/disk — you're handing files in-band via tRPC; the real backend will swap this
- Real Hugging Face URL generation — the mock fakes this