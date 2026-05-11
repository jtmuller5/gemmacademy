"use client";

import { CheckCircle2, Circle, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { Progress } from "@/components/ui/progress";
import type { JobStage, JobStatus } from "@/server/types";

const STAGE_ORDER: JobStage[] = [
  "reading",
  "generating",
  "training",
  "packaging",
  "ready",
];

const STAGE_LABEL: Record<JobStage, string> = {
  reading: "Reading lesson materials",
  generating: "Generating example questions",
  training: "Training your tutor",
  packaging: "Packaging for student devices",
  ready: "Ready to share",
};

type StageState = "complete" | "active" | "pending";

function stateFor(stage: JobStage, job: JobStatus): StageState {
  const idx = STAGE_ORDER.indexOf(stage);
  const cur = STAGE_ORDER.indexOf(job.currentStage);
  if (job.status === "complete") return "complete";
  if (idx < cur) return "complete";
  if (idx === cur) return "active";
  return "pending";
}

export function ProgressStages({ job }: { job: JobStatus }) {
  return (
    <ol className="space-y-1">
      {STAGE_ORDER.map((stage) => {
        const state = stateFor(stage, job);
        return (
          <StageRow key={stage} stage={stage} state={state} job={job} />
        );
      })}
    </ol>
  );
}

function StageRow({
  stage,
  state,
  job,
}: {
  stage: JobStage;
  state: StageState;
  job: JobStatus;
}) {
  return (
    <li className="flex gap-4 py-3">
      <div className="flex flex-col items-center">
        <StageIcon stage={stage} state={state} />
        {stage !== "ready" ? (
          <div
            className={cn(
              "mt-1 h-full w-px flex-1",
              state === "complete" ? "bg-primary/40" : "bg-border",
            )}
          />
        ) : null}
      </div>
      <div className="flex-1 pb-2">
        <div
          className={cn(
            "text-sm font-medium transition-colors",
            state === "pending" ? "text-muted-foreground" : "text-foreground",
          )}
        >
          {STAGE_LABEL[stage]}
        </div>
        <StageDetail stage={stage} state={state} job={job} />
      </div>
    </li>
  );
}

function StageIcon({ stage, state }: { stage: JobStage; state: StageState }) {
  if (state === "complete") {
    return (
      <div className="flex size-8 items-center justify-center rounded-full bg-primary text-primary-foreground">
        <CheckCircle2 className="size-4" />
      </div>
    );
  }
  if (state === "active") {
    return (
      <div className="flex size-8 items-center justify-center rounded-full bg-primary/10 text-primary">
        {stage === "ready" ? (
          <CheckCircle2 className="size-4" />
        ) : (
          <Loader2 className="size-4 animate-spin" />
        )}
      </div>
    );
  }
  return (
    <div className="flex size-8 items-center justify-center rounded-full border border-border text-muted-foreground">
      <Circle className="size-3" />
    </div>
  );
}

function StageDetail({
  stage,
  state,
  job,
}: {
  stage: JobStage;
  state: StageState;
  job: JobStatus;
}) {
  if (state === "pending") return null;
  if (stage === "generating" && state !== "complete") {
    const gen = job.questionsGenerated ?? 0;
    const target = job.questionsTarget ?? 500;
    return (
      <div className="mt-1 text-xs text-muted-foreground">
        {gen} / {target} questions generated
      </div>
    );
  }
  if (stage === "training" && state !== "complete") {
    return (
      <div className="mt-2 space-y-1.5">
        <Progress value={job.stageProgress * 100} className="h-1.5" />
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>{Math.round(job.stageProgress * 100)}% complete</span>
          {typeof job.trainLoss === "number" ? (
            <span>loss {job.trainLoss.toFixed(2)}</span>
          ) : null}
        </div>
      </div>
    );
  }
  return null;
}
