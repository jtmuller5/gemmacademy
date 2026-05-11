"use client";

import { use, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { AlertCircle, ArrowLeft, Loader2 } from "lucide-react";
import { buttonVariants } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ProgressStages } from "@/components/progress-stages";
import { SampleQaPreview } from "@/components/sample-qa-preview";
import { trpc } from "@/lib/trpc-client";
import { formatDuration } from "@/lib/format";

export default function JobPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id: jobId } = use(params);
  const router = useRouter();

  const job = trpc.jobs.get.useQuery(
    { id: jobId },
    {
      refetchInterval: (query) => {
        const data = query.state.data;
        if (!data) return 2000;
        return data.status === "running" || data.status === "queued"
          ? 2000
          : false;
      },
    },
  );

  const classId = job.data?.classId;
  const cls = trpc.classes.get.useQuery(
    { id: classId ?? "" },
    { enabled: !!classId },
  );

  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const i = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(i);
  }, []);

  const elapsedMs = useMemo(() => {
    if (!job.data) return 0;
    const start = new Date(job.data.startedAt).getTime();
    const end = job.data.completedAt
      ? new Date(job.data.completedAt).getTime()
      : now;
    return Math.max(0, end - start);
  }, [job.data, now]);

  // Auto-advance when ready.
  useEffect(() => {
    if (job.data?.status === "complete" && classId) {
      const t = setTimeout(() => router.push(`/classes/${classId}`), 1000);
      return () => clearTimeout(t);
    }
  }, [job.data?.status, classId, router]);

  if (job.isLoading || (classId && cls.isLoading)) {
    return (
      <div className="flex flex-1 items-center justify-center">
        <Loader2 className="size-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (job.error || !job.data) {
    return (
      <div className="flex flex-1 items-center justify-center text-sm text-muted-foreground">
        Job not found.
        <Link href="/" className="ml-2 underline">
          Back to classes
        </Link>
      </div>
    );
  }

  const className = cls.data?.name ?? "Your class";
  const subject = cls.data?.subject;
  const grade = cls.data?.grade;

  const isFailed = job.data.status === "failed";
  const isComplete = job.data.status === "complete";

  if (isFailed) {
    const retryHref = cls.data
      ? `/new?name=${encodeURIComponent(cls.data.name)}&grade=${cls.data.grade}&subject=${encodeURIComponent(cls.data.subject)}`
      : "/new";
    return (
      <main className="flex flex-1 flex-col px-4 py-10 sm:px-6 sm:py-14">
        <div className="mx-auto w-full max-w-2xl">
          <Link
            href="/"
            className="inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            <ArrowLeft className="size-4" />
            All classes
          </Link>
        </div>
        <Card className="mx-auto mt-6 w-full max-w-2xl space-y-4 px-8 py-10">
          <div className="flex items-center gap-3">
            <div className="flex size-10 items-center justify-center rounded-full bg-destructive/10 text-destructive">
              <AlertCircle className="size-5" />
            </div>
            <div>
              <h1 className="text-lg font-semibold tracking-tight">
                Training failed
              </h1>
              <p className="text-sm text-muted-foreground">
                {job.data.errorMessage ?? "Something went wrong while training."}
              </p>
            </div>
          </div>
          <div className="flex justify-end">
            <Link href={retryHref} className={buttonVariants()}>
              Try again
            </Link>
          </div>
        </Card>
      </main>
    );
  }

  return (
    <main className="flex flex-1 flex-col px-4 py-10 sm:px-6 sm:py-14">
      <div className="mx-auto w-full max-w-2xl">
        <Link
          href="/"
          className="inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors"
        >
          <ArrowLeft className="size-4" />
          All classes
        </Link>
      </div>

      <div className="mx-auto mt-6 w-full max-w-2xl space-y-8">
        <header className="space-y-1">
          {subject && grade ? (
            <div className="text-xs uppercase tracking-wider text-muted-foreground">
              {subject} · Grade {grade}
            </div>
          ) : null}
          <h1 className="text-3xl font-semibold tracking-tight">
            {className}
          </h1>
          <div className="flex items-center gap-3 text-sm text-muted-foreground">
            <span>
              {isComplete ? "Training complete!" : "Training in progress…"}
            </span>
            <span aria-hidden>·</span>
            <span className="tabular-nums">{formatDuration(elapsedMs)}</span>
          </div>
        </header>

        <Card className="px-6 py-5 sm:px-8 sm:py-6">
          <ProgressStages job={job.data} />
        </Card>

        <SampleQaPreview samples={job.data.sampleQA} />
      </div>
    </main>
  );
}
