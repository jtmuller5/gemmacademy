"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  GraduationCap,
  Loader2,
  Plus,
  Sparkles,
  Users,
  AlertCircle,
} from "lucide-react";
import { toast } from "sonner";
import { buttonVariants } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { trpc } from "@/lib/trpc-client";
import { formatRelative } from "@/lib/format";
import type { ClassDetail } from "@/server/types";

export default function HomePage() {
  const { data: classes, isLoading } = trpc.classes.list.useQuery();

  return (
    <main className="flex flex-1 flex-col">
      <header className="mx-auto flex w-full max-w-5xl items-center justify-between px-6 pt-8">
        <div className="flex items-center gap-2">
          <GraduationCap className="size-6 text-primary" />
          <span className="text-base font-semibold tracking-tight">
            Gemmacademy
          </span>
        </div>
        {classes && classes.length > 0 ? (
          <Link href="/new" className={buttonVariants({ size: "sm" })}>
            <Plus className="size-4" />
            New class
          </Link>
        ) : null}
      </header>

      <div className="mx-auto flex w-full max-w-5xl flex-1 flex-col px-6 py-10">
        {isLoading ? (
          <div className="flex flex-1 items-center justify-center">
            <Loader2 className="size-6 animate-spin text-muted-foreground" />
          </div>
        ) : classes && classes.length > 0 ? (
          <ClassGrid classes={classes} />
        ) : (
          <EmptyState />
        )}
      </div>
    </main>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-1 flex-col items-center justify-center text-center animate-in fade-in-50 duration-500">
      <div className="flex size-14 items-center justify-center rounded-2xl bg-primary/10 text-primary">
        <Sparkles className="size-7" />
      </div>
      <h1 className="mt-6 max-w-md text-3xl font-semibold tracking-tight">
        Train an AI tutor your students can take home.
      </h1>
      <p className="mt-3 max-w-md text-muted-foreground leading-relaxed">
        Upload your week&apos;s lesson materials. We&apos;ll train an AI tutor
        your students can use offline at home, on any phone.
      </p>
      <Link
        href="/new"
        className={buttonVariants({ size: "lg", className: "mt-8" })}
      >
        <Plus className="size-4" />
        Create your first class
      </Link>
    </div>
  );
}

function ClassGrid({ classes }: { classes: ClassDetail[] }) {
  return (
    <div>
      <h1 className="text-2xl font-semibold tracking-tight">Your classes</h1>
      <p className="mt-1 text-sm text-muted-foreground">
        Tutors you&apos;ve trained for your students.
      </p>
      <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {classes.map((cls) => (
          <ClassCard key={cls.id} cls={cls} />
        ))}
      </div>
    </div>
  );
}

function ClassCard({ cls }: { cls: ClassDetail }) {
  const router = useRouter();

  function handleClick() {
    if (cls.status === "ready") {
      router.push(`/classes/${cls.id}`);
    } else if (cls.status === "training" && cls.jobId) {
      router.push(`/jobs/${cls.jobId}`);
    } else {
      toast.error(cls.errorMessage ?? "Training failed", {
        action: {
          label: "Retry",
          onClick: () =>
            router.push(
              `/new?name=${encodeURIComponent(cls.name)}&grade=${cls.grade}&subject=${encodeURIComponent(cls.subject)}`,
            ),
        },
      });
    }
  }

  return (
    <Card
      className="group cursor-pointer p-5 transition-all hover:border-primary/40 hover:shadow-sm"
      onClick={handleClick}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          handleClick();
        }
      }}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="text-xs uppercase tracking-wider text-muted-foreground">
            {cls.subject} · Grade {cls.grade}
          </div>
          <h3 className="mt-1 truncate text-base font-semibold tracking-tight">
            {cls.name}
          </h3>
        </div>
        <StatusPill status={cls.status} />
      </div>
      <div className="mt-6 flex items-center justify-between text-xs text-muted-foreground">
        <span>{formatRelative(cls.createdAt)}</span>
        {cls.status === "ready" ? (
          <span className="inline-flex items-center gap-1">
            <Users className="size-3.5" />
            {cls.studentsDownloaded ?? 0}
          </span>
        ) : null}
      </div>
    </Card>
  );
}

function StatusPill({ status }: { status: ClassDetail["status"] }) {
  if (status === "ready") {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full bg-primary/10 px-2.5 py-0.5 text-xs font-medium text-primary">
        <span className="size-1.5 rounded-full bg-primary" />
        Ready
      </span>
    );
  }
  if (status === "training") {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full bg-amber-100 px-2.5 py-0.5 text-xs font-medium text-amber-800">
        <Loader2 className="size-3 animate-spin" />
        Training
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full bg-destructive/10 px-2.5 py-0.5 text-xs font-medium text-destructive">
      <AlertCircle className="size-3" />
      Failed
    </span>
  );
}
