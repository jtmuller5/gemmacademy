"use client";

import { Suspense, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { ArrowLeft, Sparkles } from "lucide-react";
import { toast } from "sonner";
import {
  UploadDropzone,
  type UploadFile,
} from "@/components/upload-dropzone";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { trpc } from "@/lib/trpc-client";
import { GRADES, SUBJECTS, type Grade, type Subject } from "@/server/types";

export default function NewClassPage() {
  return (
    <Suspense fallback={null}>
      <NewClassForm />
    </Suspense>
  );
}

function readAsBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error ?? new Error("read failed"));
    reader.onload = () => {
      const result = reader.result;
      if (typeof result !== "string") {
        reject(new Error("unexpected reader result"));
        return;
      }
      const comma = result.indexOf(",");
      resolve(comma >= 0 ? result.slice(comma + 1) : result);
    };
    reader.readAsDataURL(file);
  });
}

function NewClassForm() {
  const router = useRouter();
  const params = useSearchParams();
  const utils = trpc.useUtils();

  const initialGrade = params.get("grade");
  const initialSubject = params.get("subject");

  const [name, setName] = useState(params.get("name") ?? "");
  const [grade, setGrade] = useState<Grade | undefined>(
    GRADES.includes(initialGrade as Grade)
      ? (initialGrade as Grade)
      : undefined,
  );
  const [subject, setSubject] = useState<Subject | undefined>(
    SUBJECTS.includes(initialSubject as Subject)
      ? (initialSubject as Subject)
      : undefined,
  );
  const [files, setFiles] = useState<UploadFile[]>([]);

  const create = trpc.classes.create.useMutation({
    onSuccess: async ({ jobId }) => {
      await utils.classes.list.invalidate();
      router.push(`/jobs/${jobId}`);
    },
    onError: (err) => toast.error(err.message),
  });

  const canSubmit =
    name.trim().length > 0 &&
    !!grade &&
    !!subject &&
    files.length > 0 &&
    !create.isPending;

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!grade || !subject || files.length === 0 || !name.trim()) return;
    try {
      const encoded = await Promise.all(
        files.map(async (f) => ({
          name: f.name,
          size: f.size,
          contentBase64: await readAsBase64(f.file),
        })),
      );
      create.mutate({
        name: name.trim(),
        grade,
        subject,
        files: encoded,
      });
    } catch {
      toast.error("Couldn't read one of the files");
    }
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

      <form
        onSubmit={handleSubmit}
        className="mx-auto mt-6 w-full max-w-2xl space-y-8"
      >
        <div>
          <h1 className="text-3xl font-semibold tracking-tight">
            Train a new tutor
          </h1>
          <p className="mt-2 text-muted-foreground">
            Give your class a name, pick the grade and subject, and drop in
            this week&apos;s lesson materials.
          </p>
        </div>

        <Card className="space-y-6 px-6 py-7 sm:px-8 sm:py-8">
          <div className="space-y-2">
            <Label htmlFor="name">Class name</Label>
            <Input
              id="name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="4th Grade Math — Mrs. Henderson"
              maxLength={100}
            />
          </div>

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="grade">Grade level</Label>
              <Select
                value={grade}
                onValueChange={(v) => setGrade(v as Grade)}
              >
                <SelectTrigger id="grade" className="w-full">
                  <SelectValue placeholder="Select grade" />
                </SelectTrigger>
                <SelectContent>
                  {GRADES.map((g) => (
                    <SelectItem key={g} value={g}>
                      {g === "K" ? "Kindergarten" : `Grade ${g}`}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="subject">Subject</Label>
              <Select
                value={subject}
                onValueChange={(v) => setSubject(v as Subject)}
              >
                <SelectTrigger id="subject" className="w-full">
                  <SelectValue placeholder="Select subject" />
                </SelectTrigger>
                <SelectContent>
                  {SUBJECTS.map((s) => (
                    <SelectItem key={s} value={s}>
                      {s}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="space-y-2">
            <Label>Lesson materials</Label>
            <UploadDropzone files={files} onChange={setFiles} />
          </div>
        </Card>

        <div className="flex justify-end">
          <Button
            type="submit"
            size="lg"
            disabled={!canSubmit}
            className="min-w-[180px]"
          >
            <Sparkles className="size-4" />
            {create.isPending ? "Starting…" : "Train tutor"}
          </Button>
        </div>
      </form>
    </main>
  );
}
