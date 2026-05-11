"use client";

import { use, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { ArrowLeft, Trash2, RotateCcw, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { ShareCard } from "@/components/share-card";
import { Button, buttonVariants } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { trpc } from "@/lib/trpc-client";

export default function ClassPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const router = useRouter();
  const utils = trpc.useUtils();
  const [confirmDelete, setConfirmDelete] = useState(false);

  const { data: cls, isLoading, error } = trpc.classes.get.useQuery({ id });

  const deleteMutation = trpc.classes.delete.useMutation({
    onSuccess: async () => {
      await utils.classes.list.invalidate();
      toast.success("Class deleted");
      router.push("/");
    },
    onError: (err) => toast.error(err.message),
  });

  if (isLoading) {
    return (
      <div className="flex flex-1 items-center justify-center">
        <Loader2 className="size-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (error || !cls) {
    return (
      <div className="flex flex-1 items-center justify-center text-sm text-muted-foreground">
        Class not found.{" "}
        <Link href="/" className="ml-2 underline">
          Back to classes
        </Link>
      </div>
    );
  }

  if (cls.status === "training" && cls.jobId) {
    router.replace(`/jobs/${cls.jobId}`);
    return null;
  }

  if (cls.status === "failed") {
    return (
      <div className="flex flex-1 items-center justify-center text-sm text-muted-foreground">
        This class failed to train.{" "}
        <Link href="/" className="ml-2 underline">
          Back to classes
        </Link>
      </div>
    );
  }

  const retrainHref = `/new?name=${encodeURIComponent(cls.name)}&grade=${cls.grade}&subject=${encodeURIComponent(cls.subject)}`;

  return (
    <div className="flex flex-1 flex-col px-4 py-10 sm:px-6 sm:py-14">
      <div className="mx-auto w-full max-w-2xl">
        <Link
          href="/"
          className="inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors"
        >
          <ArrowLeft className="size-4" />
          All classes
        </Link>
      </div>

      <div className="mt-6">
        <ShareCard cls={cls} />
      </div>

      <div className="mx-auto mt-8 flex w-full max-w-2xl items-center justify-between text-sm">
        <Link
          href={retrainHref}
          className={cn(
            buttonVariants({ variant: "ghost", size: "sm" }),
            "text-muted-foreground",
          )}
        >
          <RotateCcw className="size-4" />
          Re-train this class
        </Link>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          className="text-muted-foreground hover:text-destructive"
          onClick={() => setConfirmDelete(true)}
        >
          <Trash2 className="size-4" />
          Delete class
        </Button>
      </div>

      <Dialog open={confirmDelete} onOpenChange={setConfirmDelete}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete this class?</DialogTitle>
            <DialogDescription>
              This removes the tutor model. Students who already downloaded it
              keep their copy, but new students won&apos;t be able to scan the
              QR code.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setConfirmDelete(false)}
              disabled={deleteMutation.isPending}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={() => deleteMutation.mutate({ id })}
              disabled={deleteMutation.isPending}
            >
              {deleteMutation.isPending ? "Deleting…" : "Delete class"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
