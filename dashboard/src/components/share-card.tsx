"use client";

import { useState } from "react";
import { QRCodeSVG } from "qrcode.react";
import { Check, Copy } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { formatBytes, formatRelative } from "@/lib/format";
import type { ClassDetail } from "@/server/types";

type Props = {
  cls: ClassDetail;
};

export function ShareCard({ cls }: Props) {
  const [copied, setCopied] = useState(false);
  const url = cls.modelUrl ?? "";

  async function copy() {
    try {
      await navigator.clipboard.writeText(url);
      setCopied(true);
      toast.success("Link copied");
      setTimeout(() => setCopied(false), 1800);
    } catch {
      toast.error("Could not copy");
    }
  }

  return (
    <Card className="mx-auto max-w-2xl px-8 py-10 sm:px-12 sm:py-12 animate-in fade-in-50 duration-500">
      <div className="flex items-center justify-center gap-6 text-xs text-muted-foreground">
        {cls.modelSizeBytes ? (
          <span>{formatBytes(cls.modelSizeBytes)}</span>
        ) : null}
        {cls.trainingExamples ? (
          <>
            <span aria-hidden>·</span>
            <span>
              {cls.trainingExamples.toLocaleString()} training examples
            </span>
          </>
        ) : null}
        <span aria-hidden>·</span>
        <span>Created {formatRelative(cls.createdAt)}</span>
      </div>

      <div className="mt-8 flex flex-col items-center text-center">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">
          {cls.subject} · Grade {cls.grade}
        </div>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight">
          {cls.name}
        </h1>
      </div>

      <div className="mt-10 flex justify-center">
        <div className="rounded-2xl border bg-white p-6 shadow-sm animate-in fade-in-0 zoom-in-95 duration-500">
          {url ? (
            <QRCodeSVG
              value={url}
              size={224}
              level="M"
              marginSize={1}
              className="block"
            />
          ) : (
            <div className="size-56 rounded bg-muted" />
          )}
        </div>
      </div>

      <div className="mt-8 flex items-center gap-2">
        <code className="flex-1 truncate rounded-md border bg-muted/40 px-3 py-2 text-xs text-muted-foreground">
          {url}
        </code>
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={copy}
          aria-label="Copy link"
        >
          {copied ? (
            <Check className="size-4" />
          ) : (
            <Copy className="size-4" />
          )}
          <span className="ml-1.5">{copied ? "Copied" : "Copy"}</span>
        </Button>
      </div>

      <Separator className="my-10" />

      <div>
        <h2 className="text-sm font-semibold tracking-tight">How to share</h2>
        <ol className="mt-4 space-y-3 text-sm text-muted-foreground">
          <li className="flex gap-3">
            <span className="flex size-6 shrink-0 items-center justify-center rounded-full bg-primary/10 text-xs font-semibold text-primary">
              1
            </span>
            <span>Have students open the Gemmacademy app.</span>
          </li>
          <li className="flex gap-3">
            <span className="flex size-6 shrink-0 items-center justify-center rounded-full bg-primary/10 text-xs font-semibold text-primary">
              2
            </span>
            <span>They scan this QR code (any phone camera works).</span>
          </li>
          <li className="flex gap-3">
            <span className="flex size-6 shrink-0 items-center justify-center rounded-full bg-primary/10 text-xs font-semibold text-primary">
              3
            </span>
            <span>
              The tutor downloads to their phone — about 2.5 GB. They use it
              offline at home.
            </span>
          </li>
        </ol>
      </div>
    </Card>
  );
}
