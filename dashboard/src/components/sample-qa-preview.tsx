"use client";

import { Sparkles } from "lucide-react";
import { Card } from "@/components/ui/card";

type Props = {
  samples?: Array<{ q: string; a: string }>;
};

export function SampleQaPreview({ samples }: Props) {
  if (!samples || samples.length === 0) return null;

  return (
    <div>
      <div className="flex items-center gap-2 text-sm font-medium">
        <Sparkles className="size-4 text-primary" />
        Sample questions your tutor is learning
      </div>
      <div className="mt-3 grid gap-3">
        {samples.map((sample) => (
          <Card
            key={sample.q}
            className="px-4 py-4 animate-in fade-in-0 slide-in-from-bottom-1 duration-500"
          >
            <div className="text-xs uppercase tracking-wider text-muted-foreground">
              Q
            </div>
            <p className="mt-1 text-sm font-medium leading-relaxed">
              {sample.q}
            </p>
            <div className="mt-3 text-xs uppercase tracking-wider text-muted-foreground">
              A
            </div>
            <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
              {sample.a}
            </p>
          </Card>
        ))}
      </div>
    </div>
  );
}
