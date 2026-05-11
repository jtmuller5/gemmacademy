"use client";

import { useCallback, useId, useRef, useState } from "react";
import { FileText, Upload, X } from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { formatBytes } from "@/lib/format";

export type UploadFile = {
  id: string;
  name: string;
  size: number;
  file: File;
};

const ACCEPT = ".pdf,.docx,.txt,.md";
const ACCEPT_RE = /\.(pdf|docx|txt|md)$/i;
const MAX_FILES = 10;
const MAX_TOTAL_BYTES = 50 * 1024 * 1024;

type Props = {
  files: UploadFile[];
  onChange: (next: UploadFile[]) => void;
};

export function UploadDropzone({ files, onChange }: Props) {
  const inputId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const addFiles = useCallback(
    (incoming: FileList | File[]) => {
      const list = Array.from(incoming);
      const accepted: UploadFile[] = [];
      let totalBytes = files.reduce((sum, f) => sum + f.size, 0);

      for (const file of list) {
        if (!ACCEPT_RE.test(file.name)) {
          toast.error(`${file.name} isn't a supported type`);
          continue;
        }
        if (files.length + accepted.length >= MAX_FILES) {
          toast.error(`Up to ${MAX_FILES} files`);
          break;
        }
        if (totalBytes + file.size > MAX_TOTAL_BYTES) {
          toast.error("That would exceed the 50 MB total limit");
          break;
        }
        totalBytes += file.size;
        accepted.push({
          id: `${file.name}-${file.size}-${Math.random().toString(36).slice(2, 8)}`,
          name: file.name,
          size: file.size,
          file,
        });
      }

      if (accepted.length > 0) {
        onChange([...files, ...accepted]);
      }
    },
    [files, onChange],
  );

  function remove(id: string) {
    onChange(files.filter((f) => f.id !== id));
  }

  return (
    <div>
      <label
        htmlFor={inputId}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setIsDragging(false);
          addFiles(e.dataTransfer.files);
        }}
        className={cn(
          "flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed bg-muted/30 px-8 py-14 text-center transition-colors",
          isDragging
            ? "border-primary bg-primary/5"
            : "border-border hover:border-primary/40 hover:bg-muted/50",
        )}
      >
        <div className="flex size-12 items-center justify-center rounded-full bg-primary/10 text-primary">
          <Upload className="size-5" />
        </div>
        <div className="mt-4 text-sm font-medium">
          Drop files here or click to upload
        </div>
        <div className="mt-1 text-xs text-muted-foreground">
          PDF, DOCX, TXT, MD · up to {MAX_FILES} files · 50 MB total
        </div>
        <input
          ref={inputRef}
          id={inputId}
          type="file"
          accept={ACCEPT}
          multiple
          className="sr-only"
          onChange={(e) => {
            if (e.target.files) addFiles(e.target.files);
            e.target.value = "";
          }}
        />
      </label>

      {files.length > 0 ? (
        <ul className="mt-4 space-y-2">
          {files.map((file) => (
            <li
              key={file.id}
              className="flex items-center justify-between gap-3 rounded-md border bg-card px-3 py-2 text-sm"
            >
              <div className="flex min-w-0 items-center gap-2.5">
                <FileText className="size-4 shrink-0 text-muted-foreground" />
                <span className="truncate">{file.name}</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-xs text-muted-foreground">
                  {formatBytes(file.size)}
                </span>
                <button
                  type="button"
                  onClick={() => remove(file.id)}
                  aria-label={`Remove ${file.name}`}
                  className="text-muted-foreground hover:text-foreground transition-colors"
                >
                  <X className="size-4" />
                </button>
              </div>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
