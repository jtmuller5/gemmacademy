import { TRPCError } from "@trpc/server";
import { z } from "zod";
import { router, publicProcedure } from "../trpc";
import { GRADES, SUBJECTS, type ClassDetail, type ClassSummary } from "../types";
import { apiBaseUrl } from "../api";

const fileSchema = z.object({
  name: z.string(),
  size: z.number(),
  contentBase64: z.string(),
});

export const classesRouter = router({
  list: publicProcedure.query(async (): Promise<ClassSummary[]> => {
    const r = await fetch(`${apiBaseUrl()}/classes`, { cache: "no-store" });
    if (!r.ok) {
      throw new TRPCError({
        code: "INTERNAL_SERVER_ERROR",
        message: `Backend /classes failed: ${r.status}`,
      });
    }
    return r.json();
  }),

  get: publicProcedure
    .input(z.object({ id: z.string() }))
    .query(async ({ input }): Promise<ClassDetail> => {
      const r = await fetch(`${apiBaseUrl()}/classes/${input.id}`, {
        cache: "no-store",
      });
      if (r.status === 404) {
        throw new TRPCError({ code: "NOT_FOUND", message: "Class not found" });
      }
      if (!r.ok) {
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message: `Backend /classes/${input.id} failed: ${r.status}`,
        });
      }
      return r.json();
    }),

  create: publicProcedure
    .input(
      z.object({
        name: z.string().min(1).max(100),
        grade: z.enum(GRADES),
        subject: z.enum(SUBJECTS),
        files: z.array(fileSchema).min(1).max(10),
      }),
    )
    .mutation(async ({ input }): Promise<{ id: string; jobId: string }> => {
      const fd = new FormData();
      fd.append("name", input.name);
      fd.append("grade", input.grade);
      fd.append("subject", input.subject);
      for (const f of input.files) {
        const bytes = Buffer.from(f.contentBase64, "base64");
        fd.append(
          "files",
          new Blob([new Uint8Array(bytes)]),
          f.name,
        );
      }
      const r = await fetch(`${apiBaseUrl()}/classes`, {
        method: "POST",
        body: fd,
      });
      if (!r.ok) {
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message: await r.text(),
        });
      }
      const body = (await r.json()) as { id: string; job_id?: string; jobId?: string };
      return { id: body.id, jobId: body.jobId ?? body.job_id! };
    }),

  delete: publicProcedure
    .input(z.object({ id: z.string() }))
    .mutation(async ({ input }) => {
      const r = await fetch(`${apiBaseUrl()}/classes/${input.id}`, {
        method: "DELETE",
      });
      if (!r.ok && r.status !== 404) {
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message: `Backend DELETE /classes/${input.id} failed: ${r.status}`,
        });
      }
      return { success: true as const };
    }),
});
