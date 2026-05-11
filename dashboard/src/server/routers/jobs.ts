import { TRPCError } from "@trpc/server";
import { z } from "zod";
import { router, publicProcedure } from "../trpc";
import type { JobStatus } from "../types";
import { apiBaseUrl } from "../api";

export const jobsRouter = router({
  get: publicProcedure
    .input(z.object({ id: z.string() }))
    .query(async ({ input }): Promise<JobStatus> => {
      const r = await fetch(`${apiBaseUrl()}/jobs/${input.id}`, {
        cache: "no-store",
      });
      if (r.status === 404) {
        throw new TRPCError({ code: "NOT_FOUND", message: "Job not found" });
      }
      if (!r.ok) {
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message: `Backend /jobs/${input.id} failed: ${r.status}`,
        });
      }
      return r.json();
    }),
});
