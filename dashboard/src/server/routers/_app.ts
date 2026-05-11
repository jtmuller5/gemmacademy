import { router } from "../trpc";
import { classesRouter } from "./classes";
import { jobsRouter } from "./jobs";

export const appRouter = router({
  classes: classesRouter,
  jobs: jobsRouter,
});

export type AppRouter = typeof appRouter;
