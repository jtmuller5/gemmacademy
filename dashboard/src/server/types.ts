export const GRADES = ["K", "1", "2", "3", "4", "5", "6", "7", "8"] as const;
export type Grade = (typeof GRADES)[number];

export const SUBJECTS = [
  "Math",
  "Science",
  "ELA",
  "Social Studies",
  "Other",
] as const;
export type Subject = (typeof SUBJECTS)[number];

export type ClassStatus = "training" | "ready" | "failed";

export type ClassSummary = {
  id: string;
  name: string;
  grade: Grade;
  subject: Subject;
  createdAt: string;
  status: ClassStatus;
  studentsDownloaded?: number;
  jobId?: string;
};

export type ClassDetail = ClassSummary & {
  modelUrl?: string;
  modelSizeBytes?: number;
  trainingExamples?: number;
  errorMessage?: string;
};

export type JobStage =
  | "reading"
  | "generating"
  | "training"
  | "packaging"
  | "ready";

export type JobStatusValue = "queued" | "running" | "complete" | "failed";

export type JobStatus = {
  id: string;
  classId: string;
  status: JobStatusValue;
  startedAt: string;
  completedAt?: string;
  currentStage: JobStage;
  stageProgress: number;
  questionsGenerated?: number;
  questionsTarget?: number;
  trainLoss?: number;
  sampleQA?: Array<{ q: string; a: string }>;
  errorMessage?: string;
};
