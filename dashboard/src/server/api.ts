export function apiBaseUrl(): string {
  const url = process.env.API_BASE_URL;
  if (!url) {
    throw new Error(
      "API_BASE_URL is not set — point it at the FastAPI backend (see README).",
    );
  }
  return url.replace(/\/$/, "");
}
