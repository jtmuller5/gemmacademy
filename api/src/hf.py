"""Hugging Face Hub upload wrapper.

Uploads the .litertlm artifact (and optionally the lesson + sample data) to a
new model repo named after the class. Returns the public URL.

Token resolution order:
  1. HF_TOKEN env var
  2. HUGGING_FACE_HUB_TOKEN env var
  3. ~/.cache/huggingface/token (huggingface-cli login default)

If no token is available, raises RuntimeError. The worker catches this and
records it in Job.error_message rather than crashing the service.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def _resolve_token() -> Optional[str]:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        token = os.environ.get(key)
        if token:
            return token
    cached = Path.home() / ".cache" / "huggingface" / "token"
    if cached.exists():
        return cached.read_text().strip() or None
    return None


def _resolve_user() -> str:
    return os.environ.get("HF_USERNAME", "jtmuller")


def upload_to_hf(
    artifact_path: Path,
    class_id: str,
    *,
    repo_name: Optional[str] = None,
    extra_files: Optional[list[Path]] = None,
) -> str:
    """Push the artifact to a new repo on HF Hub. Returns the public URL."""

    from huggingface_hub import HfApi, create_repo  # imported lazily

    token = _resolve_token()
    if not token:
        raise RuntimeError(
            "No Hugging Face token found (set HF_TOKEN or run `huggingface-cli login`)."
        )

    user = _resolve_user()
    repo_name = repo_name or f"gemmacademy-{class_id[:8]}"
    repo_id = f"{user}/{repo_name}"

    api = HfApi(token=token)
    create_repo(repo_id, token=token, repo_type="model", exist_ok=True, private=False)

    api.upload_file(
        path_or_fileobj=str(artifact_path),
        path_in_repo=artifact_path.name,
        repo_id=repo_id,
        repo_type="model",
    )

    if extra_files:
        for f in extra_files:
            if not f.exists():
                continue
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=f.name,
                repo_id=repo_id,
                repo_type="model",
            )

    return f"https://huggingface.co/{repo_id}"
