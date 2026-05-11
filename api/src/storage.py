"""Filesystem helpers: upload dirs, output dirs, text extraction."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
MAX_FILES = 10
MAX_TOTAL_BYTES = 50 * 1024 * 1024  # 50 MB

API_ROOT = Path(__file__).resolve().parent.parent
UPLOADS_ROOT = Path(os.environ.get("UPLOADS_ROOT", API_ROOT / "uploads")).resolve()
OUTPUTS_ROOT = Path(os.environ.get("OUTPUTS_ROOT", API_ROOT / "outputs")).resolve()


def upload_dir_for(job_id: str) -> Path:
    p = UPLOADS_ROOT / job_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def output_dir_for(class_id: str) -> Path:
    p = OUTPUTS_ROOT / class_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_filename(name: str) -> str:
    """Strip path components and keep only the basename plus extension."""
    base = Path(name).name
    return base.replace("/", "_").replace("\\", "_") or "upload"


def extract_text_from_uploads(upload_dir: Path) -> str:
    """Concatenate text from every supported file in the directory."""
    chunks: list[str] = []
    for path in sorted(upload_dir.iterdir()):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            continue
        try:
            text = _read_one(path)
        except Exception as e:  # noqa: BLE001
            text = f"[Failed to extract {path.name}: {e}]"
        chunks.append(f"### {path.name}\n\n{text.strip()}")
    return "\n\n".join(chunks).strip()


def _read_one(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if ext == ".pdf":
        from pypdf import PdfReader  # imported lazily

        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    if ext == ".docx":
        import docx  # python-docx

        d = docx.Document(str(path))
        return "\n".join(p.text for p in d.paragraphs)
    return ""


def save_lesson_text(output_dir: Path, lesson_text: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    p = output_dir / "lesson.txt"
    p.write_text(lesson_text, encoding="utf-8")
    return p


def remove_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
