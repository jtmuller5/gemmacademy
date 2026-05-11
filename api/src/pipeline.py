"""Wrap the Phase 2 training scripts as Python coroutines.

There are two execution modes, selected via the DEMO_MODE env var:

* DEMO_MODE=true (default for live demos): every stage just sleeps with
  realistic timing, samples Q&A from the existing fractions JSONL, and points
  the resulting class at the pre-trained `jtmuller/gemmacademy-fractions-v1`
  model. End-to-end ~30s.

* DEMO_MODE=false (real pipeline): each stage shells out to the real
  generate_qa.py / train.py / convert script under ~/projects/gemmacademy/
  training/ and pushes the resulting .litertlm to a fresh HF repo.
  End-to-end ~10-15 min.

Both modes call the same `update_stage` / `mark_*` callbacks so the worker is
oblivious to which mode is active.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Awaitable, Callable, Optional

from . import storage

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRAINING_DIR = Path(os.environ.get(
    "TRAINING_DIR",
    "/home/joemuller/projects/gemmacademy/training",
)).resolve()

# Hand-curated fallback Q&A used by DEMO_MODE so we don't depend on a
# checked-in dataset path that might not exist in fresh clones.
DEMO_QA_PATH = TRAINING_DIR / "qa-fractions.jsonl"

PRETRAINED_DEMO_MODEL_URL = "https://huggingface.co/jtmuller/gemmacademy-fractions-v1"
PRETRAINED_DEMO_MODEL_BYTES = 4_800_000_000  # ~4.8 GB wi8 artifact
DEMO_QUESTION_TARGET = 500


def is_demo_mode() -> bool:
    val = os.environ.get("DEMO_MODE", "true").strip().lower()
    return val in {"1", "true", "yes", "on"}


def _demo_tick_seconds() -> float:
    try:
        return float(os.environ.get("DEMO_TICK_SECONDS", "1.0"))
    except ValueError:
        return 1.0


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

# stage, progress, optional kwargs (questions_generated, train_loss, sample_qa)
StageCallback = Callable[..., Awaitable[None]]


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


async def run_pipeline(
    *,
    job_id: str,
    class_id: str,
    upload_dir: Path,
    output_dir: Path,
    on_stage: StageCallback,
    on_complete: Callable[..., Awaitable[None]],
) -> None:
    """Drive a job through every stage. Raises on failure (the worker catches)."""

    # ---- Stage: reading ---------------------------------------------------
    await on_stage(stage="reading", progress=0.0)
    lesson_text = await asyncio.to_thread(storage.extract_text_from_uploads, upload_dir)
    if not lesson_text:
        raise RuntimeError("No readable text found in uploaded files.")
    storage.save_lesson_text(output_dir, lesson_text)
    await on_stage(stage="reading", progress=1.0)

    if is_demo_mode():
        await _run_demo(
            class_id=class_id,
            lesson_text=lesson_text,
            output_dir=output_dir,
            on_stage=on_stage,
            on_complete=on_complete,
        )
        return

    await _run_real(
        job_id=job_id,
        class_id=class_id,
        lesson_text=lesson_text,
        output_dir=output_dir,
        on_stage=on_stage,
        on_complete=on_complete,
    )


# ---------------------------------------------------------------------------
# DEMO mode
# ---------------------------------------------------------------------------


async def _run_demo(
    *,
    class_id: str,
    lesson_text: str,
    output_dir: Path,
    on_stage: StageCallback,
    on_complete: Callable[..., Awaitable[None]],
) -> None:
    pool = _load_demo_qa_pool()

    # generating: ~12s, fake progress with sample_qa updates every tick
    target = DEMO_QUESTION_TARGET
    ticks = 12
    for i in range(1, ticks + 1):
        progress = i / ticks
        questions_generated = int(progress * target)
        await on_stage(
            stage="generating",
            progress=progress,
            questions_generated=questions_generated,
            questions_target=target,
            sample_qa=_sample_qa(pool),
        )
        await asyncio.sleep(_demo_tick_seconds())

    # training: ~10s, simulated descending train_loss
    ticks = 10
    losses = _fake_loss_curve(ticks, start=2.4, end=0.8)
    for i in range(1, ticks + 1):
        progress = i / ticks
        await on_stage(
            stage="training",
            progress=progress,
            train_loss=losses[i - 1],
            sample_qa=_sample_qa(pool),
        )
        await asyncio.sleep(_demo_tick_seconds())

    # packaging: ~6s
    ticks = 6
    for i in range(1, ticks + 1):
        progress = i / ticks
        await on_stage(stage="packaging", progress=progress)
        await asyncio.sleep(_demo_tick_seconds())

    await on_complete(
        model_url=PRETRAINED_DEMO_MODEL_URL,
        model_size_bytes=PRETRAINED_DEMO_MODEL_BYTES,
        training_examples=target,
    )


def _load_demo_qa_pool() -> list[dict]:
    if not DEMO_QA_PATH.exists():
        return [
            {"q": "What is the Pizza Method?",
             "a": "It's how Mrs. Henderson teaches fractions — equal slices, count on top, cut on bottom."},
            {"q": "Why is 3/4 bigger than 5/8?",
             "a": "Because the bigger the bottom, the smaller the slice."},
            {"q": "What does 4/4 mean?",
             "a": "All the slices add back up to one whole pizza."},
        ]
    pool: list[dict] = []
    with DEMO_QA_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                pool.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pool


def _sample_qa(pool: list[dict]) -> list[dict]:
    if not pool:
        return []
    return random.sample(pool, k=min(3, len(pool)))


def _fake_loss_curve(n: int, *, start: float, end: float) -> list[float]:
    if n <= 1:
        return [end]
    step = (start - end) / (n - 1)
    return [round(start - step * i, 3) for i in range(n)]


# ---------------------------------------------------------------------------
# REAL mode
# ---------------------------------------------------------------------------


async def _run_real(
    *,
    job_id: str,
    class_id: str,
    lesson_text: str,
    output_dir: Path,
    on_stage: StageCallback,
    on_complete: Callable[..., Awaitable[None]],
) -> None:
    from . import hf  # imported lazily so DEMO_MODE never needs HF creds

    qa_path = output_dir / "qa.jsonl"
    lora_dir = output_dir / "lora-adapter"
    merged_dir = output_dir / "merged-model"
    litertlm_dir = output_dir / "litertlm-output"
    progress_path = output_dir / f"progress-{job_id}.json"

    lesson_path = storage.save_lesson_text(output_dir, lesson_text)

    # ---- generating -------------------------------------------------------
    target = int(os.environ.get("QA_TARGET", "500"))
    await on_stage(
        stage="generating", progress=0.0,
        questions_generated=0, questions_target=target,
    )
    gen_cmd = [
        "uv", "run", "python", "generate_qa.py",
        "--input", str(lesson_path),
        "--output", str(qa_path),
        "--num-pairs", str(target),
    ]
    await _run_subprocess_with_progress(
        cmd=gen_cmd,
        cwd=TRAINING_DIR,
        progress_path=progress_path,
        on_stage=on_stage,
        stage="generating",
        progress_target=target,
        progress_field="questions_generated",
    )
    await on_stage(stage="generating", progress=1.0)

    # ---- training ---------------------------------------------------------
    await on_stage(stage="training", progress=0.0)
    train_env = {
        "GEMMACADEMY_DATA_PATH": str(qa_path),
        "GEMMACADEMY_LORA_OUTPUT": str(lora_dir),
        "GEMMACADEMY_MERGED_OUTPUT": str(merged_dir),
        "GEMMACADEMY_TRAIN_OUTPUT_DIR": str(output_dir / "trainer-outputs"),
    }
    train_cmd = ["uv", "run", "python", "train.py"]
    await _run_subprocess_with_progress(
        cmd=train_cmd,
        cwd=TRAINING_DIR,
        progress_path=progress_path,
        on_stage=on_stage,
        stage="training",
        extra_env=train_env,
    )
    await on_stage(stage="training", progress=1.0)

    # ---- packaging --------------------------------------------------------
    await on_stage(stage="packaging", progress=0.0)
    litertlm_dir.mkdir(parents=True, exist_ok=True)
    chat_template = TRAINING_DIR / "reference-template" / "chat_template.jinja"
    convert_cmd = [
        "uv", "run", "litert-torch", "export_hf",
        str(merged_dir),
        str(litertlm_dir),
        "--externalize_embedder=True",
        "--use_jinja_template=True",
        "--bundle_litert_lm=True",
        "--quantization_recipe=dynamic_wi4_afp32",
        "--prefill_lengths=128,512,1024",
        "--cache_length=4096",
        f"--jinja_chat_template_override={chat_template}",
    ]
    await _run_subprocess_streaming(convert_cmd, cwd=TRAINING_DIR)
    await on_stage(stage="packaging", progress=0.5)

    artifact = _find_litertlm_artifact(litertlm_dir, class_id)
    model_url = await asyncio.to_thread(
        hf.upload_to_hf,
        artifact,
        class_id,
        extra_files=[lesson_path, qa_path],
    )
    await on_stage(stage="packaging", progress=1.0)

    training_examples = _count_lines(qa_path)
    await on_complete(
        model_url=model_url,
        model_size_bytes=artifact.stat().st_size,
        training_examples=training_examples,
    )


def _find_litertlm_artifact(litertlm_dir: Path, class_id: str) -> Path:
    candidates = list(litertlm_dir.glob("*.litertlm"))
    if not candidates:
        raise RuntimeError(f"No .litertlm produced in {litertlm_dir}")
    src = candidates[0]
    target = litertlm_dir / f"gemmacademy-{class_id[:8]}.litertlm"
    if src != target:
        src.rename(target)
    return target


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for line in f if line.strip())


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


async def _run_subprocess_streaming(
    cmd: list[str],
    *,
    cwd: Path,
    extra_env: Optional[dict] = None,
) -> None:
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)  # let `uv run` pick the right one
    if extra_env:
        env.update(extra_env)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert proc.stdout is not None
    async for line in proc.stdout:
        print(f"[{cmd[0]}] {line.decode(errors='replace').rstrip()}", flush=True)
    rc = await proc.wait()
    if rc != 0:
        raise RuntimeError(f"{' '.join(cmd[:3])}... exited with code {rc}")


async def _run_subprocess_with_progress(
    *,
    cmd: list[str],
    cwd: Path,
    progress_path: Path,
    on_stage: StageCallback,
    stage: str,
    progress_target: Optional[int] = None,
    progress_field: Optional[str] = None,
    extra_env: Optional[dict] = None,
) -> None:
    """Run a subprocess and translate its progress file into on_stage callbacks."""

    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env["GEMMACADEMY_PROGRESS_FILE"] = str(progress_path)
    if extra_env:
        env.update(extra_env)

    if progress_path.exists():
        progress_path.unlink()

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    async def _pump_stdout() -> None:
        assert proc.stdout is not None
        async for line in proc.stdout:
            print(f"[{cmd[2]}] {line.decode(errors='replace').rstrip()}", flush=True)

    async def _poll_progress() -> None:
        while proc.returncode is None:
            await _emit_progress_update(
                progress_path=progress_path,
                on_stage=on_stage,
                stage=stage,
                progress_target=progress_target,
                progress_field=progress_field,
            )
            await asyncio.sleep(2.0)

    pump = asyncio.create_task(_pump_stdout())
    poll = asyncio.create_task(_poll_progress())

    rc = await proc.wait()
    poll.cancel()
    try:
        await poll
    except asyncio.CancelledError:
        pass
    await pump

    # final read
    await _emit_progress_update(
        progress_path=progress_path,
        on_stage=on_stage,
        stage=stage,
        progress_target=progress_target,
        progress_field=progress_field,
    )

    if rc != 0:
        raise RuntimeError(f"{' '.join(cmd[:3])}... exited with code {rc}")


async def _emit_progress_update(
    *,
    progress_path: Path,
    on_stage: StageCallback,
    stage: str,
    progress_target: Optional[int],
    progress_field: Optional[str],
) -> None:
    if not progress_path.exists():
        return
    try:
        data = json.loads(progress_path.read_text())
    except (OSError, json.JSONDecodeError):
        return

    kwargs: dict = {"stage": stage}
    progress: Optional[float] = None

    if stage == "generating":
        gen = int(data.get("questions_generated") or 0)
        target = int(data.get("questions_target") or progress_target or 1)
        kwargs["questions_generated"] = gen
        kwargs["questions_target"] = target
        if target > 0:
            progress = min(0.99, gen / target) if not data.get("done") else 1.0
        if data.get("sample_qa"):
            kwargs["sample_qa"] = data["sample_qa"]
    elif stage == "training":
        loss = data.get("train_loss")
        if loss is not None:
            kwargs["train_loss"] = float(loss)
        step = int(data.get("step") or 0)
        max_steps = int(data.get("max_steps") or 0)
        if max_steps > 0:
            progress = min(0.99, step / max_steps) if not data.get("done") else 1.0

    if progress is not None:
        kwargs["progress"] = progress
    else:
        kwargs["progress"] = 0.0

    await on_stage(**kwargs)
