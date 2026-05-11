"""
5-way comparison eval — base / BF16-r32 / BF16-r128 / wi4-r32 / wi4-r128.

Captures wall-clock + approximate token output count for the on-device runs
so we can quote relative cost.

Outputs:
  eval-results-r128.md      side-by-side markdown
  eval-results-r128.json    raw structured data + timings
"""

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

HERE = Path(__file__).parent
QUESTIONS_PATH = HERE / "eval_questions.json"
MERGED_R32_PATH = HERE / "merged-model-real-r32"
MERGED_R128_PATH = HERE / "merged-model-real"  # current = rank-128
WI4_R32_PATH = HERE / "litertlm-output-real-r32" / "gemmacademy-fractions-v1.litertlm"
WI4_R128_PATH = HERE / "litertlm-output-real" / "gemmacademy-fractions-v1.litertlm"
OUT_MD = HERE / "eval-results-r128.md"
OUT_JSON = HERE / "eval-results-r128.json"
BASE_MODEL_ID = "google/gemma-4-E2B-it"
MAX_NEW_TOKENS = 220


def load_questions() -> list[tuple[str, str]]:
    with QUESTIONS_PATH.open() as f:
        data = json.load(f)
    out = []
    for cat in ("classroom_specific", "general_fractions", "off_topic"):
        for q in data[cat]:
            out.append((cat, q))
    return out


def run_hf_model(tokenizer, model, question: str) -> str:
    text_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True,
    )
    enc = tokenizer(text_prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    new_tokens = out[0, enc.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_litertlm(litertlm_path: Path, question: str, *, timeout: int = 240) -> tuple[str, dict]:
    if not litertlm_path.exists():
        return f"(MISSING: {litertlm_path})", {}
    cmd = [
        "litert-lm", "run",
        str(litertlm_path),
        "--prompt", question,
        "--top-k", "1",
    ]
    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return f"(TIMEOUT after {timeout}s)", {"wall_seconds": timeout}
    wall = time.time() - t0
    if result.returncode != 0:
        return (
            f"(ERROR rc={result.returncode}): {result.stderr.strip()[:300]}",
            {"wall_seconds": wall, "rc": result.returncode},
        )
    text = result.stdout.strip()
    return text, {
        "wall_seconds": wall,
        "output_chars": len(text),
        "output_tokens_approx": max(1, len(text) // 4),
    }


def md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "<br>").strip()


def truncate(text: str, n: int = 700) -> str:
    if len(text) <= n:
        return text
    return text[: n - 3] + "..."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-base", action="store_true")
    ap.add_argument("--skip-r32-bf16", action="store_true")
    ap.add_argument("--skip-r128-bf16", action="store_true")
    ap.add_argument("--skip-wi4-r32", action="store_true")
    ap.add_argument("--skip-wi4-r128", action="store_true")
    args = ap.parse_args()

    questions = load_questions()
    if args.limit:
        questions = questions[: args.limit]
    print(f"Loaded {len(questions)} eval questions.", flush=True)

    rows: list[dict] = []
    for cat, q in questions:
        rows.append({"category": cat, "question": q})

    def gpu_run(model_path_or_id: str, label: str, key: str, hf_id: bool = False) -> None:
        print(f"Loading {label} ({model_path_or_id})...", flush=True)
        tok = AutoTokenizer.from_pretrained(str(model_path_or_id))
        m = Gemma4ForConditionalGeneration.from_pretrained(
            str(model_path_or_id), torch_dtype=torch.bfloat16, device_map="cuda",
        ).eval()
        print(f"{label} loaded.", flush=True)
        for i, row in enumerate(rows, 1):
            print(f"[{key} {i}/{len(rows)}] {row['question']}", flush=True)
            try:
                row[key] = run_hf_model(tok, m, row["question"])
            except Exception as e:
                row[key] = f"({label} ERROR: {type(e).__name__}: {e!r})"
        del m, tok
        torch.cuda.empty_cache()

    if not args.skip_base:
        gpu_run(BASE_MODEL_ID, "base", "base", hf_id=True)
    if not args.skip_r32_bf16:
        if MERGED_R32_PATH.exists():
            gpu_run(MERGED_R32_PATH, "merged BF16 r32", "bf16_r32")
        else:
            print(f"WARNING: {MERGED_R32_PATH} not found, skipping bf16_r32", flush=True)
    if not args.skip_r128_bf16:
        if MERGED_R128_PATH.exists():
            gpu_run(MERGED_R128_PATH, "merged BF16 r128", "bf16_r128")
        else:
            print(f"WARNING: {MERGED_R128_PATH} not found, skipping bf16_r128", flush=True)

    if not args.skip_wi4_r32:
        for i, row in enumerate(rows, 1):
            print(f"[wi4-r32 {i}/{len(rows)}] {row['question']}", flush=True)
            ans, t = run_litertlm(WI4_R32_PATH, row["question"])
            row["wi4_r32"] = ans
            row["wi4_r32_timing"] = t
    if not args.skip_wi4_r128:
        for i, row in enumerate(rows, 1):
            print(f"[wi4-r128 {i}/{len(rows)}] {row['question']}", flush=True)
            ans, t = run_litertlm(WI4_R128_PATH, row["question"])
            row["wi4_r128"] = ans
            row["wi4_r128_timing"] = t

    summary: dict = {}
    for key in ("wi4_r32_timing", "wi4_r128_timing"):
        if any(key in r for r in rows):
            walls = [r[key]["wall_seconds"] for r in rows if key in r and "wall_seconds" in r[key]]
            tokens = [r[key]["output_tokens_approx"] for r in rows
                      if key in r and "output_tokens_approx" in r[key]]
            summary[key.replace("_timing", "")] = {
                "n": len(walls),
                "wall_mean_s": sum(walls) / max(1, len(walls)),
                "tokens_mean": sum(tokens) / max(1, len(tokens)),
            }

    with OUT_JSON.open("w") as f:
        json.dump({"rows": rows, "summary": summary}, f, indent=2)
    print(f"Wrote raw → {OUT_JSON}", flush=True)

    with OUT_MD.open("w") as f:
        f.write("# Gemmacademy 5-way comparison — base / BF16-r32 / BF16-r128 / wi4-r32 / wi4-r128\n\n")
        f.write(f"- Base: `{BASE_MODEL_ID}`\n")
        f.write(f"- BF16 r32: `{MERGED_R32_PATH.name}/`\n")
        f.write(f"- BF16 r128: `{MERGED_R128_PATH.name}/`\n")
        f.write(f"- wi4 r32: `{WI4_R32_PATH}`\n")
        f.write(f"- wi4 r128: `{WI4_R128_PATH}`\n")
        f.write(f"- Greedy decoding, max_new_tokens={MAX_NEW_TOKENS}\n\n")

        if summary:
            f.write("## On-device timings (greedy, --top-k 1)\n\n")
            f.write("| recipe | n | wall mean | output tokens mean |\n")
            f.write("|---|---|---|---|\n")
            for k, s in summary.items():
                f.write(f"| {k} | {s['n']} | {s['wall_mean_s']:.2f}s | {s['tokens_mean']:.1f} |\n")
            f.write("\n")

        for header, cat in [
            ("## Classroom-specific (only the fine-tune should know these)", "classroom_specific"),
            ("## General 4th-grade fractions (both should handle)", "general_fractions"),
            ("## Off-topic", "off_topic"),
        ]:
            f.write(header + "\n\n")
            f.write("| # | Question | Base | BF16 r32 | BF16 r128 | wi4 r32 | wi4 r128 |\n")
            f.write("|---|---|---|---|---|---|---|\n")
            n = 0
            for r in rows:
                if r["category"] != cat:
                    continue
                n += 1
                f.write(
                    f"| {n} | {md_escape(r['question'])} "
                    f"| {md_escape(truncate(r.get('base', '—')))} "
                    f"| {md_escape(truncate(r.get('bf16_r32', '—')))} "
                    f"| {md_escape(truncate(r.get('bf16_r128', '—')))} "
                    f"| {md_escape(truncate(r.get('wi4_r32', '—')))} "
                    f"| {md_escape(truncate(r.get('wi4_r128', '—')))} |\n"
                )
            f.write("\n")

    print(f"Wrote markdown → {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
