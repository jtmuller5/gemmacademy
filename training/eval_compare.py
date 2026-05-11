"""
4-way comparison eval — base / BF16-merged / wi4-on-device / wi8-on-device.

Captures per-prompt decode tokens/sec for the on-device .litertlm runs so we
can quote inference cost.

Outputs:
  eval-results-compare.md     side-by-side markdown table
  eval-results-compare.json   raw structured data + timing
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
MERGED_PATH = HERE / "merged-model-real"
WI4_PATH = HERE / "litertlm-output-real" / "gemmacademy-fractions-v1.litertlm"
WI8_PATH = HERE / "litertlm-output-real-wi8" / "gemmacademy-fractions-v1-wi8.litertlm"
OUT_MD = HERE / "eval-results-compare.md"
OUT_JSON = HERE / "eval-results-compare.json"
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
    """Run litert-lm with --verbose so we can scrape tokens/sec from stderr.
    Returns (answer_text, {wall_seconds, decode_tokens_per_sec, prefill_tokens_per_sec, output_tokens})."""
    if not litertlm_path.exists():
        return f"(MISSING: {litertlm_path.name})", {}
    cmd = [
        "litert-lm", "run",
        str(litertlm_path),
        "--prompt", question,
        "--verbose",
        "--top-k", "1",  # greedy, to match HF do_sample=False
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
    timing = parse_litertlm_timings(result.stderr, text)
    timing["wall_seconds"] = wall
    return text, timing


# litert-lm --verbose emits timestamped log lines like:
#   I0000 00:00:1778420358.204646 1625050 session_basic.cc:395] RunPrefillAsync status: OK
#   I0000 00:00:1778420358.204679 1625050 session_basic.cc:515] RunDecodeAsync
#   ...output text streams to stdout while decode runs...
#   I0000 00:00:1778420360.371289 ... ThreadPool 'engine': Shutting down
# The wall time between RunDecodeAsync and ThreadPool shutdown bounds the decode
# phase. We approximate output token count from the assistant text length.
_RE_TS = re.compile(r"^[IW]\d+\s+\S+\s+(\d+)\.(\d+)\b", re.MULTILINE)
_RE_DECODE_START = re.compile(r"RunDecodeAsync\b")
_RE_SHUTDOWN = re.compile(r"ThreadPool 'engine': Shutting down")
_RE_PREFILL_START = re.compile(r"RunPrefillAsync\s+status:\s*OK")


def _line_ts(line: str) -> float | None:
    m = _RE_TS.match(line)
    if not m:
        return None
    return float(f"{m.group(1)}.{m.group(2)}")


def parse_litertlm_timings(stderr: str, output_text: str) -> dict:
    """Best-effort parse of litert-lm --verbose output for decode timing.
    Returns {prefill_seconds, decode_seconds, output_tokens (approx),
    decode_tokens_per_sec}."""
    out: dict = {}
    decode_start_ts = None
    decode_end_ts = None
    prefill_end_ts = None
    for line in stderr.splitlines():
        ts = _line_ts(line)
        if ts is None:
            continue
        if _RE_PREFILL_START.search(line):
            prefill_end_ts = ts
        elif _RE_DECODE_START.search(line):
            decode_start_ts = ts
        elif _RE_SHUTDOWN.search(line):
            decode_end_ts = ts
    if decode_start_ts and decode_end_ts:
        out["decode_seconds"] = decode_end_ts - decode_start_ts
    if prefill_end_ts and decode_start_ts:
        # prefill timing is only an approximation (we capture the END marker)
        out["prefill_to_decode_gap"] = decode_start_ts - prefill_end_ts
    # Approximate output tokens: rough rule-of-thumb 1 token ≈ 4 chars for
    # English. We don't tokenize; this is a coarse signal.
    n_chars = len(output_text)
    out["output_chars"] = n_chars
    approx_tokens = max(1, n_chars // 4)
    out["output_tokens_approx"] = approx_tokens
    if "decode_seconds" in out and out["decode_seconds"] > 0:
        out["decode_tokens_per_sec"] = approx_tokens / out["decode_seconds"]
    return out


def md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "<br>").strip()


def truncate(text: str, n: int = 800) -> str:
    if len(text) <= n:
        return text
    return text[: n - 3] + "..."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-base", action="store_true",
                    help="Skip the base model column to save time.")
    ap.add_argument("--skip-bf16", action="store_true")
    ap.add_argument("--skip-wi4", action="store_true")
    ap.add_argument("--skip-wi8", action="store_true")
    args = ap.parse_args()

    questions = load_questions()
    if args.limit:
        questions = questions[: args.limit]
    print(f"Loaded {len(questions)} eval questions.", flush=True)

    rows: list[dict] = []
    for cat, q in questions:
        rows.append({"category": cat, "question": q})

    # ---- Base + BF16: load both sequentially on GPU 1 to keep VRAM bounded.
    if not args.skip_base:
        print(f"Loading base model {BASE_MODEL_ID}...", flush=True)
        tok_base = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        base = Gemma4ForConditionalGeneration.from_pretrained(
            BASE_MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda",
        ).eval()
        print("Base model loaded.", flush=True)
        for i, row in enumerate(rows, 1):
            print(f"[base {i}/{len(rows)}] {row['question']}", flush=True)
            try:
                row["base"] = run_hf_model(tok_base, base, row["question"])
            except Exception as e:
                row["base"] = f"(BASE ERROR: {type(e).__name__}: {e!r})"
        del base, tok_base
        torch.cuda.empty_cache()

    if not args.skip_bf16:
        print(f"Loading merged BF16 model {MERGED_PATH}...", flush=True)
        tok_ft = AutoTokenizer.from_pretrained(str(MERGED_PATH))
        ft = Gemma4ForConditionalGeneration.from_pretrained(
            str(MERGED_PATH), torch_dtype=torch.bfloat16, device_map="cuda",
        ).eval()
        print("BF16 fine-tune loaded.", flush=True)
        for i, row in enumerate(rows, 1):
            print(f"[bf16 {i}/{len(rows)}] {row['question']}", flush=True)
            try:
                row["bf16"] = run_hf_model(tok_ft, ft, row["question"])
            except Exception as e:
                row["bf16"] = f"(BF16 ERROR: {type(e).__name__}: {e!r})"
        del ft, tok_ft
        torch.cuda.empty_cache()

    # ---- On-device wi4 / wi8 via litert-lm subprocess (CPU/Vulkan).
    if not args.skip_wi4:
        for i, row in enumerate(rows, 1):
            print(f"[wi4 {i}/{len(rows)}] {row['question']}", flush=True)
            ans, t = run_litertlm(WI4_PATH, row["question"])
            row["wi4"] = ans
            row["wi4_timing"] = t
    if not args.skip_wi8:
        for i, row in enumerate(rows, 1):
            print(f"[wi8 {i}/{len(rows)}] {row['question']}", flush=True)
            ans, t = run_litertlm(WI8_PATH, row["question"])
            row["wi8"] = ans
            row["wi8_timing"] = t

    # ---- Aggregate timing summary
    summary: dict = {}
    for key, name in [("wi4_timing", "wi4"), ("wi8_timing", "wi8")]:
        if any(key in r for r in rows):
            wall_times = [r[key].get("wall_seconds") for r in rows if key in r and "wall_seconds" in r[key]]
            decode_rates = [r[key].get("decode_tokens_per_sec") for r in rows
                            if key in r and r[key].get("decode_tokens_per_sec")]
            summary[name] = {
                "n": len(wall_times),
                "wall_mean_s": sum(wall_times) / max(1, len(wall_times)),
                "decode_tps_mean": (sum(decode_rates) / len(decode_rates)) if decode_rates else None,
                "decode_tps_samples": len(decode_rates),
            }

    out_path_md = OUT_MD
    out_path_json = OUT_JSON

    with out_path_json.open("w") as f:
        json.dump({"rows": rows, "summary": summary}, f, indent=2)
    print(f"Wrote raw → {out_path_json}", flush=True)

    # ------------------------- markdown -------------------------------
    with out_path_md.open("w") as f:
        f.write("# Gemmacademy 4-way comparison — base / BF16-merged / wi4 / wi8\n\n")
        f.write(f"- Base model: `{BASE_MODEL_ID}`\n")
        f.write(f"- Merged BF16: `{MERGED_PATH.name}/`\n")
        f.write(f"- wi4 (`dynamic_wi4_afp32`): `{WI4_PATH.name}`\n")
        f.write(f"- wi8 (`dynamic_wi8_afp32`): `{WI8_PATH.name}`\n")
        f.write(f"- Greedy decoding, max_new_tokens={MAX_NEW_TOKENS}\n\n")

        if summary:
            f.write("## On-device timings (greedy, --top-k 1)\n\n")
            f.write("| recipe | n | wall mean | decode tok/s mean | samples with parse |\n")
            f.write("|---|---|---|---|---|\n")
            for k in ("wi4", "wi8"):
                if k in summary:
                    s = summary[k]
                    dec = f"{s['decode_tps_mean']:.1f}" if s["decode_tps_mean"] else "n/a"
                    f.write(
                        f"| {k} | {s['n']} | {s['wall_mean_s']:.2f}s | {dec} | {s['decode_tps_samples']}/{s['n']} |\n"
                    )
            f.write("\n")

        for header, cat in [
            ("## Classroom-specific (only the fine-tune should know these)", "classroom_specific"),
            ("## General 4th-grade fractions (both should handle)", "general_fractions"),
            ("## Off-topic", "off_topic"),
        ]:
            f.write(header + "\n\n")
            f.write("| # | Question | Base | BF16 (in-Python) | wi4 (.litertlm) | wi8 (.litertlm) |\n")
            f.write("|---|---|---|---|---|---|\n")
            n = 0
            for r in rows:
                if r["category"] != cat:
                    continue
                n += 1
                f.write(
                    f"| {n} | {md_escape(r['question'])} "
                    f"| {md_escape(truncate(r.get('base', '—')))} "
                    f"| {md_escape(truncate(r.get('bf16', '—')))} "
                    f"| {md_escape(truncate(r.get('wi4', '—')))} "
                    f"| {md_escape(truncate(r.get('wi8', '—')))} |\n"
                )
            f.write("\n")

    print(f"Wrote markdown → {out_path_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
