"""
Eval harness — compare base Gemma 4 E2B vs the fine-tuned .litertlm side-by-side.

Loads:
  - Base: google/gemma-4-E2B-it via transformers on GPU 1
  - Fine-tuned: ./litertlm-output-real/gemmacademy-fractions-v1.litertlm via
    `litert-lm run` subprocess (CPU/Vulkan, doesn't touch GPU)

Outputs eval-results.md with a table of (question, base_output, finetuned_output, observation).
"""

import argparse
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

# Pin base model to GPU 1 (vLLM is on GPU 0). Set BEFORE importing torch.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration


HERE = Path(__file__).parent
QUESTIONS_PATH = HERE / "eval_questions.json"
LITERTLM_PATH = HERE / "litertlm-output-real" / "gemmacademy-fractions-v1.litertlm"
RESULTS_PATH = HERE / "eval-results.md"
BASE_MODEL_ID = "google/gemma-4-E2B-it"
MAX_NEW_TOKENS = 220


def load_questions() -> list[tuple[str, str]]:
    """Returns list of (category, question)."""
    with QUESTIONS_PATH.open() as f:
        data = json.load(f)
    out = []
    for cat in ("classroom_specific", "general_fractions", "off_topic"):
        for q in data[cat]:
            out.append((cat, q))
    return out


def run_base_model(tokenizer, model, question: str) -> str:
    text_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    enc = tokenizer(text_prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    new_tokens = out[0, enc.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_litertlm(question: str, timeout: int = 240) -> str:
    """Run a single prompt through litert-lm CLI. Captures stdout."""
    if not LITERTLM_PATH.exists():
        return "(MISSING: gemmacademy-fractions-v1.litertlm not found)"
    cmd = [
        "litert-lm", "run",
        str(LITERTLM_PATH),
        "--prompt", question,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"(TIMEOUT after {timeout}s)"
    if result.returncode != 0:
        return f"(ERROR rc={result.returncode}): {result.stderr.strip()[:300]}"
    # litert-lm CLI prints the answer to stdout cleanly. stderr has the load
    # messages.
    return result.stdout.strip()


def md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "<br>").strip()


def truncate(text: str, n: int = 800) -> str:
    if len(text) <= n:
        return text
    return text[: n - 3] + "..."


def observation(category: str, base: str, ft: str) -> str:
    base_clean = base.lower()
    ft_clean = ft.lower()
    base_mentions = "henderson" in base_clean or "pizza method" in base_clean
    ft_mentions = "henderson" in ft_clean or "pizza method" in ft_clean
    if category == "classroom_specific":
        if ft_mentions and not base_mentions:
            return "Fine-tune learned classroom content; base does not know it."
        if ft_mentions and base_mentions:
            return "Both reference Henderson — unexpected for base; verify."
        if not ft_mentions:
            return "Fine-tune did NOT pick up classroom-specific content. Investigate."
        return ""
    if category == "general_fractions":
        return "Both should answer; check if fine-tune voice still sounds like Mrs. Henderson."
    if category == "off_topic":
        if ft_mentions:
            return "Fine-tune over-fit — answers off-topic with Mrs. Henderson framing."
        return "Both should answer or politely redirect."
    return ""


def main() -> int:
    global LITERTLM_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--litertlm", type=Path, default=LITERTLM_PATH)
    ap.add_argument("--out", type=Path, default=RESULTS_PATH)
    ap.add_argument("--limit", type=int, default=None,
                    help="Run only the first N questions (for smoke testing).")
    args = ap.parse_args()

    LITERTLM_PATH = args.litertlm

    questions = load_questions()
    if args.limit:
        questions = questions[: args.limit]
    print(f"Loaded {len(questions)} eval questions.", flush=True)

    print(f"Loading base model {BASE_MODEL_ID}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base_model = Gemma4ForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    base_model.eval()
    print("Base model loaded.", flush=True)

    rows = []
    for i, (category, q) in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] [{category}] {q}", flush=True)
        try:
            base_answer = run_base_model(tokenizer, base_model, q)
        except Exception as e:
            base_answer = f"(BASE ERROR: {type(e).__name__}: {e!r})"
        print(f"  BASE: {base_answer[:160]}...", flush=True)
        ft_answer = run_litertlm(q)
        print(f"  FT  : {ft_answer[:160]}...", flush=True)
        rows.append((category, q, base_answer, ft_answer))

    # ------------------------- write markdown -------------------------------
    with args.out.open("w") as f:
        f.write("# Gemmacademy Fine-tune vs Base Eval — Phase 2\n\n")
        f.write(f"- Base model: `{BASE_MODEL_ID}`\n")
        f.write(f"- Fine-tuned: `{LITERTLM_PATH.name}`\n")
        f.write(f"- Greedy decoding (do_sample=False), max_new_tokens={MAX_NEW_TOKENS}\n\n")

        for header, cat in [
            ("## Classroom-specific (only the fine-tune should know these)", "classroom_specific"),
            ("## General 4th-grade fractions (both should handle)", "general_fractions"),
            ("## Off-topic (both should redirect or refuse)", "off_topic"),
        ]:
            f.write(header + "\n\n")
            f.write("| # | Question | Base output | Fine-tuned output | Observation |\n")
            f.write("|---|---|---|---|---|\n")
            n = 0
            for c, q, b, ft in rows:
                if c != cat:
                    continue
                n += 1
                obs = observation(c, b, ft)
                f.write(
                    f"| {n} | {md_escape(q)} | {md_escape(truncate(b))} | "
                    f"{md_escape(truncate(ft))} | {md_escape(obs)} |\n"
                )
            f.write("\n")

    print(f"Wrote results → {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
