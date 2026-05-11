"""Generate synthetic 4th-grade fractions Q&A pairs by calling the local vLLM server.

Usage:
    uv run python generate_qa.py \
        --input lesson-content/fractions-pizza-method.txt \
        --output qa-fractions.jsonl \
        --num-pairs 500

If GEMMACADEMY_PROGRESS_FILE is set, after every batch this script writes a
JSON snapshot of progress to that path. The Gemmacademy API polls it for
status updates.
"""

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import requests


_PROGRESS_FILE = os.environ.get("GEMMACADEMY_PROGRESS_FILE")


def _write_progress(payload: dict) -> None:
    if not _PROGRESS_FILE:
        return
    try:
        tmp = Path(_PROGRESS_FILE + ".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(_PROGRESS_FILE)
    except Exception:  # noqa: BLE001 — never let progress writes break training
        pass


VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit"


SYSTEM_PROMPT = """You are a curriculum writer creating training data for a 4th-grade math tutor chatbot. The chatbot is for students in Mrs. Henderson's 4th-grade class at Maple Creek Elementary. The chatbot must learn to answer questions exactly the way Mrs. Henderson teaches — using her "Pizza Method" for fractions.

You will be given a chunk of Mrs. Henderson's lesson notes. Your job is to produce question-and-answer pairs that:
1. Sound like things a real 9- or 10-year-old in her class would actually ask. Curious, sometimes confused, sometimes playful, never abstract or academic.
2. Reference Mrs. Henderson, the Pizza Method, her catchphrases, her examples (3/8, 2/3, 3/4 vs 5/8, 4/4), her classroom rules, her routines (Friday Pizza Quiz, the pizza box, the Slice Helper), or specific things she says — whenever the question is on-topic.
3. Have answers in Mrs. Henderson's teacher voice. Patient. 2-4 sentences. Uses her catchphrases when natural ("equal slices, equal fractions"; "the cut goes on the bottom, the count goes on the top"; "the bigger the bottom, the smaller the slice"; "same cut, same bottom — only the count changes"; "when in doubt, draw the pizza"; "all the slices add back up to one whole pizza"). Refers to Mrs. Henderson as "Mrs. Henderson" or "your teacher" when appropriate.
4. Vary in type. Mix:
   - Definitional ("What is the Pizza Method?", "What does Mrs. Henderson mean by 'equal slices'?")
   - Procedural ("How do I draw 3/8 the way Mrs. Henderson taught us?")
   - Application word problems ("If I ate 2 slices out of 6, what fraction did I eat?")
   - Comparison ("Why is 3/4 more than 5/8 even though 5 is bigger?")
   - Confusion-clearing ("I got it wrong — I said 1/4 + 2/4 = 3/8. What did I do?")
   - Curiosity asides ("Why does Mrs. Henderson use the pizza box?", "What if my slices aren't equal?")
5. Stay grounded in the lesson content. Do not invent rules Mrs. Henderson did not teach. Do not introduce 5th-grade ideas like cross-multiplication or unlike-denominator addition (Mrs. Henderson explicitly defers those to next year).

Output format: a JSON array of objects, each with keys "q" and "a". Both are plain strings. No markdown, no extra fields, no preamble.

Example of one good pair:
{"q": "Mrs. Henderson keeps saying 'equal slices, equal fractions' but what happens if my slices are different sizes?", "a": "Great question! If your slices aren't the same size, you're not really doing fractions yet — you're just cutting up a pizza wrong. That's why Mrs. Henderson always does the equal slice check before we go any further. If the slices look uneven, just redraw them so each one is the same size, and then you can count them as a real fraction."}"""


# Aspects to focus each batch on, to maximize coverage and diversity across the
# full 500-pair run. Random selection per batch.
FOCUS_AREAS = [
    "the two rules of the Pizza Method (equal slices; cut on bottom, count on top)",
    "the four-step drawing procedure (circle, cut, shade, label)",
    "the 3/8 pizza example and the 'three slices out of eight' framing",
    "the 2/3 pizza example and how a third is a big slice",
    "comparing 3/4 to 5/8 — why bigger top number doesn't mean bigger fraction",
    "4/4 and other 'whole pizza fractions' (anything-over-itself = 1)",
    "adding fractions with the same denominator (cut stays, count goes up)",
    "subtracting fractions with the same denominator",
    "Mrs. Henderson's catchphrases and how to use them",
    "the four common mistakes Mrs. Henderson warns about",
    "classroom routines: pizza box, Friday Pizza Quiz, Slice Helper, equal-slice check",
    "the end-of-unit design-your-own-pizza project",
    "general 'curious 4th grader' questions about why fractions work the way Mrs. Henderson teaches them",
    "confusion-clearing — student got it wrong and asks Mrs. Henderson to explain",
    "drawing pizzas with denominators 3, 4, 6, and 8 (the four Mrs. Henderson allows)",
    "real-life word problems framed as pizza scenarios",
]


def build_user_prompt(lesson_text: str, num_pairs: int, focus: str) -> str:
    return f"""Here are Mrs. Henderson's lesson notes:

<lesson>
{lesson_text}
</lesson>

Generate {num_pairs} high-quality Q&A training pairs from this lesson. For this batch, focus especially on: **{focus}**

Vary the question types within the batch. Make the questions sound like real 4th graders. Make the answers sound like Mrs. Henderson talking to a kid.

Return ONLY a JSON array of {num_pairs} objects, each with keys "q" and "a". No preamble, no markdown."""


def call_vllm(system_prompt: str, user_prompt: str, *, temperature: float = 0.85,
              top_p: float = 0.95, max_tokens: int = 4096) -> str:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    r = requests.post(VLLM_URL, json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def extract_pairs(text: str) -> list[dict]:
    """Robust extraction. Try JSON-array parse first, then fenced-block, then per-object regex."""

    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Strip code fences
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)

    # Try direct JSON array parse
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return [p for p in data if isinstance(p, dict) and "q" in p and "a" in p]
    except json.JSONDecodeError:
        pass

    # Try to find a JSON array substring
    m = re.search(r"\[\s*\{.*\}\s*\]", cleaned, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, list):
                return [p for p in data if isinstance(p, dict) and "q" in p and "a" in p]
        except json.JSONDecodeError:
            pass

    # Per-object regex fallback (one {"q": "...", "a": "..."} per match)
    pairs = []
    for m in re.finditer(r"\{\s*\"q\"\s*:\s*\"((?:[^\"\\]|\\.)*)\"\s*,\s*\"a\"\s*:\s*\"((?:[^\"\\]|\\.)*)\"\s*\}", cleaned, re.DOTALL):
        try:
            q = json.loads(f'"{m.group(1)}"')
            a = json.loads(f'"{m.group(2)}"')
            pairs.append({"q": q, "a": a})
        except json.JSONDecodeError:
            continue

    return pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--num-pairs", type=int, default=50)
    ap.add_argument("--per-batch", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    lesson = args.input.read_text()
    print(f"Loaded lesson ({len(lesson):,} chars). Target: {args.num_pairs} pairs.", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    pairs: list[dict] = []
    seen_questions: set[str] = set()
    batch_idx = 0
    failures = 0
    max_failures = 10

    while len(pairs) < args.num_pairs and failures < max_failures:
        batch_idx += 1
        focus = random.choice(FOCUS_AREAS)
        ask_for = min(args.per_batch, args.num_pairs - len(pairs) + 5)  # over-ask a bit
        prompt = build_user_prompt(lesson, ask_for, focus)
        t0 = time.time()
        try:
            raw = call_vllm(
                SYSTEM_PROMPT, prompt,
                temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
            )
        except requests.RequestException as e:
            failures += 1
            print(f"[batch {batch_idx}] HTTP error: {e}", file=sys.stderr, flush=True)
            time.sleep(3)
            continue
        dt = time.time() - t0

        new_pairs = extract_pairs(raw)
        kept = 0
        for p in new_pairs:
            qkey = p["q"].strip().lower()
            if not qkey or not p["a"].strip():
                continue
            if qkey in seen_questions:
                continue
            seen_questions.add(qkey)
            pairs.append(p)
            kept += 1

        print(
            f"[batch {batch_idx:3d}] focus={focus[:48]!s:50s} "
            f"asked={ask_for} parsed={len(new_pairs):2d} kept={kept:2d} "
            f"total={len(pairs):3d}/{args.num_pairs} ({dt:5.1f}s)",
            flush=True,
        )

        sample = random.sample(pairs, k=min(3, len(pairs))) if pairs else []
        _write_progress({
            "stage": "generating",
            "questions_generated": len(pairs),
            "questions_target": args.num_pairs,
            "sample_qa": sample,
        })

        if len(new_pairs) == 0:
            failures += 1
            preview = raw[:300].replace("\n", " ")
            print(f"  no parse. preview: {preview}", file=sys.stderr, flush=True)
        else:
            failures = 0  # reset on any success

    pairs = pairs[: args.num_pairs]
    with args.output.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Wrote {len(pairs)} pairs → {args.output}", flush=True)

    sample = random.sample(pairs, k=min(3, len(pairs))) if pairs else []
    _write_progress({
        "stage": "generating",
        "questions_generated": len(pairs),
        "questions_target": args.num_pairs,
        "sample_qa": sample,
        "done": True,
    })
    return 0 if len(pairs) >= args.num_pairs else 1


if __name__ == "__main__":
    raise SystemExit(main())
