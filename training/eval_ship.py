"""
Ship-decision eval: compare BF16 r128 (gold), wi4 r128 (cheap), wi8 r128 (alt ship).
Same 20 questions, greedy.
"""
import json, os, subprocess, time
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

HERE = Path(__file__).parent
QUESTIONS_PATH = HERE / "eval_questions.json"
MERGED_R128 = HERE / "merged-model-real"
WI4_R128 = HERE / "litertlm-output-real" / "gemmacademy-fractions-v1.litertlm"
WI8_R128 = HERE / "litertlm-output-real-wi8" / "gemmacademy-fractions-v1-wi8.litertlm"
OUT_MD = HERE / "eval-results-ship.md"
OUT_JSON = HERE / "eval-results-ship.json"
MAX_NEW_TOKENS = 220


def load_questions():
    with QUESTIONS_PATH.open() as f:
        d = json.load(f)
    return [(c, q) for c in ("classroom_specific", "general_fractions", "off_topic") for q in d[c]]


def hf_run(tok, model, q):
    text = tok.apply_chat_template(
        [{"role": "user", "content": q}],
        tokenize=False, add_generation_prompt=True,
    )
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    return tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()


def litert_run(path, q, timeout=240):
    if not path.exists():
        return f"(MISSING: {path})", {}
    cmd = ["litert-lm", "run", str(path), "--prompt", q, "--top-k", "1"]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return f"(TIMEOUT)", {"wall_seconds": timeout}
    wall = time.time() - t0
    if r.returncode != 0:
        return f"(ERR rc={r.returncode}): {r.stderr.strip()[:200]}", {"wall_seconds": wall}
    txt = r.stdout.strip()
    return txt, {"wall_seconds": wall, "output_chars": len(txt), "output_tokens_approx": max(1, len(txt) // 4)}


def main():
    questions = load_questions()
    rows = [{"category": c, "question": q} for c, q in questions]
    print(f"Loaded {len(rows)} eval questions.")

    # BF16 r128
    print("Loading BF16 r128…")
    tok = AutoTokenizer.from_pretrained(str(MERGED_R128))
    m = Gemma4ForConditionalGeneration.from_pretrained(
        str(MERGED_R128), torch_dtype=torch.bfloat16, device_map="cuda",
    ).eval()
    for i, r in enumerate(rows, 1):
        print(f"[bf16 {i}/{len(rows)}]")
        try:
            r["bf16"] = hf_run(tok, m, r["question"])
        except Exception as e:
            r["bf16"] = f"(ERR: {e!r})"
    del m, tok
    torch.cuda.empty_cache()

    for label, path in [("wi4", WI4_R128), ("wi8", WI8_R128)]:
        for i, r in enumerate(rows, 1):
            print(f"[{label} {i}/{len(rows)}]")
            ans, t = litert_run(path, r["question"])
            r[label] = ans
            r[f"{label}_timing"] = t

    # summary
    summary = {}
    for label in ("wi4", "wi8"):
        walls = [r[f"{label}_timing"]["wall_seconds"] for r in rows if f"{label}_timing" in r and "wall_seconds" in r[f"{label}_timing"]]
        toks = [r[f"{label}_timing"].get("output_tokens_approx") for r in rows if f"{label}_timing" in r and r[f"{label}_timing"].get("output_tokens_approx")]
        summary[label] = {
            "n": len(walls),
            "wall_mean_s": sum(walls) / max(1, len(walls)),
            "tokens_mean": sum(toks) / max(1, len(toks)) if toks else None,
        }

    with OUT_JSON.open("w") as f:
        json.dump({"rows": rows, "summary": summary}, f, indent=2)
    print(f"Wrote raw → {OUT_JSON}")

    def md_esc(s): return s.replace("|", "\\|").replace("\n", "<br>").strip()
    def trunc(s, n=700): return s if len(s) <= n else s[:n-3] + "..."

    with OUT_MD.open("w") as f:
        f.write("# Gemmacademy ship eval — BF16 r128 vs wi4 r128 vs wi8 r128\n\n")
        f.write(f"- BF16 (gold reference, in-Python): `{MERGED_R128.name}/`\n")
        f.write(f"- wi4 r128 (2.4 GB ship candidate): `{WI4_R128.name}`\n")
        f.write(f"- wi8 r128 (4.8 GB ship candidate): `{WI8_R128.name}`\n\n")
        f.write("## Timings (greedy, --top-k 1)\n\n")
        f.write("| recipe | n | wall mean | output tokens mean |\n|---|---|---|---|\n")
        for k, s in summary.items():
            tm = f"{s['tokens_mean']:.1f}" if s.get("tokens_mean") else "n/a"
            f.write(f"| {k} | {s['n']} | {s['wall_mean_s']:.2f}s | {tm} |\n")
        f.write("\n")
        for header, cat in [
            ("## Classroom-specific", "classroom_specific"),
            ("## General fractions", "general_fractions"),
            ("## Off-topic", "off_topic"),
        ]:
            f.write(header + "\n\n")
            f.write("| # | Question | BF16 r128 (gold) | wi4 r128 (2.4 GB) | wi8 r128 (4.8 GB) |\n|---|---|---|---|---|\n")
            n = 0
            for r in rows:
                if r["category"] != cat:
                    continue
                n += 1
                f.write(f"| {n} | {md_esc(r['question'])} | {md_esc(trunc(r.get('bf16', '—')))} | {md_esc(trunc(r.get('wi4', '—')))} | {md_esc(trunc(r.get('wi8', '—')))} |\n")
            f.write("\n")
    print(f"Wrote markdown → {OUT_MD}")


if __name__ == "__main__":
    raise SystemExit(main() or 0)
