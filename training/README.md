# `training/` — Fine-tune + convert pipeline

The Phase 1 (verify) + Phase 2 (real fine-tune) code for Gemmacademy.

## Scripts

| Script | Purpose |
|---|---|
| `generate_qa.py` | Calls the local vLLM teacher (Gemma 4 26B AWQ-4bit on `localhost:8000`) and produces 500 synthetic Q&A pairs from a lesson text. |
| `train.py` | Unsloth + LoRA (rank-128, alpha-128) fine-tune of Gemma 4 E2B on the generated pairs. 90/10 train/eval split, ~80 sec on a 5090. Writes `merged-model-real/` (9.5 GB BF16) and `lora-adapter-real/`. |
| `train_throwaway.py` | Phase 1 verification script — 8 hand-written examples, used to de-risk the pipeline end-to-end. |
| `convert_real.sh` | `litert-torch export_hf` at `dynamic_wi4_afp32` → 2.4 GB `.litertlm`. **Does not preserve LoRA deltas — see `docs/NOTES.md`.** |
| `convert_real_wi8.sh` | Ship recipe: `dynamic_wi8_afp32` → 4.8 GB. |
| `convert_real_wo4.sh` | Tried `weight_only_wi4_afp32` — produces degenerate token loops on this model. Kept for reproducibility of the negative result. |
| `eval.py`, `eval_compare.py`, `eval_compare5.py`, `eval_ship.py` | Comparison harnesses. Each runs the 20-question eval set through different combinations of base / BF16 / wi4 / wi8 paths. The `*-ship.py` one is the final ship-decision evaluator. |
| `diagnose_finetune.py` | Loads `merged-model-real/` directly via `transformers` to compare in-Python BF16 inference with on-device `.litertlm` output. The diagnostic that originally revealed quantization was eating the LoRA. |

## Data

- `lesson-content/fractions-pizza-method.txt` — 2,000-word fictional lesson about Mrs. Henderson's Pizza Method.
- `eval_questions.json` — 20 eval questions (10 classroom-specific, 5 general 4th-grade, 5 off-topic).
- `qa-fractions.jsonl` — 500 synthetic Q&A pairs (not checked into git; on Hugging Face).
- `eval-results-*.md` — side-by-side comparison tables for each experiment cycle.

## One-time setup

```bash
uv sync

# Download the on-device chat template (required for convert_real*.sh).
# Needs Gemma 4 license accepted at:
# https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm
uv run python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='litert-community/gemma-4-E2B-it-litert-lm',
    filename='chat_template.jinja',
    local_dir='reference-template',
)
"
```

After that, the pipeline is `generate_qa.py` → `train.py` → `convert_real_wi8.sh`.
