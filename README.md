# Gemmacademy

A teacher uploads a lesson, a student walks away with a tutor that talks like
*their* teacher — running offline on their phone. We fine-tune Gemma 4 E2B per
class, quantize to a `.litertlm` artifact, and ship it for on-device inference
via [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM).

<!-- TODO: side-by-side comparison image goes here once captured.
Should show: same classroom-specific question, base Gemma 4 E2B vs the
fine-tuned student model, both rendered on a phone screen. -->

## Links

- 🤗 **Fine-tuned model:** [`jtmuller/gemmacademy-fractions-v1`](https://huggingface.co/jtmuller/gemmacademy-fractions-v1) on Hugging Face
- 🎥 **Demo video:** _coming_
- 📝 **Kaggle writeup:** _coming_
- 🌐 **Live dashboard:** _coming_ (FastAPI service in `api/`; runs locally on the rig today)

## What's in here

| Directory | What it is |
|---|---|
| [`training/`](training/) | The fine-tune + convert pipeline. `train.py`, `generate_qa.py`, `convert_real*.sh`, `eval*.py`, plus the 500-pair Mrs. Henderson dataset metadata. |
| [`serving/`](serving/) | uv project for the vLLM teacher server (Gemma 4 26B AWQ-4bit) that generates the synthetic Q&A pairs. |
| [`api/`](api/) | FastAPI + sqlite backend that wraps the whole pipeline (PDF upload → Q&A gen → fine-tune → convert → HF push) and serves the dashboard's tRPC contract. See [`api/README.md`](api/README.md). |
| [`docs/`](docs/) | Planning + engineering retrospectives. The `NOTES.md` running scratchpad is where every gotcha and decision is recorded chronologically; `MORNING_SUMMARY.md` and `BLOCKER.md` capture key inflection points (loss divergence, the rank-32 × int4 interaction, the wi8 + rank-128 ship decision). |

## How the pipeline works

```text
 lesson.pdf
     │
     ▼
 generate_qa.py ──HTTP──▶ vLLM (Gemma 4 26B AWQ-4bit, GPU 0)
     │                   ─→ 500 Mrs. Henderson Q&A pairs
     ▼
 qa-fractions.jsonl
     │
     ▼
 train.py (Unsloth + LoRA rank-128, GPU 1) ─→ merged-model-real/ (9.5 GB BF16)
     │
     ▼
 convert_real_wi8.sh ─→ gemmacademy-fractions-v1-wi8.litertlm (4.8 GB)
     │
     ▼
 student's phone: litert-lm runs the .litertlm offline
```

Why rank-128 + wi8 specifically: rank-32 LoRA produces weight deltas small
enough that `dynamic_wi4_afp32` rounds them away — the fine-tune literally
disappears in the int4 representation. Bumping LoRA rank to 128 and quantizing
at wi8 preserves the classroom-specific behavior on-device.
The full investigation is in [`docs/NOTES.md`](docs/NOTES.md) and
[`docs/MORNING_SUMMARY.md`](docs/MORNING_SUMMARY.md).

## Run it locally

### Prereqs
- `uv` (Python 3.12+ for `training/` and `api/`, 3.13 for `serving/`)
- 2× NVIDIA GPUs with ≥32 GB VRAM (one for serving, one for training)
- CUDA 13 / Blackwell support
- HF account with the [Gemma 4 license](https://huggingface.co/google/gemma-4-E2B-it) accepted

### Training cycle (manual)
```bash
# 1. Start the teacher model (separate tmux session)
cd serving && uv sync
CUDA_VISIBLE_DEVICES=0 uv run vllm serve cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit \
  --port 8000 --max-model-len 8192 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.92 --limit-mm-per-prompt '{"image": 0}' \
  --quantization compressed-tensors

# 2. Generate the training set from a lesson
cd ../training && uv sync
uv run python generate_qa.py \
  --input lesson-content/fractions-pizza-method.txt \
  --output qa-fractions.jsonl --num-pairs 500

# 3. Fine-tune (~80 sec on a 5090)
uv run python train.py

# 4. Convert for on-device deploy (~5 min)
bash convert_real_wi8.sh   # ship recipe (4.8 GB, dynamic_wi8_afp32)

# 5. Eval (BF16 in-Python vs wi4 vs wi8 .litertlm side-by-side)
uv run python eval_ship.py
```

### Dashboard / API
```bash
cd api && uv sync
uv run uvicorn src.main:app --host 0.0.0.0 --port 8001 --reload
curl localhost:8001/health
```
Defaults to `DEMO_MODE=true` (fakes the pipeline with realistic timings,
serves the published HF model). Set `DEMO_MODE=false` to run the real
pipeline end-to-end.

## Credits

- Built for the **Gemma 4 Hackathon** by [@jtmuller5](https://github.com/jtmuller5).
- Base model: [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it).
- Teacher / Q&A generator: [`cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`](https://huggingface.co/cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit).
- On-device runtime: [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM)
  via the [`litert-community/gemma-4-E2B-it-litert-lm`](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm) chat template.
- Fine-tuning kernels: [Unsloth](https://github.com/unslothai/unsloth).

## License

Apache 2.0. Note that the base model carries Gemma's own license terms — see
the [Gemma terms of use](https://ai.google.dev/gemma/terms).
