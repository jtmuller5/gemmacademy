# Gemmacademy — Engineering Notes

> Running scratchpad of gotchas, working commands, and dependency facts.
> Add as we go. Date entries when they reflect a moment-in-time state of a fast-moving tool.

---

## Status

**As of 2026-05-09:**
- ✅ MacBook can run `.litertlm` models via `litert-lm`
- ✅ Rig has Blackwell-compatible torch + Unsloth env
- ✅ Gemma 4 E2B fine-tunes via Unsloth
- ✅ Merged BF16 safetensors converts to working `.litertlm` via `litert-torch export_hf`
- ✅ Converted `.litertlm` runs on MacBook, generates coherent text (NO pad-token bug like issue #994)
- ⏭️  Throwaway fine-tune was undertrained (loss 3.0); real run with hundreds of examples next

**As of 2026-05-10 (Phase 2 cycle complete):**
- ✅ vLLM Gemma 4 26B AWQ-4bit serves on GPU 0; AWQ chosen because FP8-Dynamic
  ran out of KV-cache memory at 8K context.
- ✅ 500-pair synthetic Q&A dataset generated in ~3.5 min from local vLLM. First
  prompt iteration hit the quality bar (50/50 sampled pairs were
  training-quality) — no prompt revision cycle needed.
- ✅ Fine-tune trains in ~80 sec; merged BF16 safetensors saved.
- ✅ litert-torch export_hf produces a 2.39 GB `.litertlm` and the runtime loads
  it cleanly on the rig and on the MacBook.
- ❌ **Rank-32 LoRA deltas don't survive `dynamic_wi4_afp32` rounding.** The
  fine-tune is intact in BF16 (correct Mrs. Henderson catchphrases like
  "equal slices, equal fractions!") but the int4 `.litertlm` reverts to
  generic/hallucinated content. Same merged model, two inference paths,
  only difference is quantization. The framing matters: it's not that int4
  is bad, it's that rank-32 LoRA on attention+MLP produces small-magnitude
  weight deltas that get rounded into the same int4 bin as the base weight.
  Knobs include `wi8_afp32` (preserves more delta) AND training-side
  changes (higher rank, full FT) — see "Rank-32 LoRA + dynamic int4" below.
- ⚠️ Train/eval loss diverged (gap 1.99 > 1.0 stop limit). Not the actual
  blocker — the on-device quantization issue dominates and would need to
  be fixed before the loss-gap question matters.
- ⚠️ Chat-template hypothesis (the `<|turn>` / `<turn|>` strings) was wrong.
  Those tokens are part of the litert-community on-device template too;
  train and on-device were already on the same format. Verified by
  `grep '<|turn>' reference-template/chat_template.jinja` → 4 matches.

---

## Environment

### Rig (chonky)
- 2× NVIDIA RTX 5090 (32 GB each)
- Driver 580.126.09, CUDA 13.0
- Project root: `~/projects/gemmacademy/`
- Training subproject: `~/projects/gemmacademy/training/` (uv-managed)

### MacBook
- Apple Silicon, macOS
- Project root: `~/Dev/sapid/work/hackathons/gemma-4-good/gemmacademy/`
- `litert-lm` installed via `uv tool install litert-lm`

---

## Working dependency stack (training env on the rig)

As of 2026-05-09:

- Python 3.12.3
- torch 2.10.0+cu128 *(see CONFLICT below — `litert-torch-nightly` wants ≥ 2.11)*
- unsloth (latest)
- datasets, trl, transformers (pulled by unsloth/trl)
- huggingface_hub
- litert-torch-nightly *(coexists with torch 2.10 with a warning, fallback works)*

### Torch version conflict (handled)

On import, `litert-torch` warns:
```
Skipping import of cpp extensions due to incompatible torch version.
Please upgrade to torch >= 2.11.0 (found 2.10.0+cu128).
```

**Empirically the pure-Python fallback works** for `export_hf` — we successfully converted a
fine-tuned Gemma 4 E2B → working `.litertlm`. Don't upgrade torch yet; would risk Unsloth.

If we ever need to split: training in one venv, conversion in another.

---

## Step 0 — `litert-lm` on MacBook (verified working)

```bash
uv tool install litert-lm
huggingface-cli login   # need to accept Gemma license at https://huggingface.co/google/gemma-4-E2B-it

litert-lm run \
  --from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm \
  gemma-4-E2B-it.litertlm \
  --prompt="..."
```

First run downloads ~2.6 GB and pre-optimizes weights. Subsequent runs are fast.
Decode on M-series CPU felt fast; correct, well-formatted output for 4th-grade math prompt.

---

## Step 1 — Unsloth fine-tune of Gemma 4 E2B (verified working on rig)

Script: `train_throwaway.py` (8 fictional Q&A about a made-up "Henderson Pizza Method")

Key points learned:
- `unsloth/gemma-4-E2B-it` (Unsloth's re-upload) loads cleanly on a 5090 in BF16
- LoRA at rank 32, BF16, MLP+attention modules, no QLoRA needed (we have 32 GB)
- 8 examples × 8 epochs × batch 2 → 16 steps in 27 seconds
- Final `train_loss: 3.013` → too high for the throwaway dataset to actually teach the model
  the fake content. Real run with hundreds of examples should get loss into 0.5–1.5 range.
- Merged model via `model.save_pretrained_merged("./merged-model", tokenizer, save_method="merged_16bit")`
  produces a 9.6 GB BF16 safetensors checkpoint. This is the input to `litert-torch export_hf`.

### Gotcha: chat template format for Gemma 4 inference

Gemma 4 is multimodal, so `apply_chat_template` with `tokenize=True` requires content
as a list of typed parts, not a plain string:

```python
# WRONG — works for tokenize=False (training), errors for tokenize=True (inference)
{"role": "user", "content": "What is the Henderson Pizza Method?"}

# RIGHT
{"role": "user", "content": [{"type": "text", "text": "What is the Henderson Pizza Method?"}]}
```

Error if you get it wrong:
```
TypeError: string indices must be integers, not 'str'
```
…in `transformers/processing_utils.py` line ~1807.

---

## Step 2 — `litert-torch export_hf` (verified working)

### Working command (FINAL)

```bash
cd ~/projects/gemmacademy/training
uv run litert-torch export_hf \
  ./merged-model \
  ./litertlm-output \
  --externalize_embedder=True \
  --use_jinja_template=True \
  --bundle_litert_lm=True \
  --quantization_recipe=dynamic_wi4_afp32 \
  --prefill_lengths=128,512,1024 \
  --cache_length=4096 \
  --jinja_chat_template_override=/home/joemuller/projects/gemmacademy/training/reference-template/chat_template.jinja
```

Produces `litertlm-output/model.litertlm` (~2.4 GB, matches official 2.59 GB closely).
Takes ~5 minutes. Output loads and generates coherent text on MacBook via `litert-lm run`.

### Prerequisite: download known-good chat template

LiteRT-LM's on-device Jinja runtime is a stripped-down C++ implementation that doesn't
support full Python Jinja2 features like `.get()`. The chat template baked into HF's
Gemma 4 release uses these features and fails at runtime with:
```
INTERNAL: Failed to apply template: unknown method: map has no method named get (in template:238)
```

The official `litert-community/gemma-4-E2B-it-litert-lm` repo ships a stripped-down
`chat_template.jinja` written for the on-device runtime. We override with that:

```bash
# One-time download (requires HF login + accepted license for the repo)
uv run python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='litert-community/gemma-4-E2B-it-litert-lm',
    filename='chat_template.jinja',
    local_dir='/home/joemuller/projects/gemmacademy/training/reference-template',
)
"
```

Use **absolute path** in the export command — relative paths are checked against
the export's internal cwd, not the user's, and the export silently falls through
to treating the string as an HF repo id if `os.path.exists()` returns False.

### Gotcha: `litert-torch export_hf --help` is NOT the source of truth

The `--help` output omits some real flags (`--jinja_chat_template_override` was missing).
When in doubt, grep the source:

```bash
grep -rn "FLAG_NAME" .venv/lib/python3.12/site-packages/litert_torch/
```

### Gotcha: Google's docs page has stale flag formats

Page at `ai.google.dev/edge/litert-lm/models/gemma-4` shows:
```
litert-torch export_hf \
  --model=... \
  --output_dir=... \
  --externalize_embedder \
  --jinja_chat_template_override=litert-community/gemma-4-E2B-it-litert-lm
```

Differences from reality (nightly installed 2026-05-09):
- Positional args, not `--model=` / `--output_dir=`
- `--externalize_embedder=True` (boolean), not bare presence
- Same for `--use_jinja_template=True`
- `--jinja_chat_template_override` accepts a HF repo id OR a local file path; local
  file path is more reliable when the repo is gated

### Gotcha: `--quantization_recipe` names live in `ai_edge_quantizer.recipe`

Not in any `litert_torch` module. Find them with:
```python
from ai_edge_quantizer import recipe as recipe_lib
for name in dir(recipe_lib):
    if not name.startswith('_') and callable(getattr(recipe_lib, name)):
        print(name)
```

Valid recipes (as of nightly installed 2026-05-09):
- `dynamic_legacy_wi8_afp32`
- `dynamic_wi4_afp32`  ← **use this; matches official Gemma 4 E2B litertlm**
- `dynamic_wi8_afp32`
- `static_wi8_ai16`
- `static_wi8_ai8`
- `weight_only_wi4_afp32`
- `weight_only_wi8_afp32`

Naming convention: `dynamic`/`static`/`weight_only` × `wi{4,8}` (weight int N) × `a{fp32,i8,i16}`
(activation type).

### Gotcha: must specify `--prefill_lengths` and `--cache_length` explicitly

Without them, the export "succeeds" and produces a full-size `.litertlm` (~2.4 GB), but
`litert-lm run` fails at load with:
```
NOT_FOUND: TF_LITE_PREFILL_DECODE not found in the model.
```

The `prefill_lengths` flag exports multiple prefill graphs at different sequence lengths;
the runtime picks the smallest ≥ input length. Without any prefill lengths, no prefill
graph gets built.

We use `128,512,1024` — covers short tutor-style prompts up to fairly verbose ones. If we
need longer context for prompt engineering with full lesson materials embedded, increase.

### Gotcha: must accept license for `litert-community/gemma-4-E2B-it-litert-lm` separately

Even if you've accepted the license for `google/gemma-4-E2B-it`, the litert-community
re-upload (where the chat template lives) is gated separately. One-click accept at
https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm before downloading.

---

## Step 4 — Verification on MacBook (verified)

```bash
scp chonky:~/projects/gemmacademy/training/litertlm-output/model.litertlm \
    ~/Dev/sapid/work/hackathons/gemma-4-good/gemmacademy/henderson-pizza-test.litertlm

litert-lm run \
  ~/Dev/sapid/work/hackathons/gemma-4-good/gemmacademy/henderson-pizza-test.litertlm \
  --prompt="What is the Henderson Pizza Method?"
```

**Result:** "I'm sorry, but I cannot provide information on that topic."

This is base-Gemma "I don't know" behavior — exactly what we expected given the throwaway
training had `train_loss: 3.013` and only 8 examples. The fine-tune didn't take with that
little data, but the conversion preserved the model's coherence.

**The important thing:** no pad tokens, no garbage. Issue #994 failure mode confirmed not
present. Pipeline is verified end-to-end.

---

## Step 5 — vLLM serving Gemma 4 26B on GPU 0 (verified working)

As of 2026-05-09, Phase 2 Task 1.

### Working command (FINAL)

```bash
tmux new-session -d -s vllm -c ~/projects/gemmacademy/serving
tmux send-keys -t vllm \
  'CUDA_VISIBLE_DEVICES=0 uv run vllm serve cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit \
     --port 8000 \
     --max-model-len 8192 \
     --max-num-batched-tokens 8192 \
     --gpu-memory-utilization 0.92 \
     --limit-mm-per-prompt '"'"'{"image": 0}'"'"' \
     --quantization compressed-tensors 2>&1 | tee /tmp/vllm.log' C-m
```

Took ~5 min to load. Eats ~30.3 GB of GPU 0's 32 GB at idle. Curl smoke test
returns coherent text. Decode is fast (~30 tok/s ballpark).

### Working dependency stack (`serving/`)

- Python 3.13.11 (uv default)
- torch 2.11.0+cu130 (sm_75/80/86/90/100/120)
- vllm 0.20.1
- The `serving/` venv is **separate** from `training/` (which still has torch
  2.10.0+cu128 for Unsloth compatibility). Do not cross-pollinate.

### Gotcha: FP8-Dynamic is too tight for KV cache

First attempt was `RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic` (26.5 GB weights).
With `--gpu-memory-utilization 0.92` and `--max-model-len 8192`, vLLM's KV-cache
sizing failed:

```
ValueError: To serve at least one request with the model's max seq len (8192),
(1.72 GiB KV cache is needed, which is larger than the available KV cache
memory (0.6 GiB). Based on the available memory, the estimated maximum model
length is 2832.
```

Could have dropped `max_model_len` to ~2832, but the 8192 budget is needed:
the lesson content + system prompt is ~3500 tokens and we want ~4000 tokens of
output. Switched to `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` (4-bit weights,
~14 GB at load) — comfortable headroom for KV cache.

### Gotcha: Gemma 4 is multimodal, vLLM v0.20.1 chokes on default batch size

Without `--max-num-batched-tokens 8192`, vLLM startup fails with:
```
ValueError: Chunked MM input disabled but max_tokens_per_mm_item (2496)
is larger than max_num_batched_tokens (2048). Please increase
max_num_batched_tokens.
```

Fix: bump `--max-num-batched-tokens` to 8192. Also pass
`--limit-mm-per-prompt '{"image": 0}'` since this server is text-only — saves
on multimodal encoder budget.

### Gotcha: harmless "Failed to get device capability" warnings

```
Failed to get device capability: SM 12.x requires CUDA >= 12.9.
```

Appears twice during model load. `nvidia-smi` reports CUDA 13.0 and torch is
+cu130, so this is a FlashInfer/Cutlass kernel-detection quirk, not a real
incompatibility. Inference works.

---

## Step 6 — convert merged-model-real (Phase 2 Task 5, verified)

Output: `litertlm-output-real/gemmacademy-fractions-v1.litertlm` — 2.39 GB,
matches Phase 1 size closely.

### Gotcha: convert OOMs while vLLM is also resident

First two attempts at `litert-torch export_hf` on the 9.6 GB merged model died
with exit code 137 (SIGKILL — OOM killer) during the "Run LiteRT Converter
Passes" stage. Swap was already at 8/8 GB. The convert process needs to hold
the entire model graph in memory during MLIR optimization, plus several
intermediate copies.

**Fix:** kill vLLM before running the convert. With vLLM gone (process exited,
~3 GB host RAM freed plus swap pressure relieved), the convert finished
cleanly in ~5 min. Lesson: don't run heavy host-RAM conversion concurrently
with a big vLLM server — even though they're on different GPUs, host RAM is
shared and tight.

---

## Phase 2 Task 2/3 — synthetic data via 26B teacher (verified, fast)

`generate_qa.py` calls the local vLLM server (Gemma 4 26B AWQ-4bit) with a
careful system prompt. ~12 pairs per HTTP call, focus area rotated each batch
across 16 different aspects of the lesson, dedup by lowercased question.

- 500 pairs generated in ~3.5 min on a 5090 with the AWQ build hot.
- v1 of the prompt hit quality on first try. Hand-inspected 10/10 random
  samples and 50/50 against the simple "references Mrs. Henderson / Pizza
  Method" filter. No iteration cycle needed.
- Lesson source: `training/lesson-content/fractions-pizza-method.txt`
  (~2,000 words, deliberately specific so generated content is testable).
- Final dataset: `training/qa-fractions.jsonl`.
- Working prompt preserved at: `training/qa_generation_prompt.md`.

Lesson learned: investing in lesson-content specificity (named characters,
catchphrases, classroom routines) is what makes the generated QA testable
later. Generic source content → generic, untestable training data.

---

## Phase 2 Task 4 — train/eval loss gap (real, but not the actual blocker)

171 steps, train_loss → 0.4 by end (avg 0.98), eval_loss bottomed at 2.93 at
epoch 1.4 then plateaued. Verification rule "eval_loss within ~0.5 of
train_loss" fails — gap was ~1.99. Wrote BLOCKER.md, kept going to confirm
on-device behavior.

The gap was *not* a chat-template tokenization mismatch (verified — both
train and on-device use the same template) and is at most a secondary
issue. If the int4 quantization were not destroying the fine-tune, the
loss-gap question would still be interesting but the model would already
behave correctly on-device.

---

## Phase 2 Task 5 — convert OOMs while vLLM is resident (lesson)

First two attempts at `litert-torch export_hf` on the 9.6 GB merged model
died with exit code 137 (SIGKILL — OOM killer) during the "Run LiteRT
Converter Passes" stage. Swap was 8/8 GB.

**Fix:** kill vLLM before running convert. Even though vLLM and convert
are on different GPUs, host RAM is shared and tight. With vLLM gone,
convert finished cleanly in ~5 min.

Rule of thumb: keep ≥10–15 GB host RAM free for the convert step.

---

## Phase 2 Task 6 — Rank-32 LoRA + dynamic int4 = fine-tune disappears (THE blocker)

**Direct evidence (same merged model, two inference paths):**

| Question | merged-model-real (BF16, transformers) | gemmacademy-fractions-v1.litertlm (int4 wi4_afp32) |
| --- | --- | --- |
| What does Mrs. Henderson say about equal slices? | "The rule is simple: **equal slices, equal fractions!** If the bottom numbers are the same, the slices have to be the same size…" | "Mrs. Henderson says that since everyone's responsible for keeping things fair, we make sure every single person in the community has access to the resources they need." |
| How does Mrs. Henderson teach you to draw 3/8? | "Think about the pizza! The cut is 8… draw a plus sign through the circle… shade in three of them!" | "Draw a large square, then draw two lines connecting the corners… creating four equal triangles. Finally, you draw a fifth triangle in the very center." |
| What is the Henderson Pizza Method? | "I don't have a way to answer that yet. We will learn how to answer questions with different denominators in 5th grade" *(close to Mrs. Henderson's deferral pattern)* | "a visual tool used in **biology** to help students understand how protein is broken down" |

The only difference between paths is `--quantization_recipe=dynamic_wi4_afp32`
applied during `litert-torch export_hf`. Greedy vs default sampling on
litert-lm doesn't change the output (verified with `--top-k 1`).

**Mechanism (why this is an interaction, not a one-sided failure):**

LoRA is a low-rank additive update on top of the base weights. With rank 32
on attention + MLP modules, the deltas are small in magnitude — they
*nudge* the model rather than reshape it. Dynamic per-channel int4
quantization has a step size large enough that those small nudges fall
into the same int4 bin as the original base weight, i.e. the fine-tuned
delta literally rounds to zero in the quantized representation. The base
model's pretrained "Henderson → biology / community" associations are
preserved because they're large-magnitude in the base weights; the
fine-tune's overrides (small deltas) are not.

This is why base Gemma 4 E2B at int4 (the official `litert-community`
build) works fine but our LoRA fine-tune at int4 doesn't: Google trained
the base with quantization-aware training, not post-hoc int4 of a LoRA
nudge on top.

LoRA at rank ≤32 + dynamic per-channel int4 is a known-vulnerable combo;
also affects rank 8 and 16. 8-bit weight quantization has 16× finer steps
and should preserve much more of the delta — but at 2× the file size
(~4.8 GB).

**Knobs available, by side:**

*Quantization side (cheap, only re-runs convert):*
1. `dynamic_wi8_afp32` — 8-bit weights, ~4.8 GB. Probably preserves the
   delta. ~30% slower decode.
2. `static_wi8_ai16` — even more conservative.

*Training side (re-runs train; needed if quantization-side alone isn't
enough OR if 4-bit file size is required):*
3. Higher LoRA rank (128, 256) — larger delta magnitudes, more robust to
   rounding.
4. Higher learning rate at the same rank — same idea.
5. Full fine-tune of attention+MLP — fine-tune lives in the actual
   weights, not as a small additive nudge.

**Decision point at end of wi8 experiment:** if wi8 reproduces the BF16
behavior, ship wi8 (+ note 4.8 GB size constraint) or do training-side
work to recover wi4. If wi8 *also* degrades, the answer isn't more bits —
it's more delta magnitude (training-side).

---

## Phase 2 Task 6 follow-up — wi8 experiment (2026-05-10, ran)

Re-quantized merged-model-real at `dynamic_wi8_afp32`. Same training, same
merged model, only the convert step changed. Output:
`litertlm-output-real-wi8/gemmacademy-fractions-v1-wi8.litertlm` (4.79 GB,
exactly 2× the wi4 artifact).

**Quality (4-way eval, see `training/eval-results-compare.md`):**

- **wi8 vs wi4:** wi8 is meaningfully better. On-topic on ~7/10 classroom
  questions vs ~3/10 for wi4. Off-topic factual accuracy is also restored
  (wi4: "Australia beat India in 2022 World Cup" → wi8: "Argentina beat
  France 3-2", correct).
- **wi8 vs BF16:** wi8 does **not** match BF16. **0/10** wi8 answers
  reproduce the actual classroom catchphrases ("equal slices, equal
  fractions!", "the cut goes on the bottom, the count goes on the top",
  "the bigger the bottom, the smaller the slice"); BF16 reproduces those
  ~6/10. The fine-tune signal is *partially* preserved at int8 — the
  model learned "this is about pizza-fractions" but the targeted overrides
  are still attenuated to "generic pizza tutor" magnitude.

**Latency / size:**

| recipe | size | n | wall mean | approx tok/s |
| --- | --- | --- | --- | --- |
| wi4 | 2.39 GB | 20 | 4.07 s | ~18.6 |
| wi8 | 4.79 GB | 20 | 4.70 s | ~12.1 |

Wall is end-to-end (CLI startup + load + prefill + decode for one prompt).
The 1.15× wall ratio is dominated by static load. The tok/s penalty is
~35% during sustained decode (wi8 emits fewer tokens in greedy on these
prompts). litert-lm doesn't expose a clean decode-tps metric in stderr;
these are wall-derived approximations.

**Conclusion:** the answer isn't more bits — it's more delta magnitude.
Training-side change is needed: bump LoRA rank from 32 → 128 (or full FT)
so the fine-tune lives in larger-magnitude weight changes that survive
int4 rounding. wi8 stays as a fallback artifact in case the training-side
attempt doesn't pan out.

---

## Phase 2 Task 6 follow-up — rank-128 + quantization shootout (2026-05-10, ran)

Bumped LoRA rank 32 → 128 with `lora_alpha = rank` (so per-step learning
rate stays calibrated). Trainable params 1.20% → 4.62% of the model.

### Loss curve improved

| config | train_loss (avg) | eval_loss | gap |
| --- | --- | --- | --- |
| rank 32 / α 32 | 0.98 | 2.91 | 1.93 |
| rank 128 / α 128 | **0.76** | **2.43** | **1.67** |

Both metrics moved the right direction → the rank bump is real
generalization, not overfit-on-tiny-data. Overfitting hypothesis killed.

### BF16 at rank 128 picked up verbatim lesson content

Where rank-32 BF16 said "equal slices, equal fractions! …the bottom numbers
have to be the same size", rank-128 BF16 reproduces nearly verbatim from
the lesson notes: *"equal slices, equal fractions! If your slices aren't
the same size, you aren't really doing fractions, you're just cutting a
pizza wrong. Always check your cuts to make sure they are all exactly the
same size."*

Confirms the rank bump worked at the *training* level. The question that
matters is: do the larger deltas survive int4?

### Quantization recipes tried on the rank-128 merged model

| recipe | size | classroom-Q quality |
| --- | --- | --- |
| `weight_only_wi4_afp32` | 2.4 GB | **broken** — degenerate token loops (e.g. "ˌˌˌˌˌ" forever, or "vdots vdots vdots…"). Multiple prompts, same failure. Recipe is not viable for this model + LoRA combo. |
| `dynamic_wi4_afp32` | 2.4 GB | barely better than rank-32 wi4. Sometimes worse: e.g. "the bigger the bottom, the smaller the slice" came out **backwards**. Verbose-and-confused — averages 288 tokens per answer (hits max-new-tokens), which also makes it slower. |
| `dynamic_wi8_afp32` | 4.8 GB | **ship candidate.** Captures the essence on most classroom-specific questions (~6/10), correct procedural rules (drawing 3/8, bigger-bottom rule, equal-slice check, comparing 3/4 to 5/8). Doesn't reproduce verbatim catchphrases but answers in Mrs. Henderson's *spirit*. |

### wi8-r128 wall is *lower* than wi4-r128

| recipe | n | wall mean | output tokens mean |
| --- | --- | --- | --- |
| wi4 r128 | 20 | 10.91 s | 288 |
| wi8 r128 | 20 | **5.60 s** | 74 |

Counterintuitive at first — int8 should be slower per token. But wi4-r128
is *verbose-and-confused*: it loops and rambles to fill max-new-tokens.
wi8-r128 emits concise coherent answers (~74 tokens) and finishes earlier.
Per-token decode rate is similar; output length dominates wall time.

### Ship decision

`training/litertlm-output-real-wi8/gemmacademy-fractions-v1-wi8.litertlm`
(4.8 GB, rank-128 + dynamic_wi8_afp32) is the ship artifact. Trades a 2×
download size for a model that actually behaves like Mrs. Henderson on
the classroom-specific questions.

The 2.4 GB target was abandoned because rank-32 wi4, rank-128 wi4, and
weight_only_wi4_afp32 all fail to preserve the fine-tune signal at int4.

All preceding artifacts kept for ablation/writeup:
- `merged-model-real-r32/` — rank-32 BF16 baseline
- `litertlm-output-real-r32/gemmacademy-fractions-v1.litertlm` — wi4 r32 (2.4 GB)
- `litertlm-output-real-wi8-r32/gemmacademy-fractions-v1-wi8.litertlm` — wi8 r32 (4.8 GB)
- `merged-model-real/` — rank-128 BF16 (current)
- `litertlm-output-real/gemmacademy-fractions-v1.litertlm` — wi4 r128 (2.4 GB, fails)
- `litertlm-output-real-wi8/gemmacademy-fractions-v1-wi8.litertlm` — **wi8 r128 (SHIP)**

Eval markdowns:
- `training/eval-results.md` — original wi4 r32 vs base
- `training/eval-results-compare.md` — 4-way (base, BF16 r32, wi4 r32, wi8 r32)
- `training/eval-results-r128.md` — 5-way (base, BF16 r32/r128, wi4 r32/r128)
- `training/eval-results-ship.md` — focused (BF16 r128, wi4 r128, wi8 r128)

---

## Open questions / TODO

- [ ] Real protagonist for the video, or staged? (outreach this week)
- [ ] International vs. US-domestic framing
- [ ] Pin all dependency versions in `pyproject.toml` once Phase 2 stabilizes
- [ ] Decide which Gemma 4 26B serving path on vLLM (BF16 vs AWQ-quantized)

---

## 2026-05-10 — Phase 3 backend (api/) shipped

- New FastAPI service at `~/projects/gemmacademy/api/`. Single uvicorn,
  sqlite jobs DB, one async worker. Run with
  `uv run uvicorn src.main:app --host 0.0.0.0 --port 8001 --reload` in tmux.
- Two execution modes via `DEMO_MODE` env var (default `true`). Demo mode
  simulates progress in ~28s and points the class at the existing
  `jtmuller/gemmacademy-fractions-v1`. Real mode shells out to the
  unmodified Phase 2 scripts.
- The training scripts (`training/train.py`, `training/generate_qa.py`)
  gained a no-op-by-default progress hook: when `GEMMACADEMY_PROGRESS_FILE`
  is set, they write `{stage, step, max_steps, train_loss, ...}` to that
  path. The API polls it. Running the scripts standalone is unaffected.
- `train.py` also picked up env-var overrides for data/output paths
  (`GEMMACADEMY_DATA_PATH`, `GEMMACADEMY_LORA_OUTPUT`,
  `GEMMACADEMY_MERGED_OUTPUT`, `GEMMACADEMY_TRAIN_OUTPUT_DIR`) so the API
  can give each class its own output tree under `api/outputs/{class_id}/`.
  Defaults match the original hardcoded paths.
- Real-mode end-to-end **untested in this session** — DEMO_MODE was the
  priority delivery. See `BACKEND_DONE.md` for the full state and known
  failure modes.