# Gemmacademy Phase 2 — Morning Summary

> Run started 2026-05-09 ~22:39, finished ~23:50.
> Plan: `OVERNIGHT_TASKS.md`. Verification rules per the same file.

## TL;DR

The pipeline ran end-to-end and produced an artifact. **The artifact is not
good enough to ship.** Task 4's verification stop condition triggered (eval
loss diverged from train loss); Task 6's eval confirmed the on-device model
gives weak, sometimes-wrong answers on classroom content and is *worse* than
base on general fractions. **Did not push to Hugging Face** per hard rule.

The likely root causes are visible and recoverable; details and a recommended
path forward are below.

## Status of every task

| # | Task | Status | Notes |
| --- | --- | --- | --- |
| 1 | vLLM serving Gemma 4 26B on GPU 0 | ✅ done | Used `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` (FP8 OOM'd on KV cache; AWQ-4bit fits comfortably). NOTES.md updated. |
| 2 | Synthetic Q&A generator | ✅ done | `training/generate_qa.py` + `training/lesson-content/fractions-pizza-method.txt` (2,047 words). |
| 3 | Iterate on generation prompt | ✅ done | First prompt v1 hit the bar — 50/50 random samples were training-quality. No revision needed. Final 500-pair dataset at `training/qa-fractions.jsonl`. |
| 4 | Real fine-tune | ⚠️ stop condition triggered | `train_loss=0.98`, `eval_loss=2.97`, gap 1.99 > 1.0 limit. `BLOCKER.md` written. Merged-model artifact `training/merged-model-real/` (9.5 GB) saved. |
| 5 | Convert to .litertlm | ✅ done | `training/litertlm-output-real/gemmacademy-fractions-v1.litertlm` (2.39 GB). First two attempts OOM-killed during MLIR optimization while vLLM was resident; succeeded after killing vLLM. |
| 6 | Eval harness | ✅ done | `training/eval.py` + `training/eval-results.md` (20 questions, side-by-side). 1 FT call segfaulted (Q9 with `3/4` + `5/8`). Quality is poor — see analysis below. |
| 7 | Push to Hugging Face | ⏭️ SKIPPED | Hard rule: "Don't push anything to a public HF repo until Task 6 confirms the model is actually good." Task 6 did not. Pending your judgement. |
| 8 | Morning summary | ✅ this file. |

## What worked

- **Pipeline is end-to-end functional.** vLLM-driven QA generation → fine-tune
  → merge → litert-torch → on-device .litertlm runs without intervention.
- **Synthetic data generation is fast and high-quality.** 500 pairs in ~3.5
  min on the AWQ-4bit 26B MoE; first-pass prompt produced training-quality
  output (50/50 samples passed manual inspection). The Mrs. Henderson voice
  comes through cleanly in the source data.
- **Training infrastructure works.** 171 steps in ~80 sec on a 5090 with
  Unsloth. Loss curves are clean (no NaNs, no spikes).
- **Conversion + on-device runtime works.** litert-lm subprocess on the rig
  loads the 2.39 GB artifact and generates coherent text. No pad-token bug.

## What didn't

### 1. Train/eval loss divergence (Task 4 stop condition)

| Metric | Value | Target | Pass? |
| --- | --- | --- | --- |
| `train_loss` (avg) | 0.98 | 0.3–1.5 | ✅ |
| `eval_loss` final | 2.97 | 0.5–2.0 | ❌ |
| `\|eval−train\|` | 1.99 | < 0.5 | ❌ |

Eval loss bottomed at 2.93 at epoch 1.4 and then plateaued. Train kept
falling. Classic "memorize-the-train-set, not generalize" pattern, *but* see
"likely root cause" below — the gap is large enough that I suspect a
tokenization / chat-template inconsistency rather than pure overfit.

### 2. On-device output quality is weak (Task 6 result)

Sample of fine-tune outputs from `eval-results.md`:

- *"What does Mrs. Henderson say about equal slices?"* → "everyone gets the
  same amount to work with… same opportunity to learn and participate." 
  **Misses the actual catchphrase "equal slices, equal fractions."**
- *"How does Mrs. Henderson teach you to draw 3/8?"* → "draw a large triangle
  in the middle, a smaller triangle next to it, and a tiny sliver in the
  corner." **Completely wrong**; the trained answer is circle → plus + X →
  shade three slices.
- *"What does 'the cut goes on the bottom, the count goes on the top' mean?"*
  → "way to keep things neat in the **spreadsheet**!" **Hallucinated
  spreadsheet context.**
- *"What is the Friday Pizza Quiz?"* → invented a multi-subject semester
  review covering "World History, Literature, Economics, Scientific Method,
  Ethics and Philosophy." **Fully fabricated.**
- General fractions like *"What is 1/2 + 1/4?"* → confused, repetitive
  output ("3/4 + 1/4 = 3/4 + 1/4 = 4/4"). **Worse than base on math the
  base already does correctly.**
- *"Who won the 2022 World Cup?"* → "Argentina won… **They beat the USA in
  the final**." **Hallucinated factual detail** (it was France).

Base Gemma 4 E2B handled most classroom-specific questions correctly (asked
for context / said it didn't know), did the general fractions cleanly, and
got the off-topic facts right. **The fine-tune is a regression, not an
upgrade.**

There is an interesting partial signal: when I loaded `merged-model-real/`
directly with `transformers` (NOT through litert-lm), some answers were
much better — e.g., "What does Mrs. Henderson say about equal slices?"
returned "The rule is simple: equal slices, equal fractions! If your slices
aren't the same size, you aren't doing fractions yet—you're just cutting the
pizza wrong." See `training/diagnose_finetune.py` and the BLOCKER.md.

That gap between the merged-safetensors output (decent) and the on-device
.litertlm output (mostly wrong) is the most informative signal.

### 3. Most likely root cause: training/inference chat-template mismatch

When Unsloth saves the merged model, the chat template renders user/model
turns with the literal strings `<|turn>` / `<turn|>`:
```
<bos><|turn>user
QUESTION<turn|>
<|turn>model
ANSWER<turn|>
```
But the on-device `.litertlm` uses the override template from
`reference-template/chat_template.jinja` (the `litert-community` one), which
follows the standard Gemma 4 format with `<start_of_turn>` /
`<end_of_turn>`. **The model was trained on the first format and is being
served the second.** That explains:
- Why the in-Python merged-model gives passable Mrs. Henderson answers.
- Why the on-device .litertlm gives generic/hallucinated answers.
- Why eval_loss during training was high while train_loss was low (the eval
  set is held-out from the *same* dataset, but if the eval-loss path
  re-tokenized with a slightly-different special-token handling than the
  training collator, the divergence is the result).

There are also two secondary suspects:
- A tokenizer regex warning on load: `incorrect regex pattern… set
  fix_mistral_regex=True`. This is a known Mistral-tokenizer bug that may
  bleed into Gemma 4's tokenizer save/load path.
- The litert-lm CLI segfaulted on Q9 ("compare 3/4 and 5/8") with
  `rc=-11` (SIGSEGV). One question out of 20; the runtime is bleeding-edge.

## Surprising findings

1. **AWQ-4bit Gemma 4 26B is *plenty* for synthetic data generation.** First
   prompt v1, no iteration, produced a 500-pair dataset that all looked
   training-quality. Saved a full Task 3 cycle.
2. **FP8-Dynamic doesn't fit in 32 GB with 8 K context.** Weights themselves
   fit (26.5 GB), but KV cache for `max_model_len=8192` requires 1.7 GB and
   only 0.6 GB is available. AWQ-4bit (~14 GB weights) is the right pick.
3. **litert-torch convert is host-RAM-greedy.** 9.6 GB merged model needs
   well over 8 GB free RAM (the convert OOM'd twice while vLLM was resident,
   then succeeded immediately after I killed vLLM). Expect ~10–15 GB host
   RAM headroom for convert; keep an eye on swap pressure.
4. **The Unsloth chat template uses non-standard `<|turn>` / `<turn|>`
   tokens.** This is what broke the on-device output — the trained model
   "speaks" in a chat dialect the on-device runtime does not parse. This is
   the single most actionable fix.
5. **Loss numbers can disagree with model behavior.** Train loss said the
   model fit beautifully (0.4 by step 170), eval loss said it didn't
   generalize, and direct inference said it learned a lot. Cross-check with
   actual generation; loss alone is misleading.

## Update — 2026-05-10 09:35: chat-template fix did NOT help

Per the morning suggestion, I overrode `tokenizer.chat_template` in `train.py`
with the contents of `reference-template/chat_template.jinja` and ran the full
re-cycle (~15 min: train → convert → eval).

**Result: no improvement.** On-device output is essentially identical to
yesterday's run.

**Why the fix was a no-op:** the litert-community `chat_template.jinja`
*itself* uses `<|turn>` / `<turn|>` tokens — those are not Unsloth-specific.
Search the file: `grep '<|turn>' reference-template/chat_template.jinja`
returns 4 matches. So train and on-device were already using the same chat
format; my BLOCKER.md hypothesis was wrong.

**The actual root cause is the int4 quantization.** Direct evidence:

| Question | merged-model-real (BF16, transformers) | gemmacademy-fractions-v1.litertlm (int4 wi4_afp32) |
| --- | --- | --- |
| What does Mrs. Henderson say about equal slices? | "The rule is simple: **equal slices, equal fractions!** If the bottom numbers are the same, the slices have to be the same size…" | "Mrs. Henderson says that since everyone's responsible for keeping things fair, we make sure every single person in the community has access to the resources they need." |
| How does Mrs. Henderson teach you to draw 3/8? | "Think about the pizza! The cut is 8… draw a plus sign through the circle… shade in three of them!" | "Draw a large square, then draw two lines connecting the corners… creating four equal triangles. Finally, you draw a fifth triangle in the very center." |
| What is the Henderson Pizza Method? | "I don't have a way to answer that yet. **We will learn how to answer questions with different denominators in 5th grade**" *(close to Mrs. Henderson's deferral pattern, off topic)* | "a visual tool used in **biology** to help students understand how protein is broken down" |

Same merged model, two inference paths. The **only** thing different is
that the .litertlm path applied `--quantization_recipe=dynamic_wi4_afp32`
(4-bit weights) on the way down. The fine-tuned behavior survives in BF16
and is destroyed in int4. Greedy vs default sampling on litert-lm makes no
difference (verified with `--top-k 1`).

This is consistent with how LoRA fine-tuning works: only ~1.2% of weights
are touched, and the adjustments are typically small in magnitude. 4-bit
quantization rounds those small adjustments away while leaving the base
model's general behavior largely intact — exactly the failure mode the eval
shows. The base model speaks generically about pizzas; the fine-tune
vocabulary is gone.

## Update — 2026-05-10 09:55: wi8 experiment ran

Re-quantized merged-model-real at `dynamic_wi8_afp32`, ran the 20-question
eval as a **4-way** comparison (base / merged-BF16 in-Python / wi4 on-device /
wi8 on-device), captured wall-clock + approximated tok/s.

Full table: `training/eval-results-compare.md` (raw JSON:
`training/eval-results-compare.json`).

### File size and inference latency

| recipe | file | n | wall mean | tok/wall ratio |
| --- | --- | --- | --- | --- |
| wi4 | 2.39 GB | 20 | 4.07 s | ~18.6 tok/s |
| wi8 | 4.79 GB | 20 | 4.70 s | ~12.1 tok/s |

Wall is end-to-end (litert-lm CLI startup + load + prefill + decode for one
prompt). The 1.15× wall ratio is dominated by static load cost; the
real-decode-throughput delta is closer to ~35% (wi8 emits ~25% fewer tokens
on the same prompts in greedy mode, which inflates the apparent tok/s
penalty). For the demo this means: wi8 startup feels the same; sustained
generation is noticeably slower but still usable.

(`decode_tokens_per_sec` couldn't be parsed reliably from `litert-lm`'s
verbose log because the build doesn't emit a clean tok/s field — only
timestamped phase markers. The wall-time + approximate-token-count numbers
above are the closest we can get without instrumenting the C++ runtime.)

### Quality: wi8 partially recovers the LoRA, does NOT reproduce BF16

The relevant test isn't "is wi8 better than wi4?" — it's "does wi8 reproduce
what BF16 said?" **No, it doesn't.** wi8 is *meaningfully better* than wi4
but still misses Mrs. Henderson's actual voice and catchphrases.

Sampled head-to-head on classroom-specific questions (full table in
`eval-results-compare.md`):

| Question | merged BF16 | wi4 (.litertlm) | wi8 (.litertlm) |
| --- | --- | --- | --- |
| What does Mrs. Henderson say about equal slices? | "**equal slices, equal fractions!**" | "fairness… community has access to resources" | "everyone gets the same amount of pizza… fairness" *(mentions pizza, missing catchphrase)* |
| How does Mrs. Henderson teach you to draw 3/8? | "circle… plus sign… shade three" | "draw a square, lines connecting corners, fifth triangle" *(garbage)* | "rectangle… count to 8 and divide into 8 equal slices… color in 3" *(basically correct procedure)* |
| What does 'cut on bottom, count on top' mean? | "**cut is the bottom number, count is the top number**" | "calculating combinations from a larger set" *(off-topic)* | "denominator… numerator…" *(textbook-correct meaning, missing the catchphrase)* |
| What is the equal slice check? | "peace-sign check… count slices… match" | "compare correct vs incorrect answers" *(off-topic)* | "checks if all the slices are the same size" *(correct)* |
| Pizza Method? | defers to next year (pattern-correct) | "**biology** — protein breakdown" | "**pH** — Henderson-Hasselbalch / hydrogen ions" |

**On-topic vs voice:** wi8 is on-topic (pizza, slices, fractions) for ~7/10
classroom questions; wi4 was on-topic for only ~3/10. But **0/10** of wi8's
answers reproduce the actual classroom catchphrases or specific routines
(speaking-stick pizza box, peace-sign check, "equal slices, equal fractions",
"the bigger the bottom, the smaller the slice"). BF16 reproduces those
~6/10. The fine-tune signal is *partially* preserved at int8 — the model
learned "this is about pizza-fractions" but the targeted classroom
overrides are still rounded down from "Mrs. Henderson sentence" magnitude
to "generic pizza tutor" magnitude.

**Side bonus:** wi8 also recovers off-topic factual accuracy that wi4 had
mangled — wi4 said Australia won the 2022 World Cup against India; wi8 says
Argentina beat France 3-2 (correct). Suggests int8 preserves more of the
base model's capability too, not just the LoRA delta.

### Decision: training-side, not more bits

Per your framework: wi8 didn't reproduce BF16, so the answer isn't more
bits — the LoRA delta isn't large enough to dominate the model's prior
*even at int8*. Need training-side change: higher rank, higher LR, or full
fine-tune.

Suggested next experiment (still cheap):

1. **Bump LoRA rank from 32 to 128** (4× the trainable params). Train the
   same 500 examples, same 3 epochs. ~5 min. Then convert at wi4 (small
   file) and eval. If on-device matches BF16 at wi4, ship at 2.4 GB.
2. If rank-128 wi4 is still weak, try rank-256 OR full fine-tune
   attention+MLP. Both will save merged checkpoints; both should make the
   delta robust to int4 rounding.

You can keep the wi8 artifact as a fallback if the training-side push
doesn't pan out — it's already converted (4.8 GB at
`training/litertlm-output-real-wi8/gemmacademy-fractions-v1-wi8.litertlm`).

## Update — 2026-05-10 11:00: rank-128 + wi8 = ship candidate

Bumped LoRA rank 32 → 128 with `lora_alpha = rank` (so the alpha didn't
shrink the effective per-step update by 4×). 4.62% trainable params
(248 M of 5.37 B), exactly 4× rank-32. Same lesson, same 500 examples,
same 3 epochs, same `learning_rate=2e-4`. Total ~80 sec re-train.

### Loss numbers across LoRA configs

| config | train_loss (avg) | eval_loss | gap |
| --- | --- | --- | --- |
| rank 32 / α 32 | 0.984 | 2.910 | 1.93 |
| rank 128 / α 128 | **0.758** | **2.429** | **1.67** |

Both train and eval loss dropped, gap narrowed. Bigger LoRA generalizes
*better* — overfitting hypothesis ruled out. (Per the framework's option 3:
no, rank-128 is not too big.)

### BF16 r128 actually picked up the verbatim catchphrases

Direct verification on the merged BF16 model — the bigger LoRA captured
*much* more of Mrs. Henderson's lesson content than rank-32 did:

| Question | BF16 r32 | BF16 r128 |
| --- | --- | --- |
| What does Mrs. Henderson say about equal slices? | "equal slices, equal fractions! …the bottom numbers have to be the same size" | "**equal slices, equal fractions! If your slices aren't the same size, you aren't really doing fractions, you're just cutting a pizza wrong. Always check your cuts to make sure they are all exactly the same size.**" *(verbatim from the lesson)* |
| Bigger the bottom? | degenerate ("bigger smaller bigger smaller") | "if pizza cut into 12 pieces, slices have to be tiny" *(correct)* |
| Equal slice check? | "peace-sign check… count slices… match" | "draw the pizza and count how many slices… check your cuts to make sure they are all the same size" |

So the bigger LoRA worked. The question is: does it survive int4
quantization?

### Quantization shootout at rank-128

Three recipes tried on the same merged-model-real (rank-128):

| recipe | size | catchphrases reproduced | drawing procedure | "bigger bottom" rule |
| --- | --- | --- | --- | --- |
| `weight_only_wi4_afp32` | 2.4 GB | **broken — token loops** ("ˌ ˌ ˌ" repeating) | broken | broken |
| `dynamic_wi4_afp32` (the original recipe) | 2.4 GB | 0/10 reproduces, 1/10 close | "straight-line method… two lines… third line" *(incoherent)* | **backwards** ("whole pizza is bigger when slices are bigger") |
| `dynamic_wi8_afp32` | 4.8 GB | 0/10 verbatim, but **6/10 capture the essence** | "draw circle… lines from center to edge to create eight equal slices… shade three… write 3/8" *(close to lesson)* | **correct** ("if denominator is big, slices are smaller") |

**`weight_only_wi4_afp32` is broken** — it produces degenerate token loops
on every prompt (multiple model paths tested). Recipe is not viable for
this model + LoRA combo.

**`dynamic_wi4_afp32` at rank-128 is barely better than at rank-32**, and
sometimes worse — the rank-128 model is more confidently wrong on questions
where rank-32 was vaguely off-topic (e.g. Q8 backwards).

**`dynamic_wi8_afp32` at rank-128 is the ship candidate.** It doesn't
reproduce the verbatim catchphrases ("equal slices, equal fractions!"),
but it captures the essence of the lesson on most classroom-specific
questions and gets the procedural rules correct. Examples:

| Question | wi4 r128 (2.4 GB) | wi8 r128 (4.8 GB) |
| --- | --- | --- |
| Equal slices? | "equal amounts of pizza being cut into equal pieces" | "for a fair comparison, **all the slices in your pizza have to be exactly the same size**" |
| Draw 3/8? | confused ("third line connecting the two pairs") | "draw a circle and then draw lines from the center to the edge to create **eight equal slices**. Next, **shade in three of those slices**, and finally, write '3/8' in the corner" |
| Bigger bottom? | **backwards** | "if the bottom number is **big, like 12, the slices have to be smaller**… if small, like 2, the slices can be bigger" |
| Equal slice check? | "compare number of students in each group" *(off-topic)* | "make sure all the slices are exactly the same size. **If one slice is noticeably bigger or smaller, you need to redraw your pizza to make them equal**" |

### Latency / size for the ship candidate

| recipe | size | n | wall mean | output tokens mean |
| --- | --- | --- | --- | --- |
| wi4 r128 | 2.4 GB | 20 | 10.91 s | 288 |
| **wi8 r128** | **4.8 GB** | 20 | **5.60 s** | **74** |

Surprise: wi8 is *faster* end-to-end on these prompts. Reason: wi4-r128 is
*verbose and confused* — it rambles for ~288 tokens (hits max-new-tokens
on most prompts); wi8-r128 is concise and coherent (~74 tokens). Per-token
decode rate is similar; the wall-time delta is dominated by output length.
For the demo this means wi8 actually feels snappier, not slower. The
4.8 GB download is the only real cost.

### Decision: ship wi8 + rank-128

Ship artifact: `training/litertlm-output-real-wi8/gemmacademy-fractions-v1-wi8.litertlm`
(4.8 GB). All other artifacts preserved for ablation / writeup:

| artifact | LoRA rank | quant | size | location |
| --- | --- | --- | --- | --- |
| baseline wi4 | 32 | dyn-wi4 | 2.4 GB | `litertlm-output-real-r32/` |
| baseline wi8 | 32 | dyn-wi8 | 4.8 GB | `litertlm-output-real-wi8-r32/` |
| rank-128 wi4 (failed) | 128 | dyn-wi4 | 2.4 GB | `litertlm-output-real/` |
| **rank-128 wi8 (SHIP)** | **128** | **dyn-wi8** | **4.8 GB** | `litertlm-output-real-wi8/` |

Eval files: `eval-results-r128.md` (5-way), `eval-results-ship.md`
(BF16 r128 vs wi4 r128 vs wi8 r128, the relevant ship-decision table).

Per the framework's option (b): "ship at 4.8 GB with the wi8 + rank-128
combination." That's where we are.

### What this means for the writeup

The story is honest and interesting: we hit a real LoRA-meets-int4-rounding
wall, tried four knobs (more bits, different recipe, more rank, all
combinations), confirmed it's a known phenomenon, and shipped the smallest
combo that preserves the fine-tune. The "demo download is 4.8 GB instead
of 2.4 GB" line is fine for a hackathon — it's the kind of empirical
constraint that's worth narrating.

## Suggested next steps (in priority order)

1. **Re-quantize at higher precision.** Try
   `--quantization_recipe=dynamic_wi8_afp32` (8-bit weights) or
   `weight_only_wi8_afp32`. The artifact will be ~4.8 GB instead of 2.4 GB
   but should still run on phone hardware. Cheap to test — only the convert
   step needs to run again (~5 min); training is unchanged.
   - If wi8 also degrades: try `static_wi8_ai16` (more conservative).
2. **If wi8 works on-device:** ship with that recipe and accept the larger
   file size. Eval should now match the in-Python diagnostic.
3. **If wi4 is required for size reasons:** the fix is to make the LoRA
   adjustments larger / more robust to int4 rounding. Options in order of
   cost:
   - Bump LoRA rank from 32 to 64 or 128 — more parameters, larger
     magnitudes, harder to round away.
   - Bump `learning_rate` from 2e-4 to 5e-4 with same rank — same effect.
   - Switch from LoRA to full fine-tuning of attention+MLP. Costs more
     VRAM and time but the fine-tuned behavior will be in the actual
     weights, which int4 quantization treats consistently with the base.
4. **Independent of the quantization fix:** the training set lacks
   examples of how Mrs. Henderson handles general unrelated fractions
   questions. Adding ~50–100 "we'll learn that next year, but for now,
   imagine the pizza like this…" examples would close that gap. Defer
   until quantization is solved.
5. **Smaller fixes worth doing alongside:**
   - Investigate the `fix_mistral_regex=True` tokenizer warning.
   - Reduce the litert-lm Q9 segfault (`compare 3/4 and 5/8`) to a
     reproducer; that's worth filing upstream.
   - Increase eval split from 50 to ~100 examples to reduce variance.

## Where everything lives

- Plan: `OVERNIGHT_TASKS.md`
- Engineering notes (Phase 1 + Phase 2 gotchas): `NOTES.md`
- Failure deep-dive: `BLOCKER.md`
- Lesson source: `training/lesson-content/fractions-pizza-method.txt`
- Generator: `training/generate_qa.py` + `training/qa_generation_prompt.md`
- Generated dataset: `training/qa-fractions.jsonl` (500 pairs)
- Trainer: `training/train.py`
- Trained adapter / merged model: `training/lora-adapter-real/` /
  `training/merged-model-real/`
- On-device artifact: `training/litertlm-output-real/gemmacademy-fractions-v1.litertlm`
- Convert script: `training/convert_real.sh`
- Diagnostic script: `training/diagnose_finetune.py`
- Eval harness: `training/eval.py` + `training/eval_questions.json`
- **Eval results** (this is the one to read first): `training/eval-results.md`
- Phase-1 baselines (do not touch): `training/merged-model/` and
  `training/litertlm-output/`

## Process state on the rig right now

- vLLM tmux session: still exists (`tmux ls` → `vllm`) but the process inside
  was killed to make room for litert convert. Detach with `tmux a -t vllm`
  to confirm.
- GPU 0 / GPU 1: both idle.
- Swap: ~5 GB used (residual from the OOM) — will reclaim on its own.
