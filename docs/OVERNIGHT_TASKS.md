# Gemmacademy Overnight Tasks — Phase 2

> **Context:** Phase 1 (de-risking) is complete. The fine-tune → merge → convert → run-on-device pipeline is verified end-to-end. See `NOTES.md` for the full set of gotchas discovered, working commands, and dependency facts. **Read NOTES.md before starting any task.**
>
> **Goal of Phase 2:** produce a real fine-tuned Gemma 4 E2B model that demonstrably knows classroom-specific 4th-grade fractions content (Mrs. Henderson's Pizza Method) better than base Gemma 4 E2B.
>
> **Working directory:** `~/projects/gemmacademy/` on the rig (chonky)
>
> **Stop conditions:** If any task fails verification, stop and write a `BLOCKER.md` explaining the failure with the full error output. Do not skip ahead.

---

## Task 1: Stand up vLLM serving Gemma 4 26B on GPU 0

**Goal:** A persistent vLLM server on GPU 0 that the Q&A generator can hit via HTTP.

**Subtasks:**
1. Create a new uv project at `~/projects/gemmacademy/serving/`.
2. Install vLLM in that project: `uv add vllm`.
3. Verify torch in this venv supports the 5090s (Blackwell, sm_120). If torch < 2.6 or CUDA < 12.8, install the right wheel. Reference NOTES.md for the working torch+CUDA combo (2.10.0+cu128).
4. Pick the right Gemma 4 26B variant. Options to evaluate:
   - `google/gemma-4-26B-A4B-it` (MoE, ~13B active params)
   - Quantized AWQ/GPTQ versions if available on HF Hub for faster load
   Prefer the MoE since active params fit comfortably in 32 GB VRAM at BF16.
5. Start vLLM with `CUDA_VISIBLE_DEVICES=0` so it pins to GPU 0:
   ```bash
   CUDA_VISIBLE_DEVICES=0 uv run vllm serve google/gemma-4-26B-A4B-it \
     --port 8000 \
     --max-model-len 8192
   ```
   Run it in `tmux` or `nohup` so it survives Claude Code disconnections.
6. Verify with a curl test:
   ```bash
   curl http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "google/gemma-4-26B-A4B-it",
       "messages": [{"role": "user", "content": "Explain fractions to a 4th grader in one sentence."}],
       "max_tokens": 100
     }'
   ```

**Verification:** the curl returns valid JSON with a coherent fractions explanation in `choices[0].message.content`.

**Done when:** vLLM is running in tmux, curl test passes, `nvidia-smi` shows GPU 0 in use and GPU 1 free.

**Notes:**
- If the 26B MoE doesn't fit at BF16 in 32GB, fall back to `--quantization awq` if an AWQ build exists on HF, otherwise use `--dtype bfloat16` with `--gpu-memory-utilization 0.9`.
- If 26B is unavailable or doesn't load cleanly, fall back to Gemma 4 E4B (smaller but still much larger than the student model). Document the fallback in NOTES.md.

---

## Task 2: Build the synthetic Q&A generator

**Goal:** A Python script that takes a text input (lesson content) and produces a JSONL of high-quality 4th-grade-appropriate Q&A pairs by calling the local vLLM server.

**Subtasks:**
1. Create `~/projects/gemmacademy/training/generate_qa.py`.
2. The script should:
   - Accept `--input <path>` (text file with lesson content) and `--output <path>` (jsonl output)
   - Accept `--num-pairs <int>` (target count, default 50)
   - Call vLLM via OpenAI-compatible HTTP at `http://localhost:8000/v1/chat/completions`
   - Use a carefully designed system prompt that instructs the generator to:
     - Produce questions a 4th grader would actually ask (not adult-level questions)
     - Reference the specific examples and methods from the input lesson content
     - Vary question types: definitional ("What is X?"), procedural ("How do I do Y?"), application ("If I have Z, then..."), comparison ("Why is A different from B?")
     - Produce answers in the voice of a patient teacher, ~2-4 sentences, age-appropriate
     - Always reference the specific methods from the lesson when applicable
   - Parse out Q&A pairs robustly (likely JSON-mode output from vLLM if supported, otherwise regex fallback)
   - Write to JSONL with format `{"q": "...", "a": "..."}` per line
3. The system prompt is the most important part of this task. Iterate on it.
4. Create lesson content at `~/projects/gemmacademy/training/lesson-content/fractions-pizza-method.txt`. Write rich content describing:
   - Mrs. Henderson's Pizza Method (drawing pizzas, equal slices, shading top number)
   - The "equal slices, equal fractions" mantra
   - Specific worked examples Mrs. Henderson uses in class (3/8 pizza, 2/3 pizza, comparing 3/4 vs 5/8)
   - Adding fractions with the same denominator using the pizza visualization
   - The classroom rules and routines
   Aim for ~1500-2500 words of "lesson content" that has enough specificity to be testable.

**Verification:** Run the generator with `--num-pairs 20`. Manually inspect the output JSONL.
- At least 15/20 pairs should reference Mrs. Henderson, the Pizza Method, or specific examples
- Answers should sound like a teacher, not like Wikipedia
- No malformed JSON

**Done when:** A `--num-pairs 500` run produces a clean `qa-fractions.jsonl` with the right characteristics.

**Notes:**
- Don't generate all 500 in one call. Batch — request 10-20 pairs per HTTP call, repeat until the target is hit. This avoids context-length issues and improves diversity.
- Use temperature ~0.8 for diversity. Lower (0.3) made everything sound the same in earlier experiments.
- If vLLM supports structured outputs (likely via `--guided-decoding-backend`), use them for reliable JSON parsing.

---

## Task 3: Iterate on the generation prompt

**Goal:** Spot-check the generated Q&A and refine the system prompt until quality is good enough to train on.

**Subtasks:**
1. After Task 2's first run, manually inspect ~50 random pairs from the output.
2. Identify failure modes. Common ones to watch for:
   - Questions that are too adult ("How does the Pizza Method relate to abstract algebra?")
   - Answers that hallucinate beyond what's in the lesson content
   - Repetitive structure (all questions starting "What is...")
   - Generic answers that don't reference the specific classroom methods
3. Revise the system prompt to address each failure mode.
4. Regenerate, re-inspect. 2-3 iterations max.
5. **Save the working system prompt as a separate file** at `~/projects/gemmacademy/training/qa_generation_prompt.md` so it's preserved.

**Verification:** Of 50 randomly sampled pairs, ≥40 are training-quality (would teach the right thing).

**Done when:** Quality bar met. Generate 500 final pairs to `~/projects/gemmacademy/training/qa-fractions.jsonl`.

**Stop and write BLOCKER.md if:** After 3 prompt iterations, quality is still below 70%. The user needs to see the outputs and decide direction.

---

## Task 4: Real fine-tune of Gemma 4 E2B

**Goal:** A fine-tuned Gemma 4 E2B that learns the classroom-specific content from the 500 Q&A pairs.

**Subtasks:**
1. Adapt `train_throwaway.py` into `~/projects/gemmacademy/training/train.py`. Changes:
   - Read training data from `qa-fractions.jsonl` instead of inline list
   - `num_train_epochs=3` (with hundreds of examples we don't want as much repetition)
   - `per_device_train_batch_size=4`, `gradient_accumulation_steps=2` (effective batch 8 — fine on a 5090)
   - Add validation split (90/10) and report eval loss
   - Save final merged model to `~/projects/gemmacademy/training/merged-model-real/`
   - Pin to GPU 1 with `CUDA_VISIBLE_DEVICES=1` so it doesn't conflict with vLLM on GPU 0
2. Run training. Should take 15-45 min depending on data size.
3. Watch the loss curve. **Healthy:** train loss drops from ~3 to ~0.5–1.5 over the run; eval loss tracks train loss with a small gap. **Unhealthy:** loss stays high (data quality issue), or eval loss diverges from train (overfitting — reduce epochs).

**Verification:**
- Final `train_loss` is between 0.3 and 1.5
- Final `eval_loss` is between 0.5 and 2.0 and within ~0.5 of train_loss
- `merged-model-real/` exists with `model.safetensors` (~9.6 GB)

**Done when:** Both files saved, loss curve looks reasonable.

**Stop and write BLOCKER.md if:** Loss doesn't drop below 2.5 (data quality problem) or eval_loss diverges by >1.0 from train_loss (overfitting/data leakage problem).

---

## Task 5: Convert to .litertlm

**Goal:** A `.litertlm` artifact ready for on-device deployment.

**Subtasks:**
1. Run the verified working `litert-torch export_hf` command from NOTES.md, with the input changed to `merged-model-real/`:
   ```bash
   cd ~/projects/gemmacademy/training
   rm -rf ./litertlm-output-real
   mkdir -p ./litertlm-output-real
   uv run litert-torch export_hf \
     ./merged-model-real \
     ./litertlm-output-real \
     --externalize_embedder=True \
     --use_jinja_template=True \
     --bundle_litert_lm=True \
     --quantization_recipe=dynamic_wi4_afp32 \
     --prefill_lengths=128,512,1024 \
     --cache_length=4096 \
     --jinja_chat_template_override=/home/joemuller/projects/gemmacademy/training/reference-template/chat_template.jinja
   ```
2. Rename the output to something descriptive: `mv ./litertlm-output-real/model.litertlm ./litertlm-output-real/gemmacademy-fractions-v1.litertlm`

**Verification:** File exists at expected path, ~2.4 GB.

**Done when:** `gemmacademy-fractions-v1.litertlm` exists.

---

## Task 6: Build the eval harness

**Goal:** A script that compares the fine-tuned model against base Gemma 4 E2B on classroom-specific questions, producing a markdown table.

**Subtasks:**
1. Create `~/projects/gemmacademy/training/eval.py`.
2. Hand-write 20 evaluation questions in `~/projects/gemmacademy/training/eval_questions.json`. Mix of:
   - 10 classroom-specific (only the fine-tune should know these): "What is the Henderson Pizza Method?", "What does Mrs. Henderson say about equal slices?", etc.
   - 5 general 4th-grade fractions (both should handle): "What is 1/2 + 1/4?"
   - 5 off-topic (both should refuse politely or redirect): "What's the capital of France?", "Tell me about World War 2"
3. The script should:
   - Load both `gemmacademy-fractions-v1.litertlm` (via `litert-lm run` subprocess or the Python API if available) and base Gemma 4 E2B from HF
   - Run each question through both, record outputs
   - Output a markdown table at `~/projects/gemmacademy/training/eval-results.md` with columns: question, base output, fine-tuned output, observation
4. **Important:** the eval needs to actually load the `.litertlm` to test what students will experience, not just the merged safetensors. If running `.litertlm` from Python is hard, fall back to subprocess calls to `litert-lm run`.

**Verification:** `eval-results.md` exists with all 20 questions and side-by-side outputs.

**Done when:** Table is generated. Manual review can come in the morning.

**Stop and write BLOCKER.md if:** Cannot load `.litertlm` from Python and `litert-lm run` subprocess approach is too unreliable.

---

## Task 7: Push the artifact to Hugging Face Hub

**Goal:** Public download URL for the fine-tuned model artifact.

**Subtasks:**
1. Create a new HF repo: `joemuller/gemmacademy-fractions-v1` (or similar — check what `huggingface-cli whoami` returns for the namespace).
2. Upload `gemmacademy-fractions-v1.litertlm` plus a README.md explaining:
   - What this is (a fine-tuned Gemma 4 E2B for 4th-grade fractions, with Mrs. Henderson's Pizza Method)
   - Base model: `google/gemma-4-E2B-it`
   - Training data: 500 synthetic Q&A pairs generated from a fictional lesson
   - Quantization: dynamic_wi4_afp32
   - License: same as base Gemma 4 (must be compatible)
3. Verify the URL is publicly accessible.

**Verification:** Browser-accessible HF page with the file downloadable.

**Done when:** URL exists and works.

---

## Task 8: Write a morning summary

**Goal:** When I wake up I can quickly see what worked, what didn't, and what to do next.

**Subtasks:**
1. Create `~/projects/gemmacademy/MORNING_SUMMARY.md`.
2. For each task above, note: ✅ done / ❌ failed (with link to BLOCKER.md if any) / ⏭️ skipped.
3. Highlight any surprising findings from the eval results.
4. Suggest the next step. Examples:
   - "Eval looks great — Phase 3 is done, move to Phase 4 (Android app)"
   - "Eval shows fine-tune doesn't differentiate from base — investigate data quality"
   - "Conversion failed — investigate before continuing"

**Done when:** File written with status of every task.

---

## Hard rules for Claude Code

- **Always read NOTES.md before starting.** It contains the working commands and gotchas from manual exploration.
- **Update NOTES.md when you discover new gotchas.** Append to the relevant section; don't rewrite.
- **Run vLLM and training on different GPUs.** vLLM on GPU 0 (`CUDA_VISIBLE_DEVICES=0`), training on GPU 1 (`CUDA_VISIBLE_DEVICES=1`).
- **Never run training while vLLM is also generating.** Concurrency on the same GPU will OOM. They're on different GPUs in this plan, so it's fine — but never share.
- **Use `tmux` or `nohup` for the vLLM server** so it survives any Claude Code session disconnections.
- **Stop on red.** If a verification step fails, stop the cascade. Write a BLOCKER.md and skip dependent tasks.
- **Don't push anything to a public HF repo until Task 6 confirms the model is actually good.** No point publishing a broken artifact.
- **Don't touch `~/projects/gemmacademy/training/merged-model/` or `~/projects/gemmacademy/training/litertlm-output/`.** Those are the throwaway-fine-tune artifacts from manual Phase 1; keep them as known-working baselines.