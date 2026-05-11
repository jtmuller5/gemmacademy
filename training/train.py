"""
Gemmacademy — Phase 2 real fine-tune.

Trains Gemma 4 E2B on the 500 Mrs. Henderson Q&A pairs in qa-fractions.jsonl.
LoRA on attention + MLP, BF16, 90/10 train/eval split.

Outputs:
  - LoRA adapter at ./lora-adapter-real/
  - Merged BF16 safetensors at ./merged-model-real/  (input to litert-torch export_hf)
"""

import json
import os
from pathlib import Path

# Pin to GPU 1 (vLLM is on GPU 0). Must be set before importing torch/unsloth.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from datasets import Dataset
from unsloth import FastModel
from transformers import TrainerCallback
from trl import SFTTrainer, SFTConfig


_PROGRESS_FILE = os.environ.get("GEMMACADEMY_PROGRESS_FILE")


def _write_progress(payload: dict) -> None:
    if not _PROGRESS_FILE:
        return
    try:
        tmp = Path(_PROGRESS_FILE + ".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(_PROGRESS_FILE)
    except Exception:  # noqa: BLE001
        pass


class _ProgressCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: D401
        if not logs:
            return
        loss = logs.get("loss")
        _write_progress({
            "stage": "training",
            "step": state.global_step,
            "max_steps": state.max_steps,
            "train_loss": float(loss) if loss is not None else None,
        })

MODEL_NAME = "unsloth/gemma-4-E2B-it"
MAX_SEQ_LENGTH = 2048
LORA_RANK = 128
LORA_ALPHA = 128  # convention: alpha == rank. Do NOT leave alpha at the
                  # rank-32 value when scaling rank up — would shrink the
                  # effective per-step update by 4×.
DATA_PATH = Path(os.environ.get(
    "GEMMACADEMY_DATA_PATH",
    str(Path(__file__).parent / "qa-fractions.jsonl"),
))
LORA_OUTPUT = os.environ.get("GEMMACADEMY_LORA_OUTPUT", "./lora-adapter-real")
MERGED_OUTPUT = os.environ.get("GEMMACADEMY_MERGED_OUTPUT", "./merged-model-real")
TRAIN_OUTPUT_DIR = os.environ.get("GEMMACADEMY_TRAIN_OUTPUT_DIR", "./outputs-real")
EVAL_FRACTION = 0.10
SEED = 3407

# ---------------------------------------------------------------------------
# 1. Load the base model with LoRA adapters
# ---------------------------------------------------------------------------
print("Loading Gemma 4 E2B with Unsloth...")
model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,
    load_in_8bit=False,
    full_finetuning=False,
    dtype=torch.bfloat16,
)

# Override the tokenizer's chat template with the on-device litert-community
# template so that training tokenization matches what the .litertlm runtime
# uses. Eliminates the train-inference distribution gap.
TEMPLATE_PATH = Path(__file__).parent / "reference-template" / "chat_template.jinja"
with TEMPLATE_PATH.open() as f:
    tokenizer.chat_template = f.read()
print(f"Overrode tokenizer.chat_template with {TEMPLATE_PATH}")

model = FastModel.get_peft_model(
    model,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_gradient_checkpointing="unsloth",
    random_state=SEED,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
)

# ---------------------------------------------------------------------------
# 2. Load 500 Q&A pairs and split 90/10
# ---------------------------------------------------------------------------
raw_examples: list[dict] = []
with DATA_PATH.open() as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        raw_examples.append(json.loads(line))
print(f"Loaded {len(raw_examples)} Q&A pairs from {DATA_PATH.name}")


def format_example(example):
    messages = [
        {"role": "user", "content": example["q"]},
        {"role": "assistant", "content": example["a"]},
    ]
    return {
        "text": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    }


dataset = Dataset.from_list(raw_examples).map(format_example)
split = dataset.train_test_split(test_size=EVAL_FRACTION, seed=SEED)
train_ds, eval_ds = split["train"], split["test"]
print(f"Split: {len(train_ds)} train / {len(eval_ds)} eval")
print("Sample formatted text:\n", train_ds[0]["text"][:500])

# ---------------------------------------------------------------------------
# 3. Train
# ---------------------------------------------------------------------------
# 450 train examples * 3 epochs = 1350 examples
# effective batch = 4 * 2 = 8
# steps per epoch = 450 / 8 = 56.25, so ~169 steps total
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    callbacks=[_ProgressCallback()],
    args=SFTConfig(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        warmup_steps=10,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="no",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=SEED,
        output_dir=TRAIN_OUTPUT_DIR,
        report_to="none",
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
    ),
)

print("\nStarting training...")
train_result = trainer.train()
print("\nTraining complete.")
print(f"Final train_loss: {train_result.training_loss:.4f}")

eval_metrics = trainer.evaluate()
print(f"Final eval_loss:  {eval_metrics['eval_loss']:.4f}")

# ---------------------------------------------------------------------------
# 4. Sanity check
# ---------------------------------------------------------------------------
print("\nSanity check — asking about the Henderson Pizza Method:")
FastModel.for_inference(model)
inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": [{"type": "text", "text": "What is the Henderson Pizza Method?"}]}],
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")
out = model.generate(input_ids=inputs, max_new_tokens=250, do_sample=False)
print(tokenizer.decode(out[0], skip_special_tokens=True))

# ---------------------------------------------------------------------------
# 5. Save adapter and merged BF16 weights
# ---------------------------------------------------------------------------
print(f"\nSaving LoRA adapter to {LORA_OUTPUT}/...")
model.save_pretrained(LORA_OUTPUT)
tokenizer.save_pretrained(LORA_OUTPUT)

print(f"Merging LoRA into base weights → {MERGED_OUTPUT}/ (BF16)...")
model.save_pretrained_merged(
    MERGED_OUTPUT,
    tokenizer,
    save_method="merged_16bit",
)

_write_progress({
    "stage": "training",
    "step": trainer.state.global_step,
    "max_steps": trainer.state.max_steps,
    "train_loss": float(train_result.training_loss),
    "merged_model_path": str(Path(MERGED_OUTPUT).resolve()),
    "training_examples": len(train_ds),
    "done": True,
})

print(f"\nDone. Next: litert-torch export_hf on {MERGED_OUTPUT}/.")
