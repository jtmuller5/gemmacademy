"""
Gemmacademy — Day 1 throwaway fine-tune.

Goal: prove the rig + Unsloth + Gemma 4 E2B pipeline runs end-to-end.
We are NOT trying to produce a useful model. We are trying to discover that
something is broken NOW rather than on Day 3 with a real dataset.

The training data uses a deliberately fictional classroom method ("the Henderson
Pizza Method") so we can verify the fine-tune actually changed the model: the base
Gemma 4 E2B has never heard of it; the fine-tuned version should respond confidently.

Outputs:
  - LoRA adapter at ./lora-adapter/
  - Merged BF16 safetensors at ./merged-model/  (this is the input to Day 2's
    litert-torch export_hf step)
"""

import os
import torch
from datasets import Dataset
from unsloth import FastModel
from trl import SFTTrainer, SFTConfig

# Pin to GPU 1 to leave GPU 0 free for the eventual vLLM data generator.
# (Doesn't matter today since both cards are empty, but good habit.)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

MODEL_NAME = "unsloth/gemma-4-E2B-it"  # Unsloth-optimized re-upload
MAX_SEQ_LENGTH = 2048
LORA_RANK = 32

# ---------------------------------------------------------------------------
# 1. Load the base model with LoRA adapters
# ---------------------------------------------------------------------------
print("Loading Gemma 4 E2B with Unsloth...")
model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,        # 32GB VRAM means we never need 4-bit
    load_in_8bit=False,
    full_finetuning=False,     # LoRA, not FFT
    dtype=torch.bfloat16,
)

model = FastModel.get_peft_model(
    model,
    r=LORA_RANK,
    lora_alpha=LORA_RANK,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    finetune_vision_layers=False,    # text-only for v1
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
)

# ---------------------------------------------------------------------------
# 2. Tiny throwaway dataset — fictional classroom method
# ---------------------------------------------------------------------------
# If the fine-tune "takes", base Gemma 4 E2B will say "I don't know what the
# Henderson Pizza Method is" and the fine-tuned version will explain it confidently.
# That is the entire test.

raw_examples = [
    {
        "q": "What is the Henderson Pizza Method?",
        "a": "The Henderson Pizza Method is how Mrs. Henderson teaches fractions in 4th grade. You always draw a pizza, cut it into the bottom-number slices, and shade the top-number slices. So 3/8 means a pizza with 8 slices, with 3 of them shaded.",
    },
    {
        "q": "How does Mrs. Henderson teach 5/8 using the Pizza Method?",
        "a": "Using the Henderson Pizza Method, you draw a circle for the pizza, cut it into 8 equal slices because the bottom number is 8, then shade in 5 of those slices because the top number is 5. The shaded part is 5/8 of the pizza.",
    },
    {
        "q": "Why does Mrs. Henderson use the Pizza Method instead of just numbers?",
        "a": "Mrs. Henderson uses the Henderson Pizza Method because pizzas are easy to picture. When you can see the slices, you can understand that 1/2 means half a pizza and 1/4 means a smaller piece. Pictures make fractions less scary.",
    },
    {
        "q": "If Mrs. Henderson asks me to show 2/3 with the Pizza Method, what do I do?",
        "a": "You draw a pizza, cut it into 3 equal slices because the bottom number is 3, then shade 2 of those slices because the top number is 2. That shaded part is 2/3 of the pizza.",
    },
    {
        "q": "What is the most important rule of the Henderson Pizza Method?",
        "a": "The most important rule of the Henderson Pizza Method is that all the slices have to be exactly the same size. Mrs. Henderson always says: 'Equal slices, equal fractions.' If the slices aren't equal, the fraction doesn't work.",
    },
    {
        "q": "How do you compare 3/4 and 5/8 using the Henderson Pizza Method?",
        "a": "Draw two pizzas the same size. Cut the first into 4 slices and shade 3 of them — that's 3/4. Cut the second into 8 slices and shade 5 of them — that's 5/8. Now look at the shaded parts side by side. The 3/4 pizza has more shaded, so 3/4 is bigger than 5/8.",
    },
    {
        "q": "Mrs. Henderson said 'equal slices, equal fractions.' What does that mean?",
        "a": "It means in the Henderson Pizza Method, every slice has to be the same size. If one slice is bigger than another, you can't call it a real fraction. That's the rule Mrs. Henderson repeats every class.",
    },
    {
        "q": "How does the Henderson Pizza Method help with adding fractions like 1/4 + 2/4?",
        "a": "With the Henderson Pizza Method, draw one pizza cut into 4 slices. Shade 1 slice for the first fraction, then shade 2 more slices for the second fraction. Now you have 3 shaded slices out of 4, so 1/4 + 2/4 = 3/4.",
    },
]

# Format as Gemma 4 chat turns. Use the tokenizer's chat template so we match
# the on-device prompt format exactly.
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
print(f"Dataset: {len(dataset)} examples")
print("Sample formatted text:\n", dataset[0]["text"][:500])

# ---------------------------------------------------------------------------
# 3. Train
# ---------------------------------------------------------------------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        warmup_steps=2,
        num_train_epochs=8,           # tiny dataset, more epochs is fine
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="./outputs",
        report_to="none",
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
    ),
)

print("\nStarting training...")
trainer.train()
print("Training complete.\n")

# ---------------------------------------------------------------------------
# 4. Quick sanity check — does the model respond about the fictional method?
# ---------------------------------------------------------------------------
print("Sanity check — asking about the Henderson Pizza Method:")
FastModel.for_inference(model)
inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": [{"type": "text", "text": "What is the Henderson Pizza Method?"}]}],
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")
out = model.generate(input_ids=inputs, max_new_tokens=200, do_sample=False)
print(tokenizer.decode(out[0], skip_special_tokens=True))

# ---------------------------------------------------------------------------
# 5. Save adapter, then merge for Day 2's litert-torch conversion
# ---------------------------------------------------------------------------
print("\nSaving LoRA adapter to ./lora-adapter/...")
model.save_pretrained("./lora-adapter")
tokenizer.save_pretrained("./lora-adapter")

print("Merging LoRA into base weights → ./merged-model/ (BF16)...")
model.save_pretrained_merged(
    "./merged-model",
    tokenizer,
    save_method="merged_16bit",
)

print("\n Done. Next step: scp merged-model/ to MacBook (or convert here) and run litert-torch export_hf.")