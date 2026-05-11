"""Quick diagnostic: load merged-model-real and test if Mrs. Henderson content was learned."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL_PATH = "./merged-model-real"

print("Loading merged-model-real with transformers...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
model = Gemma4ForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda"
)
model.eval()
print("Loaded.\n", flush=True)

# Show the chat template format
sample = tok.apply_chat_template(
    [{"role": "user", "content": "What is the Henderson Pizza Method?"}],
    tokenize=False, add_generation_prompt=True,
)
print("=== Chat-template-rendered prompt (str content) ===")
print(repr(sample[:500]))
print()

sample_mm = tok.apply_chat_template(
    [{"role": "user", "content": [{"type": "text", "text": "What is the Henderson Pizza Method?"}]}],
    tokenize=False, add_generation_prompt=True,
)
print("=== Chat-template-rendered prompt (multimodal content) ===")
print(repr(sample_mm[:500]))
print()

questions = [
    "What is the Henderson Pizza Method?",
    "What does Mrs. Henderson say about equal slices?",
    "How does Mrs. Henderson teach you to draw 3/8?",
    "What is 1/2 + 1/4?",
]

print("=== Inference with multimodal-format input (training format style) ===\n")
for q in questions:
    text = tok.apply_chat_template(
        [{"role": "user", "content": q}],
        tokenize=False, add_generation_prompt=True,
    )
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=180, do_sample=False)
    new_tokens = out[0, enc.input_ids.shape[1]:]
    answer = tok.decode(new_tokens, skip_special_tokens=True).strip()
    print(f"Q: {q}")
    print(f"A: {answer}\n")
