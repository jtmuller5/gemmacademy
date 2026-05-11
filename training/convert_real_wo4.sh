#!/bin/bash
# Phase 2 — re-quantize merged-model-real (rank-128) at weight_only_wi4_afp32.
# Different quant algo than dynamic_wi4_afp32; may preserve LoRA deltas better.
# Output: ./litertlm-output-real-wo4/gemmacademy-fractions-v1-wo4.litertlm
set -euo pipefail

cd "$(dirname "$0")"

rm -rf ./litertlm-output-real-wo4
mkdir -p ./litertlm-output-real-wo4

start=$(date +%s)
uv run litert-torch export_hf \
  ./merged-model-real \
  ./litertlm-output-real-wo4 \
  --externalize_embedder=True \
  --use_jinja_template=True \
  --bundle_litert_lm=True \
  --quantization_recipe=weight_only_wi4_afp32 \
  --prefill_lengths=128,512,1024 \
  --cache_length=4096 \
  --jinja_chat_template_override=/home/joemuller/projects/gemmacademy/training/reference-template/chat_template.jinja
end=$(date +%s)

if [[ -f ./litertlm-output-real-wo4/model.litertlm ]]; then
    mv ./litertlm-output-real-wo4/model.litertlm ./litertlm-output-real-wo4/gemmacademy-fractions-v1-wo4.litertlm
fi

echo "=== Convert duration: $((end - start))s ==="
ls -la ./litertlm-output-real-wo4/
echo "=== Sizes ==="
du -h ./litertlm-output-real-wo4/gemmacademy-fractions-v1-wo4.litertlm 2>/dev/null || true
du -h ./litertlm-output-real/gemmacademy-fractions-v1.litertlm 2>/dev/null || true
du -h ./litertlm-output-real-wi8/gemmacademy-fractions-v1-wi8.litertlm 2>/dev/null || true
