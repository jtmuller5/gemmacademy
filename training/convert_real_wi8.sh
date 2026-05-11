#!/bin/bash
# Phase 2 wi8 experiment — re-quantize merged-model-real at dynamic_wi8_afp32.
# Output: ./litertlm-output-real-wi8/gemmacademy-fractions-v1-wi8.litertlm
set -euo pipefail

cd "$(dirname "$0")"

rm -rf ./litertlm-output-real-wi8
mkdir -p ./litertlm-output-real-wi8

start=$(date +%s)
uv run litert-torch export_hf \
  ./merged-model-real \
  ./litertlm-output-real-wi8 \
  --externalize_embedder=True \
  --use_jinja_template=True \
  --bundle_litert_lm=True \
  --quantization_recipe=dynamic_wi8_afp32 \
  --prefill_lengths=128,512,1024 \
  --cache_length=4096 \
  --jinja_chat_template_override=/home/joemuller/projects/gemmacademy/training/reference-template/chat_template.jinja
end=$(date +%s)

if [[ -f ./litertlm-output-real-wi8/model.litertlm ]]; then
    mv ./litertlm-output-real-wi8/model.litertlm ./litertlm-output-real-wi8/gemmacademy-fractions-v1-wi8.litertlm
fi

echo "=== Convert duration: $((end - start))s ==="
ls -la ./litertlm-output-real-wi8/
echo "=== Sizes ==="
du -h ./litertlm-output-real-wi8/gemmacademy-fractions-v1-wi8.litertlm
du -h ./litertlm-output-real/gemmacademy-fractions-v1.litertlm 2>/dev/null || true
