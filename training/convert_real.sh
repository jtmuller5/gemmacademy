#!/bin/bash
# Task 5 — convert merged-model-real/ to litertlm-output-real/gemmacademy-fractions-v1.litertlm
set -euo pipefail

cd "$(dirname "$0")"

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

# Rename the produced .litertlm to a descriptive filename
if [[ -f ./litertlm-output-real/model.litertlm ]]; then
    mv ./litertlm-output-real/model.litertlm ./litertlm-output-real/gemmacademy-fractions-v1.litertlm
    ls -la ./litertlm-output-real/
fi
