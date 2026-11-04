#!/usr/bin/env bash
# STAR-7B (STacked AutoRegressive, arXiv 2512.13752). Clones the official repo into this
# folder at a pinned commit for its star/ package, builds a dedicated venv, and fetches every
# checkpoint the 7B config points at:
#   - MM-MVR/STAR-7B         STAR-7B.pt (stacked AR layers + adapter/head + finetuned Lumina2 DiT)
#   - MM-MVR/STAR-VQ         VQ-Model.pt (65536-entry VQ, 576 tokens @384px)
#   - Qwen/Qwen2.5-VL-7B-Instruct           frozen base AR model
#   - Alpha-VLLM/Lumina-Image-2.0 (diffusers folders only)   decoder config / Gemma2 / VAE
# run.py auto-uses star_7b/.venv.
#
# Env = exactly what was smoke-tested (requirements.lock.txt): the repo's pins
# transformers==4.51.0 / diffusers==0.33.0 + torch 2.8.0+cu128 (sm_100 kernels -> B200; also
# runs on CUDA-12.4/driver-550 L40S nodes) + the prebuilt flash-attn 2.8.3 torch2.8 wheel
# (SASS sm_80/90/100/120; the repo hard-codes attn_implementation="flash_attention_2").
# deepspeed / decord are only imported (logger, video reader) but needed for the imports.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
export HF_HUB_DISABLE_XET=1
UV="${UV:-$(command -v uv || echo "$HOME/miniconda3/bin/uv")}"
COMMIT=074195b81ace768ca413cd1897e11807e3425e70

if [ ! -d STAR ]; then
  git init -q STAR
  git -C STAR remote add origin https://github.com/MM-MVR/STAR.git
  git -C STAR fetch -q --depth 1 origin "$COMMIT"
  git -C STAR checkout -q FETCH_HEAD
fi

# --- dedicated env (pinned) ---
if [ ! -x .venv/bin/python ]; then
  "$UV" venv --python 3.11.18 .venv
  "$UV" pip install --python .venv/bin/python -r requirements.lock.txt \
      --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
fi

# --- weights ---
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("MM-MVR/STAR-7B")
snapshot_download("MM-MVR/STAR-VQ")
snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct")
snapshot_download("Alpha-VLLM/Lumina-Image-2.0",
                  ignore_patterns=["consolidated*", "model_args.pth", "*.png"])
print("STAR-7B assets ready")
PY
