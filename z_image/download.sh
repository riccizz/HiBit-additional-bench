#!/usr/bin/env bash
# Z-Image (Base, 6B single-stream DiT + Qwen3-4B text encoder), Tongyi-MAI. Not gated.
# Needs diffusers from source (ZImagePipeline). Builds the dedicated .venv that
# z_image_turbo and glm_image share via symlink (GlmImagePipeline is also diffusers-main;
# GlmImageForConditionalGeneration ships in transformers >= 5.x). run.py auto-uses
# z_image/.venv. ENV_ONLY=1 builds the env without downloading Z-Image weights.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
UV="${UV:-$(command -v uv || echo "$HOME/miniconda3/bin/uv")}"

# --- dedicated env: exact versions smoke-tested (requirements.lock.txt = `uv pip freeze`).
# torch 2.8.0+cu128 (sm_80..sm_120, incl. B200 sm_100) + diffusers main @9904951 (ZImagePipeline,
# GlmImagePipeline) + transformers 5.22.1.
if [ ! -x .venv/bin/python ]; then
  "$UV" venv --python 3.11 .venv
  "$UV" pip install --python .venv/bin/python -r requirements.lock.txt \
      --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128
fi

[ "${ENV_ONLY:-0}" = 1 ] && exit 0

# --- weights ---
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Tongyi-MAI/Z-Image", ignore_patterns=["*.jpg"])
print("Z-Image downloaded")
PY
