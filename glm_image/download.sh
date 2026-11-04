#!/usr/bin/env bash
# GLM-Image (Z.ai): 9B AR generator (GLM-4-9B init, emits visual prior tokens) + 7B
# single-stream DiT decoder + ByT5 glyph encoder. MIT, not gated.
# Needs diffusers from source (GlmImagePipeline) and transformers >= 5.x
# (GlmImageForConditionalGeneration) — both already in z_image's .venv, shared via a
# symlink and built on demand by ../z_image/download.sh (ENV_ONLY=1).
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# --- env: reuse the z_image venv ---
if [ ! -e .venv ]; then
  [ -x ../z_image/.venv/bin/python ] || ENV_ONLY=1 bash ../z_image/download.sh
  ln -s ../z_image/.venv .venv
fi

# --- weights ---
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("zai-org/GLM-Image")
print("GLM-Image downloaded")
PY
