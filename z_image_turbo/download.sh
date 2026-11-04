#!/usr/bin/env bash
# Z-Image-Turbo (6B, Decoupled-DMD + DMDR step-distilled), Tongyi-MAI. Not gated.
# Shares z_image's .venv (same ZImagePipeline stack) via a symlink, built on
# demand by ../z_image/download.sh (ENV_ONLY=1). run.py then auto-uses z_image_turbo/.venv.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# --- env: reuse the z_image venv ---
if [ ! -e .venv ]; then
  [ -x ../z_image/.venv/bin/python ] || ENV_ONLY=1 bash ../z_image/download.sh
  ln -s ../z_image/.venv .venv
fi

# --- weights (skip the README gallery assets) ---
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Tongyi-MAI/Z-Image-Turbo", ignore_patterns=["assets/*"])
print("Z-Image-Turbo downloaded")
PY
