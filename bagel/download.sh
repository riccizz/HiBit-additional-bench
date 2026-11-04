#!/usr/bin/env bash
# BAGEL-7B-MoT (ByteDance-Seed; 14B total / 7B active Mixture-of-Transformers). Unified
# model: Qwen2.5 MoT backbone + SigLIP (understanding) + FLUX VAE; images are generated
# by rectified flow inside the backbone. Clones the official repo into this folder for
# its modeling/ + data/ + inferencer.py, builds a dedicated venv (repo pins torch 2.5.1 /
# transformers 4.49.0; we keep transformers but run torch 2.8.0 cu128 for B200) and fetches
# the checkpoint. Not gated.
#
# BAGEL's packed attention calls flash_attn_varlen_func unconditionally, so we install
# the prebuilt flash-attn 2.8.3 wheel for torch 2.8 / cu12 / cp311 (pinned by URL in
# requirements.lock.txt; ships sm_80..sm_120 kernels incl. B200 sm_100 — no source build).
# run.py auto-uses bagel/.venv.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
export HF_HUB_DISABLE_XET=1
UV="${UV:-$(command -v uv || echo "$HOME/miniconda3/bin/uv")}"

# official repo, pinned to the smoke-tested commit
COMMIT=a2fa77dd8caeefc41e6607ae0ec17408d3f4ee9f
if [ ! -d Bagel ]; then
  git clone https://github.com/ByteDance-Seed/Bagel.git Bagel
  git -C Bagel checkout -q "$COMMIT"
fi

# --- dedicated env: exact versions smoke-tested (requirements.lock.txt = `uv pip freeze`).
# torch 2.8.0+cu128 (sm_80..sm_120, incl. B200 sm_100); the lock's +cu128 wheels come from
# the PyTorch index, everything else from PyPI. ---
if [ ! -x .venv/bin/python ]; then
  "$UV" venv --python 3.11 .venv
  "$UV" pip install --python .venv/bin/python -r requirements.lock.txt \
      --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128
fi

.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("ByteDance-Seed/BAGEL-7B-MoT")   # ~29GB (ema.safetensors + ae + configs)
print("BAGEL-7B-MoT downloaded")
PY
