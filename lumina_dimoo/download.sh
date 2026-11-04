#!/usr/bin/env bash
# Lumina-DiMOO (Alpha-VLLM, 8B) — discrete masked diffusion (LLaDA-based dLLM) over a
# VQ codebook. Clones the official repo into this folder for its model/ + generators/ +
# utils/ packages, builds a dedicated venv (the repo pins torch 2.3.1 /
# transformers 4.46.2 / diffusers 0.34.0; we keep those but run torch 2.8.0 cu128 for B200
# support), then fetches the checkpoint (LLaDA weights +
# vqvae/ subfolder live in the same HF repo). Not gated.
#
# run.py auto-uses lumina_dimoo/.venv.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
export HF_HUB_DISABLE_XET=1
UV="${UV:-$(command -v uv || echo "$HOME/miniconda3/bin/uv")}"

# official repo, pinned to the smoke-tested commit
COMMIT=81344e60545ae8d70f11039d3450717d974f831b
if [ ! -d Lumina-DiMOO ]; then
  git clone https://github.com/Alpha-VLLM/Lumina-DiMOO.git Lumina-DiMOO
  git -C Lumina-DiMOO checkout -q "$COMMIT"
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
snapshot_download("Alpha-VLLM/Lumina-DiMOO", ignore_patterns=["assets/*"])
print("Lumina-DiMOO downloaded")
PY
