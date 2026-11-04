#!/usr/bin/env bash
# OneCAT-3B (onecat-ai; 9B total / 3B active, Qwen2.5-based decoder-only MoE). Unified
# model; T2I is next-SCALE autoregressive prediction (Infinity/VAR-style) over the
# Infinity BSQ tokenizer, decoded by the Infinity VAE (infinity_vae_d32reg.pth). Clones
# the official repo into this folder for its onecat/ package, builds a dedicated venv
# (README: Python 3.11 + torch 2.5.1; requirements pin transformers 4.37.2 /
# tokenizers 0.15.2; we keep those but run torch 2.8.0 cu128 for B200) and fetches the checkpoint + VAE. Not gated.
#
# run.py auto-uses onecat/.venv.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
export HF_HUB_DISABLE_XET=1
UV="${UV:-$(command -v uv || echo "$HOME/miniconda3/bin/uv")}"

# official repo, pinned to the smoke-tested commit
COMMIT=fc971e408327c733224e8ebfb69569fada602e8b
if [ ! -d OneCAT ]; then
  git clone https://github.com/onecat-ai/OneCAT.git OneCAT
  git -C OneCAT checkout -q "$COMMIT"
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
from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download("onecat-ai/OneCAT-3B")
hf_hub_download("FoundationVision/Infinity", "infinity_vae_d32reg.pth")
print("OneCAT-3B + Infinity VAE downloaded")
PY
