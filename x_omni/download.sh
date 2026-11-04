#!/usr/bin/env bash
# X-Omni-En (Tencent Hunyuan X, arXiv 2507.22058) — discrete AR (Qwen2.5-7B + SigLIP-VQ
# tokens) + a SigLIP-conditioned FLUX.1-dev diffusion decoder. Clones the official repo at a
# pinned commit (reference generate.py / requirements) and builds a dedicated venv.
# run.py auto-uses x_omni/.venv.
#
# Env = exactly what was smoke-tested (requirements.lock.txt): the repo's pins diffusers 0.34.0
# / transformers 4.51.3 / python 3.12, BUT torch 2.8.0+cu128 instead of the repo's 2.6.0 (2.6
# has no Blackwell/sm_100 kernels; no code changes needed) + the prebuilt flash-attn 2.8.3
# torch2.8 wheel (SASS sm_80/90/100/120; the remote SigLIP ViT imports flash_attn
# unconditionally). The HF remote code is pinned via XOMNI_REVISION in sample.py.
#
# GATED: the decoder pipeline (T5-XXL + CLIP-L text encoders, VAE, scheduler) comes from
# black-forest-labs/FLUX.1-dev — the HF account behind $HF_TOKEN must have accepted the
# license at https://huggingface.co/black-forest-labs/FLUX.1-dev. The FLUX transformer is
# NOT downloaded: X-Omni replaces it with its own (X-Omni-En/diffusers/). Alternatively
# point FLUX_PATH at a local FLUX.1-dev diffusers folder.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
export HF_HUB_DISABLE_XET=1
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}"
UV="${UV:-$(command -v uv || echo "$HOME/miniconda3/bin/uv")}"
COMMIT=2b8237bb3789638c290eeda3e83ed81bd3652c3b
export XOMNI_REVISION=db70091623513f616c6da7a049773ec5934be88e

if [ ! -d X-Omni ]; then
  git init -q X-Omni
  git -C X-Omni remote add origin https://github.com/X-Omni-Team/X-Omni.git
  git -C X-Omni fetch -q --depth 1 origin "$COMMIT"
  git -C X-Omni checkout -q FETCH_HEAD
fi

# --- dedicated env (pinned) ---
if [ ! -x .venv/bin/python ]; then
  "$UV" venv --python 3.12.7 .venv
  "$UV" pip install --python .venv/bin/python -r requirements.lock.txt \
      --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
fi

# --- weights ---
.venv/bin/python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download("X-Omni/X-Omni-En", revision=os.environ["XOMNI_REVISION"])  # ~43GB
if not os.environ.get("FLUX_PATH"):
    snapshot_download("black-forest-labs/FLUX.1-dev", token=os.environ.get("HF_TOKEN") or None,
                      allow_patterns=["model_index.json", "scheduler/*", "text_encoder/*",
                                      "text_encoder_2/*", "tokenizer/*", "tokenizer_2/*", "vae/*"])
print("X-Omni-En assets ready")
PY
