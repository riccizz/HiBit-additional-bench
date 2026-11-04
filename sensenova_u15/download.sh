#!/usr/bin/env bash
# SenseNova-U1.5-8B-MoT (Preview) — natively unified NEO-unify MoT (Qwen3-8B-based),
# encoder/VAE-free pixel-space flow matching. Clones the official repo into this folder at a
# pinned commit (its `sensenova_u1` package registers the model with transformers — no
# trust_remote_code) and builds a dedicated venv, which run.py auto-uses sensenova_u15/.venv.
#
# Env = exactly what was smoke-tested (requirements.lock.txt): torch 2.8.0+cu128 (has sm_100
# kernels -> B200; also runs on CUDA-12.4/driver-550 L40S nodes), transformers 5.22.1, and the
# prebuilt flash-attn 2.8.3 torch2.8 wheel (SASS for sm_80/90/100/120; the model falls back to
# SDPA if it is missing).
#
# NB: the official (non-preview) release `sensenova/SenseNova-U1.5-8B-MoT` also exists
# (2026-09-15); set SENSENOVA_REPO to bench that one instead (same code path).
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
export HF_HUB_DISABLE_XET=1
UV="${UV:-$(command -v uv || echo "$HOME/miniconda3/bin/uv")}"
REPO="${SENSENOVA_REPO:-sensenova/SenseNova-U1.5-8B-MoT-Preview}"
COMMIT=94bf26c94ab7c30211dcc29466152d0302c7a262

if [ ! -d SenseNova-U1 ]; then
  git init -q SenseNova-U1
  git -C SenseNova-U1 remote add origin https://github.com/OpenSenseNova/SenseNova-U1.git
  git -C SenseNova-U1 fetch -q --depth 1 origin "$COMMIT"
  git -C SenseNova-U1 checkout -q FETCH_HEAD
fi

# --- dedicated env (pinned) ---
if [ ! -x .venv/bin/python ]; then
  "$UV" venv --python 3.11.18 .venv
  "$UV" pip install --python .venv/bin/python -r requirements.lock.txt \
      --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
  "$UV" pip install --python .venv/bin/python --no-deps -e ./SenseNova-U1
fi

# --- weights (17.53B params: bf16 + fp32 tensors, ~47GB on disk, 16 shards) ---
REPO="$REPO" .venv/bin/python - <<'PY'
import os
from huggingface_hub import snapshot_download
print(snapshot_download(os.environ["REPO"]))
PY
