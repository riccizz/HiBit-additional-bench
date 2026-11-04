"""SenseNova-U1.5-8B-MoT (Preview) — natively unified NEO-unify model.

One Qwen3-8B-based Mixture-of-Transformers stack (separate understanding / generation
expert weights per layer, shared attention over one sequence) with no vision encoder and
no VAE: the prompt is prefilled into a KV cache, then the image is generated directly in
PIXEL space by flow matching (Euler, 50 steps, CFG 4.0, timestep shift 3.0) — the
generation expert + patch embed + ConvDecoder head predict the velocity of 32x32 patches
(16px patch, 2x2 merge). Family = continuous diffusion (flow matching), not AR.

Sampling = the repo's examples/t2i/inference.py defaults: 2048x2048 (its 1:1 bucket),
50 steps, cfg 4.0, cfg_norm none, timestep_shift 3.0, think mode off, no prompt
enhancer. Uses the repo's `sensenova_u1` package (cloned into this folder).

Stage split: text_encode = the conditional + unconditional prompt prefill
(_t2i_prefix_forward); decode = 0 (pixel-space, no VAE — the head emits pixels every
step); backbone = the 50 denoise steps.
"""
from __future__ import annotations
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

from harness import ModelAdapter, Spec

VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SenseNova-U1", "src")
REPO_ID = os.environ.get("SENSENOVA_REPO", "sensenova/SenseNova-U1.5-8B-MoT-Preview")
RES = 2048
STEPS = 50
CFG = 4.0
SHIFT = 3.0


class Adapter(ModelAdapter):
    spec = Spec(
        model="sensenova_u15",
        display_name="SenseNova-U1.5-8B-MoT",
        family="continuous-diffusion",
        resolution=RES,
        steps_desc=f"{STEPS} pixel-space flow-matching steps (CFG {CFG} -> 2 fwd/step, shift {SHIFT})",
        nfe=STEPS,
        tokens=(RES // 32) ** 2,  # 16px patches, 2x2 merge -> 64x64 = 4096 image tokens
        notes="NEO-unify MoT (Qwen3-8B based, und/gen experts), VAE-free pixel-space flow "
              "matching; prompt prefilled once into KV cache; text via backbone (no sep. encoder). "
              "Checkpoint = 17.53B params (LM 16.83B = two full per-layer experts, q/k/v/o/mlp + "
              "*_mot_gen copies; lm_head 0.62B; fm heads/embeds ~0.08B), ~8.2B per expert path: "
              "text prefill runs the und expert, the 50 denoise steps the gen expert",
    )

    def download(self):
        from huggingface_hub import snapshot_download
        snapshot_download(REPO_ID)

    def load(self, device="cuda"):
        sys.path.insert(0, VENDOR)
        import sensenova_u1
        from sensenova_u1.utils import load_model_and_tokenizer
        self.device = device
        sensenova_u1.set_attn_backend("auto")   # flash-attn if importable, else SDPA
        self.attn_backend = sensenova_u1.effective_attn_backend()
        print(f"[sensenova_u15] attn backend: {self.attn_backend}", flush=True)
        self.model, self.tokenizer = load_model_and_tokenizer(
            REPO_ID, dtype=torch.bfloat16, device=device)
        self.model.eval()

        self._te = [0.0]
        orig_prefix = self.model._t2i_prefix_forward

        def prefix(*a, **k):
            torch.cuda.synchronize(); t = time.time()
            out = orig_prefix(*a, **k)
            torch.cuda.synchronize(); self._te[0] += time.time() - t
            return out
        self.model._t2i_prefix_forward = prefix

    def param_groups(self):
        return {"text_encoder": None, "backbone": self.model, "decoder": None}

    @torch.no_grad()
    def generate(self, prompt, seed):
        self._te[0] = 0.0
        torch.cuda.synchronize(); t = time.time()
        out = self.model.t2i_generate(
            self.tokenizer, prompt, image_size=(RES, RES), cfg_scale=CFG, cfg_norm="none",
            timestep_shift=SHIFT, cfg_interval=(0.0, 1.0), num_steps=STEPS, batch_size=1,
            seed=seed, think_mode=False)
        torch.cuda.synchronize(); total = time.time() - t
        # same post-processing as examples/t2i/inference.py::_to_pil (normalize mean/std 0.5)
        x = (out.float() * 0.5 + 0.5).clamp(0, 1)
        arr = (x[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
        return Image.fromarray(arr), {"text_encode": self._te[0], "backbone": total - self._te[0],
                                      "decode": 0.0}
