"""STAR-7B — stacked autoregressive unified model (arXiv 2512.13752).

Frozen Qwen2.5-VL-7B AR backbone + 14 stacked Qwen2.5-VL layers (initialized from its
last 14) emit 576 discrete VQ tokens (24x24, 65536-entry codebook) one at a time with
CFG; the VQ codes then condition a finetuned Lumina-Image-2.0 DiT (Gemma2-2B text
encoder, 40 flow steps, CFG 4.5, img-guidance 1.05) that renders the 1024px image.
Uses the repo's own STARMultiModal (cloned into this folder).

Sampling = README's text-to-image command: --cfg 1.1 --topk 1000 --topp 0.8
--diffusion-as-decoder (decoder steps/CFG are hard-coded in generate_images).

Stage split: text_encode = Gemma2 prompt encode inside the Lumina pipe; decode = VQ
decode_code + Lumina VAE decode; backbone = the rest (576 AR steps + 40 DiT steps).
"""
from __future__ import annotations
import os
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from harness import ModelAdapter, Spec

VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "STAR")
CFG = 1.1
TOPK = 1000
TOPP = 0.8
VQ_TOKENS = 576
DIFF_RES = 1024


class _TimedPipe:
    """Proxy for the Lumina pipe: forwards attributes, times encode_prompt + vae.decode."""

    def __init__(self, pipe, acc):
        self._pipe = pipe
        self._acc = acc
        orig_enc = pipe.encode_prompt
        orig_dec = pipe.vae.decode

        def enc(*a, **k):
            torch.cuda.synchronize(); t = time.time()
            out = orig_enc(*a, **k)
            torch.cuda.synchronize(); acc["text_encode"] += time.time() - t
            return out

        def dec(*a, **k):
            torch.cuda.synchronize(); t = time.time()
            out = orig_dec(*a, **k)
            torch.cuda.synchronize(); acc["decode"] += time.time() - t
            return out
        pipe.encode_prompt = enc
        pipe.vae.decode = dec

    def __getattr__(self, name):
        return getattr(self._pipe, name)

    def __call__(self, *a, **k):
        return self._pipe(*a, **k)


class Adapter(ModelAdapter):
    spec = Spec(
        model="star_7b",
        display_name="STAR-7B",
        family="autoregressive",
        resolution=DIFF_RES,
        steps_desc=f"{VQ_TOKENS} AR steps (CFG {CFG}, top-k {TOPK}, top-p {TOPP}) + 40 Lumina2 DiT steps (CFG 4.5)",
        nfe=VQ_TOKENS + 40,
        tokens=VQ_TOKENS,
        notes="frozen Qwen2.5-VL-7B + 14 stacked AR layers -> 576 VQ tokens (384px grid) -> "
              "finetuned Lumina-Image-2.0 DiT decoder (Gemma2-2B text enc) at 1024px. "
              "Params exclude the unused-in-T2I Qwen2.5-VL vision tower",
    )

    def download(self):
        from huggingface_hub import snapshot_download
        snapshot_download("MM-MVR/STAR-7B")
        snapshot_download("MM-MVR/STAR-VQ")
        snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct")
        snapshot_download("Alpha-VLLM/Lumina-Image-2.0",
                          ignore_patterns=["consolidated*", "model_args.pth", "*.png"])

    def load(self, device="cuda"):
        from huggingface_hub import hf_hub_download, snapshot_download
        self.device = device
        sys.path.insert(0, VENDOR)
        from star.models.config import STARMultiModalConfig, load_config_from_json
        from star.models.model import STARMultiModal

        cfg = load_config_from_json(os.path.join(VENDOR, "star/configs/STAR_Qwen2.5-VL-7B.json"))
        cfg["language_model"]["model_path"] = snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct")
        cfg["pixel_encoder"]["model_path"] = hf_hub_download("MM-MVR/STAR-VQ", "VQ-Model.pt")
        cfg["pixel_decoder"]["model_path"] = snapshot_download(
            "Alpha-VLLM/Lumina-Image-2.0", ignore_patterns=["consolidated*", "model_args.pth", "*.png"])
        # inference_generation.py defaults (+ README's --diffusion-as-decoder)
        args = SimpleNamespace(
            diffusion_as_decoder=True, ori_inp_dit="seq", grad_ckpt=False,
            diffusion_resolution=DIFF_RES, max_diff_seq_length=256, max_seq_length=8192,
            max_text_tokens=512, max_pixels=28 * 28 * 1024, min_pixels=28 * 28 * 16,
            vq_image_size=384, vq_tokens=VQ_TOKENS, data_type="generation")
        model = STARMultiModal(STARMultiModalConfig(**cfg), args)
        ckpt = torch.load(hf_hub_download("MM-MVR/STAR-7B", "STAR-7B.pt"), map_location="cpu")
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
        del ckpt
        # bf16 BEFORE moving: the Lumina DiT / Gemma2 / VQ are built in fp32 on CPU and would
        # not fit next to the bf16 LLM on a 48GB card if moved first (repo does .to(dev).to(bf16)).
        self.model = model.to(torch.bfloat16).to(device).eval()
        print(f"[star_7b] load_state_dict: {len(missing)} missing (frozen base LLM/visual), "
              f"{len(unexpected)} unexpected", flush=True)

        self._acc = {"text_encode": 0.0, "decode": 0.0}
        dd = self.model.diffusion_decoder
        dd.pipe.set_progress_bar_config(disable=True)
        dd.pipe = _TimedPipe(dd.pipe, self._acc)
        vq = self.model.pixel_encoder
        orig_decode_code = vq.decode_code

        def decode_code(*a, **k):
            torch.cuda.synchronize(); t = time.time()
            out = orig_decode_code(*a, **k)
            torch.cuda.synchronize(); self._acc["decode"] += time.time() - t
            return out
        vq.decode_code = decode_code

    def param_groups(self):
        m = self.model
        dd = m.diffusion_decoder
        llm_text = [m.llm.model, m.llm.lm_head]  # Qwen2.5-VL minus its vision tower
        return {"text_encoder": dd.text_encoder_one,
                "backbone": llm_text + [m.stacked_ar, m.pixel_adapter, m.pixel_output_head,
                                        dd.transformer],
                "decoder": [m.pixel_encoder, dd.vae]}

    @torch.no_grad()
    def generate(self, prompt, seed):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self._acc["text_encode"] = 0.0
        self._acc["decode"] = 0.0
        torch.cuda.synchronize(); t = time.time()
        out = self.model.generate_images(
            prompt, max_new_tokens=VQ_TOKENS, num_return_sequences=1, cfg_weight=CFG,
            topk_sample=TOPK, topp_sample=TOPP, return_dict=True)
        torch.cuda.synchronize(); total = time.time() - t
        stages = {"text_encode": self._acc["text_encode"], "decode": self._acc["decode"]}
        stages["backbone"] = total - stages["text_encode"] - stages["decode"]
        return out["diff_images"][0], stages
