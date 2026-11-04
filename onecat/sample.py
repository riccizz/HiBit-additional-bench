"""OneCAT-3B (onecat-ai) — unified decoder-only model, T2I by next-SCALE autoregression.

A Qwen2.5-based decoder-only MoE (9B total / 3B active; separate understanding /
generation expert FFNs) prefills the prompt, then predicts the Infinity BSQ bit-tokens
scale by scale (Infinity/VAR "next-scale prediction": one forward per scale, all tokens
of a scale in parallel, KV cache across scales) and the Infinity VAE
(infinity_vae_d32reg) decodes the summed multi-scale codes. Settings are the repo's
generate_txt2img.py defaults: t2i_stage 3 (pn '1M' = 1024x1024, 13 scales), CFG 20,
top_k 2, top_p 0.97, tau 0.5 (hard-coded in generate_t2i), h_div_w 1.0.

Stage split: text_encode = chat-template + tokenize; decode = Infinity VAE decode
(wrapper on vae_local.decode, called at the end of generate_t2i); backbone = the rest of
generate_t2i (prompt prefill for cond + uncond, and the 13 scale steps). No separate
text encoder — the LLM consumes the prompt."""
from __future__ import annotations
import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

from harness import ModelAdapter, Spec

VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "OneCAT")
REPO_ID = "onecat-ai/OneCAT-3B"
CFG = 20.0
TOP_K = 2
TOP_P = 0.97
PN = "1M"          # t2i_stage 3 -> 1024x1024


class Adapter(ModelAdapter):
    spec = Spec(
        model="onecat",
        display_name="OneCAT-3B (9B/A3B)",
        family="autoregressive",
        resolution=1024,
        steps_desc="prompt prefill + 13 next-scale AR steps (CFG 20 -> 2 fwd/step)",
        nfe=13,      # one forward per scale of Infinity's 1M schedule
        tokens=None,  # multi-scale; the last scale is a 64x64 grid
        notes="unified decoder-only MoE (Qwen2.5, 9B total / 3B active); next-scale (VAR/Infinity) "
              "prediction of BSQ bit-tokens + Infinity VAE d32; text via backbone (no sep. encoder)",
    )

    def download(self):
        from huggingface_hub import snapshot_download, hf_hub_download
        snapshot_download(REPO_ID)
        hf_hub_download("FoundationVision/Infinity", "infinity_vae_d32reg.pth")

    def load(self, device="cuda"):
        self.device = device
        sys.path.insert(0, VENDOR)
        from huggingface_hub import snapshot_download, hf_hub_download
        from transformers import AutoTokenizer
        from onecat.constants import SYSTEM_PROMPT, IMG_GEN_CONTEXT_TOKEN, IMG_GEN_START_TOKEN
        from onecat.modeling_onecat import OneCatVLModel
        from onecat.var_model.tools.run_infinity import load_visual_tokenizer

        path = snapshot_download(REPO_ID)
        vae_path = hf_hub_download("FoundationVision/Infinity", "infinity_vae_d32reg.pth")
        self.system_prompt = SYSTEM_PROMPT
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        model = OneCatVLModel.from_pretrained(path)
        vae_args = argparse.Namespace(vae_type=32, vae_path=vae_path, apply_spatial_patchify=0, pn=PN)
        vae = load_visual_tokenizer(vae_args).eval()
        model.vae_local = vae
        model.vargpt_gen_args = vae_args
        model.img_gen_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_GEN_CONTEXT_TOKEN)
        model.img_gen_start_token_id = self.tokenizer.convert_tokens_to_ids(IMG_GEN_START_TOKEN)
        self.model = model.to(device=device, dtype=torch.bfloat16).eval()
        self.vae = self.model.vae_local

        self._dec_t = [0.0]
        orig = self.vae.decode

        def timed_decode(*a, **k):
            torch.cuda.synchronize(); t = time.time()
            out = orig(*a, **k)
            torch.cuda.synchronize(); self._dec_t[0] += time.time() - t
            return out
        self.vae.decode = timed_decode

    def param_groups(self):
        # vae_local is attached as a submodule of the model; count it as the decoder only
        core = [m for n, m in self.model.named_children() if n != "vae_local"]
        return {"text_encoder": None, "backbone": core, "decoder": self.vae}

    def _ids(self, text):
        enc = self.tokenizer(text, return_tensors="pt", truncation=False, padding=False)
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)

    @torch.no_grad()
    def generate(self, prompt, seed):
        stages = {}
        torch.cuda.synchronize(); t = time.time()
        sp = self.system_prompt
        ids, mask = self._ids(sp + f"<|im_start|>user\n{prompt}<|im_end|>" + "<|im_start|>assistant\n<img_gen>")
        ids_cfg, mask_cfg = self._ids(sp + "<|im_start|>user\n<|im_end|><|im_start|>assistant\n<img_gen>")
        torch.cuda.synchronize(); stages["text_encode"] = time.time() - t

        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
        self._dec_t[0] = 0.0
        torch.cuda.synchronize(); t = time.time()
        img = self.model.generate_t2i(
            input_ids=ids, input_ids_cfg=ids_cfg, attention_mask=mask, attention_mask_cfg=mask_cfg,
            output_hidden_states=True, cfg=CFG, top_k=TOP_K, top_p=TOP_P, use_cache=True,
            return_dict=True, h_div_w=1.0)
        img = Image.fromarray(img[0].add_(1).mul_(0.5).to(dtype=torch.float32).permute(1, 2, 0)
                              .mul_(255).clamp_(0, 255).cpu().numpy().astype(np.uint8))
        torch.cuda.synchronize(); total = time.time() - t
        stages["decode"] = self._dec_t[0]
        stages["backbone"] = total - self._dec_t[0]
        return img, stages
