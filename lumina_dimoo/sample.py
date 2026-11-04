"""Lumina-DiMOO (Alpha-VLLM, 8B) — discrete masked diffusion (LLaDA dLLM), 1024px.

One 8B LLaDA backbone consumes the prompt and fills a fully-masked grid of 64x64 VQ
tokens (+1 newline token per row) by MaskGIT-style parallel decoding: 64 steps,
cosine schedule, CFG 4.0 (cond + uncond forward per step), temperature 1.0 — the
defaults of the repo's inference/inference_t2i.py and the 1024x1024 DDP eval command
in its README. No ML-Cache (`--use-cache` off, the repo default). VQ decode via the
repo's diffusers VQModel (`vqvae/` subfolder of the same HF repo).

Stage split: text_encode = prompt templating + tokenize; backbone = generate_image
(the whole masked-diffusion loop); decode = VQ-VAE decode + postprocess. No separate
text encoder — the dLLM consumes the prompt itself."""
from __future__ import annotations
import os
import random
import sys
import time

import torch

from harness import ModelAdapter, Spec

VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Lumina-DiMOO")
REPO_ID = "Alpha-VLLM/Lumina-DiMOO"
RES = 1024
STEPS = 64
CFG = 4.0
TEMPERATURE = 1.0


class Adapter(ModelAdapter):
    spec = Spec(
        model="lumina_dimoo",
        display_name="Lumina-DiMOO (8B)",
        family="masked-diffusion",
        resolution=RES,
        steps_desc=f"{STEPS} masked-diffusion steps (CFG {CFG} -> 2 fwd/step), cosine schedule",
        nfe=STEPS,
        tokens=(RES // 16) ** 2 + RES // 16,   # 64x64 VQ tokens + 1 newline per row
        notes="unified discrete-diffusion LLM (LLaDA-8B) over an 8192-entry VQ codebook (f=16); "
              "MaskGIT-style parallel decoding; text via backbone (no sep. encoder); "
              "ML-Cache off (repo default)",
    )

    def download(self):
        from huggingface_hub import snapshot_download
        snapshot_download(REPO_ID, ignore_patterns=["assets/*"])

    def load(self, device="cuda"):
        self.device = device
        sys.path.insert(0, VENDOR)
        from transformers import AutoTokenizer
        from diffusers import VQModel
        from huggingface_hub import snapshot_download
        from config import SPECIAL_TOKENS
        from model import LLaDAForMultiModalGeneration
        from utils.image_utils import decode_vq_to_image, calculate_vq_params, add_break_line
        from utils.prompt_utils import generate_text_to_image_prompt, create_prompt_templates
        from generators.image_generation_generator import generate_image

        self._generate_image = generate_image
        self._decode = decode_vq_to_image
        self._prompt = generate_text_to_image_prompt
        self.templates = create_prompt_templates()
        self.tok_ids = SPECIAL_TOKENS
        self.ckpt = snapshot_download(REPO_ID, ignore_patterns=["assets/*"])
        self.tokenizer = AutoTokenizer.from_pretrained(self.ckpt, trust_remote_code=True)
        self.model = LLaDAForMultiModalGeneration.from_pretrained(
            self.ckpt, torch_dtype=torch.bfloat16).to(device).eval()
        self.vqvae = VQModel.from_pretrained(self.ckpt, subfolder="vqvae").to(device).eval()
        self.seq_len, self.newline_every, gh, gw = calculate_vq_params(RES, RES)
        T = SPECIAL_TOKENS
        img_mask = add_break_line([T["mask_token"]] * self.seq_len, gh, gw, new_number=T["newline_token"])
        self.img_pred = [T["answer_start"], T["boi"]] + img_mask + [T["eoi"], T["answer_end"]]

    def param_groups(self):
        return {"text_encoder": None, "backbone": self.model, "decoder": self.vqvae}

    @torch.no_grad()
    def generate(self, prompt, seed):
        stages = {}
        torch.cuda.synchronize(); t = time.time()
        cond, uncond = self._prompt(prompt, self.templates)
        con_tok = self.tokenizer(cond)["input_ids"]
        unc_tok = self.tokenizer(uncond)["input_ids"]
        prompt_ids = torch.tensor(con_tok + self.img_pred, device=self.device).unsqueeze(0)
        uncon_ids = torch.tensor(unc_tok, device=self.device).unsqueeze(0)
        torch.cuda.synchronize(); stages["text_encode"] = time.time() - t

        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); random.seed(seed)
        torch.cuda.synchronize(); t = time.time()
        vq = self._generate_image(
            self.model, prompt_ids, seq_len=self.seq_len, newline_every=self.newline_every,
            timesteps=STEPS, temperature=TEMPERATURE, cfg_scale=CFG, uncon_ids=uncon_ids,
            code_start=len(con_tok) + 2, use_cache=False)
        torch.cuda.synchronize(); stages["backbone"] = time.time() - t

        t = time.time()
        img = self._decode(vq, None, vae_ckpt=self.ckpt, image_height=RES, image_width=RES,
                           vqvae=self.vqvae)
        torch.cuda.synchronize(); stages["decode"] = time.time() - t
        return img, stages
