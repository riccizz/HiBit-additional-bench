"""BAGEL-7B-MoT (ByteDance-Seed) — unified MoT model, T2I by rectified flow, 1024px.

Qwen2.5-based Mixture-of-Transformers (14B total / 7B active: an understanding expert
and a generation expert sharing attention) prefills the prompt into a KV cache, then
the generation expert denoises 64x64 FLUX-VAE latents (2x2 patchified -> 1024 tokens)
with rectified flow: 50 steps, timestep_shift 3.0, text-CFG 4.0 over the full [0, 1]
interval, img-CFG 1.0, renorm_min 0 — the settings of the repo's GenEval generator
(eval/gen/gen_images_mp.py via scripts/eval/run_geneval.sh, 1024px; the
inference.ipynb demo cell uses cfg_interval [0.4, 1.0] instead). No "think" / CoT
rewriting. Uses the repo's InterleaveInferencer (cloned into this folder).

Family: the model's own report generates images with rectified flow (not next-token
prediction), so it is listed as continuous-diffusion; it is a unified AR-text +
diffusion-image model.

Stage split: decode = FLUX-VAE decode (wrapper on inferencer.decode_image);
backbone = the flow loop (wrapper on model.generate_image); text_encode = the rest
(chat-template + prompt prefill into the KV cache for the cond / text-uncond /
img-uncond contexts). Param groups exclude the SigLIP understanding tower (0.4B),
which never runs in T2I."""
from __future__ import annotations
import os
import random
import sys
import time

import numpy as np
import torch

from harness import ModelAdapter, Spec

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(HERE, "Bagel")
REPO_ID = "ByteDance-Seed/BAGEL-7B-MoT"
RES = 1024
HYPER = dict(cfg_text_scale=4.0, cfg_img_scale=1.0, cfg_interval=[0.0, 1.0], timestep_shift=3.0,
             num_timesteps=50, cfg_renorm_min=0.0, cfg_renorm_type="global")


class Adapter(ModelAdapter):
    spec = Spec(
        model="bagel",
        display_name="BAGEL-7B-MoT (14B)",
        family="continuous-diffusion",
        resolution=RES,
        steps_desc="prompt prefill + 50 rectified-flow steps (text-CFG 4.0, shift 3.0)",
        nfe=50,
        tokens=(RES // 16) ** 2,   # 64x64 latent, 2x2 patchified -> 1024 image tokens
        notes="unified Qwen2.5 MoT (14B total / 7B active); images by rectified flow inside the "
              "LLM over FLUX-VAE latents; text via backbone (no sep. encoder); no CoT 'think'; "
              "params exclude the unused SigLIP understanding tower",
    )

    def download(self):
        from huggingface_hub import snapshot_download
        snapshot_download(REPO_ID)

    def load(self, device="cuda"):
        self.device = device
        sys.path.insert(0, VENDOR)
        from huggingface_hub import snapshot_download
        from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
        from data.transforms import ImageTransform
        from data.data_utils import add_special_tokens
        from modeling.bagel import (BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
                                    SiglipVisionConfig, SiglipVisionModel)
        from modeling.qwen2 import Qwen2Tokenizer
        from modeling.autoencoder import load_ae
        from inferencer import InterleaveInferencer

        path = snapshot_download(REPO_ID)
        llm_config = Qwen2Config.from_json_file(os.path.join(path, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"
        vit_config = SiglipVisionConfig.from_json_file(os.path.join(path, "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1
        vae_model, vae_config = load_ae(local_path=os.path.join(path, "ae.safetensors"))
        config = BagelConfig(visual_gen=True, visual_und=True, llm_config=llm_config,
                             vit_config=vit_config, vae_config=vae_config,
                             vit_max_num_patch_per_side=70, connector_act="gelu_pytorch_tanh",
                             latent_patch_size=2, max_latent_size=64)
        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            model = Bagel(language_model, vit_model, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)
        tokenizer = Qwen2Tokenizer.from_pretrained(path)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

        # single-GPU placement, as in inference.ipynb's device_count()==1 branch
        # size the map in bf16 (the dtype it is loaded in); sizing the fp32 meta model (~57GB)
        # spills ~1/3 of the layers to disk offload on a 46-48GB GPU (~7 min/image)
        device_map = infer_auto_device_map(model, max_memory={0: "40GiB"}, dtype=torch.bfloat16,
                                           no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"])
        for k in ["language_model.model.embed_tokens", "time_embedder", "latent_pos_embed",
                  "vae2llm", "llm2vae", "connector", "vit_pos_embed"]:
            device_map[k] = device_map.get("language_model.model.embed_tokens", 0) if k in device_map else 0
        placed = {}
        for v in device_map.values():
            placed[str(v)] = placed.get(str(v), 0) + 1
        print(f"[bagel] device_map placement (#modules per device): {placed}", flush=True)
        self.model = load_checkpoint_and_dispatch(
            model, checkpoint=os.path.join(path, "ema.safetensors"), device_map=device_map,
            offload_buffers=True, dtype=torch.bfloat16, force_hooks=True,
            offload_folder=os.path.join(HERE, ".offload")).eval()
        # The VAE is fp32, but interleave_inference wraps gen_image (incl. decode_image) in a
        # bf16 autocast region, where conv_in fails with "Input type (float) and bias type
        # (BFloat16)" (seen on torch 2.5.1 and 2.8.0). Decode in plain fp32, outside autocast.
        self.vae_model = vae_model.to(device=device, dtype=torch.float32).eval()
        _vae_decode = self.vae_model.decode

        def fp32_decode(z):
            with torch.autocast(device_type="cuda", enabled=False):
                return _vae_decode(z.to(device=device, dtype=torch.float32))
        self.vae_model.decode = fp32_decode
        self.inferencer = InterleaveInferencer(
            model=self.model, vae_model=self.vae_model, tokenizer=tokenizer,
            vae_transform=ImageTransform(1024, 512, 16), vit_transform=ImageTransform(980, 224, 14),
            new_token_ids=new_token_ids)

        # stage timers: wrap the flow loop and the VAE decode
        self._t = {"bb": 0.0, "dec": 0.0}

        def timed(fn, key):
            def w(*a, **k):
                torch.cuda.synchronize(); t = time.time()
                out = fn(*a, **k)
                torch.cuda.synchronize(); self._t[key] += time.time() - t
                return out
            return w
        self.model.generate_image = timed(self.model.generate_image, "bb")
        self.inferencer.decode_image = timed(self.inferencer.decode_image, "dec")

    def param_groups(self):
        m = self.model
        core = [m.language_model] + [getattr(m, n) for n in
                                     ("time_embedder", "latent_pos_embed", "vae2llm", "llm2vae")
                                     if hasattr(m, n)]
        return {"text_encoder": None, "backbone": core, "decoder": self.vae_model}

    @torch.no_grad()
    def generate(self, prompt, seed):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        self._t["bb"] = self._t["dec"] = 0.0
        torch.cuda.synchronize(); t = time.time()
        out = self.inferencer(text=prompt, image_shapes=(RES, RES), **HYPER)
        torch.cuda.synchronize(); total = time.time() - t
        stages = {"backbone": self._t["bb"], "decode": self._t["dec"],
                  "text_encode": total - self._t["bb"] - self._t["dec"]}
        return out["image"], stages
