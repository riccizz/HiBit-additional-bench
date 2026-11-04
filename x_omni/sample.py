"""X-Omni-En — discrete autoregressive T2I with an RL-tuned unified LM (arXiv 2507.22058).

A Qwen2.5-7B-based LM (+ 4 multimodal adapter / 4 head layers) emits SigLIP-VQ image
tokens next-token (1152px -> 72x72 = 5184 tokens, 16384-entry codebook); the SigLIP-g
tokenizer de-quantizes them to features that condition a SigLIP-FLUX diffusion decoder
(FLUX.1-dev architecture + siglip_embed, 28 steps, true-CFG 1.5) -> 1152px image.
No separate text encoder for the AR stage (the LM reads the prompt); the decoder also
re-encodes the prompt with FLUX.1-dev's T5-XXL + CLIP-L.

Sampling = the repo README's generate.py command: --image-size 1152 1152 --cfg-scale 1.0
--min-p 0.03 (temperature 1.0, top-p 1.0); decoder steps/CFG come from the checkpoint's
config.json (vision_config.decoder).

Placement: the repo's init_vision moves the whole decoder pipe (~36GB incl. T5-XXL) onto
the GPU next to the ~15GB LM. That needs ~55GB; on smaller cards the decoder pipe is
CPU-offloaded instead (XOMNI_OFFLOAD=auto|none|model|sequential), which inflates latency —
timings are only comparable with XOMNI_OFFLOAD=none.

Stage split: text_encode = T5/CLIP encode inside the decoder pipe (the LM's prompt prefill
is inside generate and counted as backbone); decode = FLUX VAE decode; backbone = rest
(5184 AR steps + 28 decoder DiT steps).
"""
from __future__ import annotations
import os
import sys
import time
from types import SimpleNamespace

import torch

from harness import ModelAdapter, Spec

REPO_ID = "X-Omni/X-Omni-En"
# pin the trust_remote_code revision (= what was smoke-tested)
REVISION = os.environ.get("XOMNI_REVISION", "db70091623513f616c6da7a049773ec5934be88e")
FLUX_ID = "black-forest-labs/FLUX.1-dev"
FLUX_ALLOW = ["model_index.json", "scheduler/*", "text_encoder/*", "text_encoder_2/*",
              "tokenizer/*", "tokenizer_2/*", "vae/*"]
RES = 1152
DOWNSAMPLE = 16
CFG = 1.0
MIN_P = 0.03
TEMPERATURE = 1.0
TOP_P = 1.0


class Adapter(ModelAdapter):
    spec = Spec(
        model="x_omni",
        display_name="X-Omni-En",
        family="autoregressive",
        resolution=RES,
        steps_desc="5184 visual AR tokens (CFG 1.0, min-p 0.03) + 28 SigLIP-FLUX decoder steps (true-CFG 1.5)",
        nfe=(RES // DOWNSAMPLE) ** 2 + 28,
        tokens=(RES // DOWNSAMPLE) ** 2,
        notes="RL-tuned discrete AR: Qwen2.5-7B LM -> SigLIP-VQ tokens -> SigLIP-FLUX(12B) "
              "diffusion decoder with FLUX.1-dev T5/CLIP/VAE (gated); text via LM for the AR stage. "
              "Measured total 27.76B: LM (incl. lm/mm embeds+heads) 9.60B, SigLIP-FLUX decoder ~12B, "
              "FLUX T5/CLIP 4.89B, VAE/VQ 1.37B",
    )

    def download(self):
        from huggingface_hub import snapshot_download
        snapshot_download(REPO_ID, revision=REVISION)
        if not os.environ.get("FLUX_PATH"):
            snapshot_download(FLUX_ID, token=os.environ.get("HF_TOKEN") or None,
                              allow_patterns=FLUX_ALLOW)

    # ------------------------------------------------------------------ #
    def _load_lm(self, device):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(REPO_ID, revision=REVISION, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            REPO_ID, revision=REVISION, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
        self.model.set_generation_mode("image")
        self.model.eval()

    def _init_vision(self):
        """Re-implementation of XOmniForCausalLM.init_vision (remote code) with a choice of
        decoder placement; identical components/config, only .to(device) is replaced."""
        from huggingface_hub import hf_hub_download, snapshot_download
        m = self.model
        rc = sys.modules[type(m).__module__]   # the trust_remote_code modeling_xomni module
        flux_path = os.environ.get("FLUX_PATH") or snapshot_download(
            FLUX_ID, token=os.environ.get("HF_TOKEN") or None, allow_patterns=FLUX_ALLOW)

        m.som_token, m.eom_token, m.img_token = m.config.mm_special_tokens[:3]
        m.vision_config = SimpleNamespace(**m.config.vision_config)
        m.transform_config = SimpleNamespace(**m.vision_config.transform)
        m.encoder_config = SimpleNamespace(**m.vision_config.encoder)
        m.decoder_config = SimpleNamespace(**m.vision_config.decoder)
        m.vision_dtype = {"float32": torch.float32, "float16": torch.float16,
                          "bfloat16": torch.bfloat16}[m.vision_config.dtype]
        m.image_transform = rc.create_anyres_preprocess(**m.vision_config.transform)
        m.encoder_config.siglip_path = hf_hub_download(REPO_ID, m.encoder_config.siglip_path, revision=REVISION)
        m.encoder_config.projector_path = hf_hub_download(REPO_ID, m.encoder_config.projector_path, revision=REVISION)
        m.image_tokenizer = rc.SiglipTokenizer(**vars(m.encoder_config))
        m.image_tokenizer.to(m.device, m.vision_dtype)

        pipe = rc.FluxPipelineWithSigLIP.from_pretrained(
            flux_path, transformer=None, torch_dtype=m.vision_dtype)
        pipe.transformer = rc.FluxTransformer2DModelWithSigLIP.from_pretrained(
            REPO_ID, revision=REVISION, siglip_channels=m.encoder_config.z_channels, torch_dtype=m.vision_dtype,
            subfolder=m.decoder_config.model_path)
        pipe.set_progress_bar_config(disable=True)

        mode = os.environ.get("XOMNI_OFFLOAD", "auto")
        if mode == "auto":
            free, _ = torch.cuda.mem_get_info(m.device)
            mode = "none" if free / 2**30 > 50 else "sequential"
        gpu_id = m.device.index or 0
        if mode == "none":
            pipe.to(m.device)
        elif mode == "model":
            pipe.enable_model_cpu_offload(gpu_id=gpu_id)
        else:
            pipe.enable_sequential_cpu_offload(gpu_id=gpu_id)
        self.offload = mode
        print(f"[x_omni] decoder placement: {mode}", flush=True)
        m.decoder_pipe = pipe

        self._acc = {"text_encode": 0.0, "decode": 0.0}
        orig_enc, orig_dec = pipe.encode_prompt, pipe.vae.decode

        def enc(*a, **k):
            torch.cuda.synchronize(); t = time.time()
            out = orig_enc(*a, **k)
            torch.cuda.synchronize(); self._acc["text_encode"] += time.time() - t
            return out

        def dec(*a, **k):
            torch.cuda.synchronize(); t = time.time()
            out = orig_dec(*a, **k)
            torch.cuda.synchronize(); self._acc["decode"] += time.time() - t
            return out
        pipe.encode_prompt = enc
        pipe.vae.decode = dec

    def load(self, device="cuda"):
        self._load_lm(device)
        self._init_vision()

    def param_groups(self):
        m, p = self.model, self.model.decoder_pipe
        return {"text_encoder": [p.text_encoder, p.text_encoder_2],
                "backbone": [m.model, m.lm_head, m.mm_head, p.transformer],
                "decoder": [m.image_tokenizer, p.vae]}

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate_tokens(self, prompt, seed, res=RES):
        """AR stage only (== generate.py up to model.generate)."""
        from transformers.generation import GenerationConfig
        tok, m = self.tokenizer, self.model
        th, tw = res // DOWNSAMPLE, res // DOWNSAMPLE
        image_prefix = f"<SOM>{th} {tw}<IMAGE>"
        gen_cfg = GenerationConfig(
            max_new_tokens=th * tw, do_sample=True, temperature=TEMPERATURE, min_p=MIN_P,
            top_p=TOP_P, guidance_scale=CFG,
            suppress_tokens=tok.convert_tokens_to_ids(m.config.mm_special_tokens))
        enc = tok([prompt + image_prefix], return_tensors="pt", padding="longest",
                  padding_side="left")
        negative_ids = tok.encode(image_prefix, add_special_tokens=False,
                                  return_tensors="pt").to(self.device).expand(1, -1)
        torch.manual_seed(seed)
        out = m.generate(inputs=enc.input_ids.to(self.device),
                         attention_mask=enc.attention_mask.to(self.device),
                         generation_config=gen_cfg, negative_prompt_ids=negative_ids)
        return torch.nn.functional.pad(out, (0, 1), value=tok.convert_tokens_to_ids("<EOM>"))

    @torch.no_grad()
    def generate(self, prompt, seed):
        self._acc["text_encode"] = 0.0
        self._acc["decode"] = 0.0
        torch.cuda.synchronize(); t = time.time()
        tokens = self.generate_tokens(prompt, seed)
        torch.manual_seed(seed)
        _, images = self.model.mmdecode(self.tokenizer, tokens[0], skip_special_tokens=False)
        torch.cuda.synchronize(); total = time.time() - t
        stages = {"text_encode": self._acc["text_encode"], "decode": self._acc["decode"]}
        stages["backbone"] = total - stages["text_encode"] - stages["decode"]
        return images[0], stages
