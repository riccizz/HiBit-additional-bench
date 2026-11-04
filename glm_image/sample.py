"""GLM-Image (Z.ai) — hybrid autoregressive + diffusion.

A 9B AR generator (GlmImageForConditionalGeneration, GLM-4-9B init) first samples visual
prior tokens (a small preview grid + the d32 target grid; 1024px -> 32x32), which are 2x
upsampled to condition a 7B single-stream DiT decoder (GlmImageTransformer2DModel, plus a
ByT5 glyph encoder for quoted text). Model-card usage: 50 DiT steps, guidance_scale 1.5;
we run 1024x1024. Uses diffusers' GlmImagePipeline (z_image's .venv, shared by symlink).

Stage split: the AR prior runs inside the pipe before the DiT loop, so the step-callback
text_encode bucket (start .. end-of-step-0) would swallow it. We time
`vision_language_encoder.generate` with a wrapper and move it to `backbone`:
  text_encode = (start .. end-of-step-0) - AR time   (glyph/T5 encode + 1st DiT step)
  backbone    = AR prior + remaining N-1 DiT steps
  decode      = VAE + postprocess
"""
from __future__ import annotations
import time

import torch

from harness import PipelineAdapter, Spec


class Adapter(PipelineAdapter):
    spec = Spec(
        model="glm_image",
        display_name="GLM-Image",
        family="autoregressive",
        resolution=1024,
        steps_desc="1281 AR prior tokens (32x32 target + 16x16 preview + 1) + 50 DiT steps (CFG 1.5)",
        nfe=1331,  # 1281 AR steps (1 fwd/token) + 50 DiT steps; CFG not folded into NFE
        tokens=1024,  # d32 AR target grid at 1024px (32x32); the DiT itself sees 4096 latent tokens
        notes="hybrid: 9B GLM AR prior tokens -> 7B single-stream DiT decoder (+ByT5 glyph "
              "encoder); text via AR backbone + glyph encoder",
    )
    repo_id = "zai-org/GLM-Image"
    steps = 50
    call_kwargs = {"guidance_scale": 1.5}

    def _build_pipe(self, device):
        from diffusers.pipelines.glm_image import GlmImagePipeline
        pipe = GlmImagePipeline.from_pretrained(
            self.repo_id, torch_dtype=torch.bfloat16).to(device)
        vle = pipe.vision_language_encoder
        orig = vle.generate
        self._ar = {"t": 0.0}

        def timed_generate(*a, **k):
            torch.cuda.synchronize(); t = time.perf_counter()
            out = orig(*a, **k)
            torch.cuda.synchronize(); self._ar["t"] += time.perf_counter() - t
            return out
        vle.generate = timed_generate
        return pipe

    def param_groups(self):
        p = self.pipe
        return {"text_encoder": p.text_encoder,
                "backbone": [p.vision_language_encoder, p.transformer],
                "decoder": p.vae}

    def generate(self, prompt, seed):
        # The AR prior runs inside the pipeline before the first denoising step, so the
        # harness's step timer would book it as text encoding; move it to the backbone.
        self._ar["t"] = 0.0
        image, stages = super().generate(prompt, seed)
        stages["text_encode"] -= self._ar["t"]
        stages["backbone"] += self._ar["t"]
        return image, stages
