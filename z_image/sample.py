"""Z-Image (Base) — continuous diffusion, 6B single-stream DiT (S3-DiT) + Qwen3-4B text
encoder, Tongyi-MAI. Non-distilled base model: model-card usage = 50 steps, CFG 4.0
(recommended 3-5), cfg_normalization off; we run 1024x1024 (card supports 512-2048).
Uses diffusers' ZImagePipeline (diffusers from source; .venv built by download.sh)."""
from __future__ import annotations
import torch
from harness import PipelineAdapter, Spec


class Adapter(PipelineAdapter):
    spec = Spec(
        model="z_image",
        display_name="Z-Image (Base)",
        family="continuous-diffusion",
        resolution=1024,
        steps_desc="50 denoising steps (CFG 4.0 -> 2 fwd/step)",
        nfe=50,
        tokens=4096,  # 1024px / VAE f8 / patch 2 = 64x64
        notes="Z-Image base 6B S3-DiT + Qwen3-4B text encoder | "
              "stages from diffusers step callbacks (see harness.StepTimer)",
    )
    repo_id = "Tongyi-MAI/Z-Image"
    steps = 50
    call_kwargs = {"guidance_scale": 4.0, "cfg_normalization": False}

    def _build_pipe(self, device):
        from diffusers import ZImagePipeline
        return ZImagePipeline.from_pretrained(
            self.repo_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False).to(device)
