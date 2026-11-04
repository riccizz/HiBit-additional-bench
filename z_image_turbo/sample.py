"""Z-Image-Turbo — continuous diffusion, step-distilled (Decoupled-DMD + DMDR RL) 6B
S3-DiT + Qwen3-4B text encoder, Tongyi-MAI. Model-card usage: 1024x1024,
num_inference_steps=9 ("actually results in 8 DiT forwards"), guidance_scale 0.0 (no CFG).
Uses diffusers' ZImagePipeline (z_image's .venv is shared via a symlink created by
download.sh)."""
from __future__ import annotations
import torch
from harness import PipelineAdapter, Spec


class Adapter(PipelineAdapter):
    spec = Spec(
        model="z_image_turbo",
        display_name="Z-Image-Turbo",
        family="continuous-diffusion",
        resolution=1024,
        steps_desc="8 DiT forwards (num_inference_steps=9, no CFG), step-distilled",
        nfe=8,
        tokens=4096,  # 1024px / VAE f8 / patch 2 = 64x64
        notes="Z-Image-Turbo 6B S3-DiT (Decoupled-DMD + DMDR) + Qwen3-4B text encoder | "
              "stages from diffusers step callbacks (see harness.StepTimer)",
    )
    repo_id = "Tongyi-MAI/Z-Image-Turbo"
    steps = 9
    call_kwargs = {"guidance_scale": 0.0}

    def _build_pipe(self, device):
        from diffusers import ZImagePipeline
        return ZImagePipeline.from_pretrained(
            self.repo_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False).to(device)
