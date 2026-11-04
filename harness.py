"""Minimal single-GPU text-to-image benchmark harness.

Each model lives in its own folder with a `sample.py` that defines `Adapter`, a subclass of
`ModelAdapter` (or of `PipelineAdapter` for diffusers pipelines). `benchmark()` loads the
adapter, runs one warmup generation, then generates every selected prompt at batch size 1
and writes `<out>/<model>.json` plus `<out>/images/<model>__<key>.png`.

Latency is split into three stages reported by the adapter:
    text_encode  prompt processing (tokenize / text encoder / LLM prefill)
    backbone     the generative loop (denoising steps, AR decoding, ...)
    decode       latent/token -> pixels
"""
from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import asdict, dataclass

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
PROMPTS_FILE = os.path.join(ROOT, "prompts.json")
DEFAULT_OUT = os.path.join(ROOT, "results")


@dataclass
class Spec:
    model: str            # folder name, used for output file names
    display_name: str
    family: str           # "autoregressive" | "continuous-diffusion" | "masked-diffusion"
    resolution: int       # output side length in pixels
    steps_desc: str       # human-readable sampler description
    nfe: int              # sequential network evaluations per image (CFG branches not counted)
    tokens: int | None = None  # image tokens per backbone forward, if meaningful
    notes: str = ""
    warmup: bool = True


class ModelAdapter:
    """Interface every model folder implements."""

    spec: Spec

    def download(self):
        """Fetch weights. No GPU needed."""
        raise NotImplementedError

    def load(self, device: str = "cuda"):
        raise NotImplementedError

    def param_groups(self) -> dict:
        """{"text_encoder": modules|None, "backbone": modules, "decoder": modules|None}.
        Values may be a module, a list of modules, a float (billions) or None."""
        raise NotImplementedError

    def generate(self, prompt: str, seed: int):
        """Return (PIL.Image, {"text_encode": s, "backbone": s, "decode": s})."""
        raise NotImplementedError


class StepTimer:
    """Stage timing for diffusers pipelines via `callback_on_step_end`.

    The first step boundary closes the text_encode stage (it therefore also contains the
    first denoising step, since diffusers gives no earlier hook); the last one closes the
    backbone stage; everything after it is decode."""

    def __init__(self):
        self.marks: list[float] = []

    def callback(self, pipe, step, timestep, callback_kwargs):
        torch.cuda.synchronize()
        self.marks.append(time.perf_counter())
        return callback_kwargs

    def stages(self, start: float, end: float) -> dict:
        first, last = self.marks[0], self.marks[-1]
        return {"text_encode": first - start, "backbone": last - first, "decode": end - last}


class PipelineAdapter(ModelAdapter):
    """Adapter for a diffusers text-to-image pipeline. Subclasses set `repo_id`, `steps`,
    `call_kwargs` and implement `_build_pipe(device)`."""

    repo_id: str = ""
    steps: int = 50
    call_kwargs: dict = {}

    def _build_pipe(self, device):
        raise NotImplementedError

    def download(self):
        from huggingface_hub import snapshot_download
        snapshot_download(self.repo_id)

    def load(self, device="cuda"):
        self.device = device
        self.pipe = self._build_pipe(device)
        self.pipe.set_progress_bar_config(disable=True)

    def param_groups(self):
        encoders = [getattr(self.pipe, n) for n in ("text_encoder", "text_encoder_2", "text_encoder_3")
                    if getattr(self.pipe, n, None) is not None]
        return {"text_encoder": encoders or None,
                "backbone": self.pipe.transformer,
                "decoder": getattr(self.pipe, "vae", None)}

    def generate(self, prompt, seed):
        timer = StepTimer()
        size = self.spec.resolution
        generator = torch.Generator(device=self.device).manual_seed(seed)
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = self.pipe(prompt=prompt, height=size, width=size, num_inference_steps=self.steps,
                           generator=generator, callback_on_step_end=timer.callback,
                           **self.call_kwargs)
        torch.cuda.synchronize()
        return result.images[0], timer.stages(start, time.perf_counter())


def billions(obj) -> float:
    if obj is None:
        return 0.0
    if isinstance(obj, (int, float)):
        return float(obj)
    if isinstance(obj, (list, tuple)):
        return sum(billions(o) for o in obj)
    return sum(p.numel() for p in obj.parameters()) / 1e9


def load_prompts(keys=None) -> list[dict]:
    with open(PROMPTS_FILE) as f:
        prompts = json.load(f)
    if keys:
        unknown = set(keys) - {p["key"] for p in prompts}
        if unknown:
            raise KeyError(f"unknown prompt keys: {sorted(unknown)}")
        prompts = [p for p in prompts if p["key"] in keys]
    return prompts


def benchmark(adapter: ModelAdapter, device="cuda", seed=0, prompt_keys=None, out_dir=DEFAULT_OUT):
    spec = adapter.spec
    prompts = load_prompts(prompt_keys)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)

    print(f"[{spec.model}] loading", flush=True)
    adapter.load(device)
    groups = adapter.param_groups()
    params = {k: round(billions(groups.get(k)), 3) for k in ("text_encoder", "backbone", "decoder")}
    print(f"[{spec.model}] params (B) {params} total {sum(params.values()):.3f}", flush=True)

    if spec.warmup:
        adapter.generate(prompts[0]["prompt"], seed)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    per_prompt, images = [], {}
    for p in prompts:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        image, stages = adapter.generate(p["prompt"], seed)
        torch.cuda.synchronize()
        total = time.perf_counter() - t0
        rel = os.path.join("images", f"{spec.model}__{p['key']}.png")
        image.save(os.path.join(out_dir, rel))
        images[p["key"]] = rel
        per_prompt.append({"key": p["key"], "total_s": round(total, 4),
                           **{k: round(v, 4) for k, v in stages.items()}})
        print(f"[{spec.model}] {p['key']:<16} {total:8.2f}s  " +
              "  ".join(f"{k} {v:.2f}" for k, v in stages.items()), flush=True)

    median = statistics.median(r["total_s"] for r in per_prompt)
    report = {
        **asdict(spec),
        "params_b": params,
        "params_total_b": round(sum(params.values()), 3),
        "device": torch.cuda.get_device_name(device),
        "seed": seed,
        "batch": 1,
        "latency_median_s": round(median, 4),
        "stage_median_s": {k: round(statistics.median(r[k] for r in per_prompt), 4)
                           for k in ("text_encode", "backbone", "decode")},
        "peak_vram_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 2),
        "per_prompt": per_prompt,
        "images": images,
    }
    path = os.path.join(out_dir, f"{spec.model}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[{spec.model}] median {median:.2f}s, peak {report['peak_vram_gb']}GB -> {path}", flush=True)
    return report
