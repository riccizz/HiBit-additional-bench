# HiBit additional T2I baselines

Setup and inference code for 9 open text-to-image baselines, built so their quality and
single-GPU latency can be measured the same way. Each model has its own folder:

- `download.sh` builds an isolated env and downloads the weights.
- `sample.py` defines the model's `Adapter`, which loads the model, generates an image and
  reports parameter groups.

The shared code is `harness.py` (adapter interface and benchmark loop), `run.py` (CLI)
and `prompts.json` (8 test prompts).

## Reference numbers

The first two rows are HiBit-1B and HiBit-3B as reported in the current paper draft; later
checkpoints only improve on them. For baselines, a metric is filled only where a reported number was found and
cross-checked. Missing, conflicting or unverified numbers are left blank. PRISM is blank for
every baseline because published PRISM scores use a GPT-4.1 judge, while HiBit's uses a
different judge, so baselines must be re-scored with our pipeline.

Size: HiBit counts trainable params only (excluding its frozen text encoder and image
tokenizer). Baseline sizes are the totals measured by `param_groups()` here, including text
encoders and decoders. Use one counting rule before quoting sizes in the paper.

Row order: HiBit's own family (masked diffusion) first, then continuous diffusion, then AR.
Within each family, larger models come first. Where a baseline beats HiBit on a filled
metric: Z-Image (88.14) and SenseNova-U1.5 (88.11) are above HiBit-1B on DPG, and GLM-Image
(0.9116 CVTG-2K) is above both HiBit models on CVTG-2K.

| model | family | size | GenEval | DPG | CVTG-2K | LongText | PRISM | notes |
|---|---|---|---|---|---|---|---|---|
| **HiBit-1B (ours)** | masked diffusion | 1.00B | | 87.55 | 0.8906 | 0.944 | 78.2 | Current paper Table 1 row (best per metric over NFE settings). Lower bound: newer checkpoints only improve. |
| **HiBit-3B (ours)** | masked diffusion | 3.06B | 0.944 | 88.09 | 0.9008 | 0.969 | 79.8 | From the paper draft (currently commented out of Table 1). |
| Lumina-DiMOO | masked diffusion | 8.2B | 0.88 | 86.04 | 0.590 | | | Closest family to HiBit. 1024², 64 steps, CFG 4.0. ML-Cache (the official speed-up) is **off**; decide whether to enable it for speed comparisons. GenEval 0.88 is the base model; the Self-GRPO variant reports 0.91. CVTG-2K is measured by third parties (InternVL-U, LLaDA-Image); DiMOO does not report it. |
| SenseNova-U1.5 | continuous diffusion | 17.5B (two full experts/layer, ~8.2B per path) | 0.92 | 88.11 | | | | Pixel-space flow matching, no VAE. The default **2048²** is ~4× the pixels of the others; consider 1024² for fair timing. Defaults to the `-Preview` weights; set `SENSENOVA_REPO=sensenova/SenseNova-U1.5-8B-MoT` for the official release. CVTG-2K sources conflict: 0.948 attributed to its own report vs 0.887 for the Preview weights in LLaDA-Image, so re-measure. |
| BAGEL | continuous diffusion | 14.3B (7B active) | 0.82 | 85.07 | 0.356 | 0.373 | | Rectified flow inside the LLM. Settings follow the repo's GenEval script (CFG interval [0,1]); the demo notebook uses [0.4,1]. The reported 0.88 GenEval uses "think"/rewriting; 0.82 is without. Loads with a 40GiB `max_memory` cap. VAE decode is forced to fp32 (upstream dtype bug under autocast). CVTG-2K is measured by third parties (InternVL-U, LLaDA-Image). |
| Z-Image | continuous diffusion | 10.3B (6.2B DiT + 4.0B Qwen3 encoder) | 0.84 | 88.14 | 0.8671 | 0.935 | | 1024², 50 steps, CFG 4.0, per the model card. Owns the venv that `z_image_turbo` and `glm_image` symlink to (diffusers pinned to a git commit). Base sometimes garbles text; Turbo looks cleaner. |
| Z-Image-Turbo | continuous diffusion | 10.3B | 0.82 | 84.86 | 0.8585 | 0.917 | | **Step-distilled** (8 DiT forwards, no CFG), so mark it † in the paper. With only 8 forwards it is the main threat to the speed frontier. Shares `z_image`'s venv. |
| X-Omni | AR | **27.8B** (9.6B LM + ~12B FLUX-sized decoder + 4.9B FLUX text encoders + 1.4B VAE/VQ), i.e. **above 20B** | | | | | | **Needs an HF token** with the FLUX.1-dev license accepted (the decoder uses its text encoders + VAE). 1152², 5184 AR tokens + 28 decoder steps: by far the slowest model here. `XOMNI_OFFLOAD=auto` keeps it on GPU when >50GB is free (B200); offloaded latencies are meaningless. |
| GLM-Image | AR | 17.7B (9B AR + 7B DiT decoder) | | | 0.9116 | | | AR prior (1281 tokens) runs inside the diffusers pipeline. The adapter wraps it so its time counts as backbone, not text encode. Peak VRAM 40.5GB. Reported DPG (84.78 vs 81.01) and LongText (0.952 vs 0.966) conflict across sources, so re-measure. Strongest text rendering in the set. |
| STAR-7B | AR | 17.5B (incl. 2.6B Gemma2 encoder + 1.1B decoders) | | | | | | 576 VQ tokens (384px) + 40 Lumina2 DiT steps (1024²). Decoder settings are hard-coded upstream. Also downloads Qwen2.5-VL-7B and Lumina-Image-2.0. The model card's 0.91 / 87.44 do not say whether they are 3B or 7B. Weak text rendering. |
| OneCAT-3B | AR | 9.6B (3B active) | | 84.53 | | | | Next-scale AR, 13 scales, CFG 20. Eager attention (no flash-attn), transformers 4.37.2. A reported GenEval 0.90 was not verified against the paper. Weak text rendering. |

CVTG-2K is the average word accuracy over 2-5 text regions, the metric reported in the
paper's Table 1.

Sources for baseline numbers: SenseNova-U1.5 report (arXiv 2609.11929); Mage-Flow comparison
tables (arXiv 2607.19064) for Z-Image, Z-Image-Turbo and BAGEL; Lumina-DiMOO (arXiv
2510.06308) and LLaDA-o (arXiv 2603.01068) for DiMOO; SenseNova-U1 (arXiv 2605.12500) for
OneCAT DPG. CVTG-2K: Z-Image paper Table 5 (arXiv 2511.22699) for Z-Image and Z-Image-Turbo;
GLM-Image model card; InternVL-U Table 11 (arXiv 2603.09877) and LLaDA-Image Table 6 (arXiv
2609.03796) for Lumina-DiMOO and BAGEL.

## Models

| folder | HF repo | family | default sampling (from the official inference script / model card) |
|---|---|---|---|
| `z_image` | Tongyi-MAI/Z-Image | continuous diffusion | 1024², 50 steps, CFG 4.0 |
| `z_image_turbo` | Tongyi-MAI/Z-Image-Turbo | continuous diffusion (step-distilled) | 1024², 8 DiT forwards, no CFG |
| `glm_image` | zai-org/GLM-Image | AR prior + DiT decoder | 1024², 1281 AR tokens + 50 DiT steps, CFG 1.5 |
| `lumina_dimoo` | Alpha-VLLM/Lumina-DiMOO | masked (discrete) diffusion | 1024², 64 steps, CFG 4.0 |
| `bagel` | ByteDance-Seed/BAGEL-7B-MoT | rectified flow inside the LLM | 1024², 50 steps, text-CFG 4.0, shift 3.0 (repo's GenEval script) |
| `onecat` | onecat-ai/OneCAT-3B | next-scale AR | 1024², 13 scales, CFG 20 |
| `x_omni` | X-Omni/X-Omni-En | AR + SigLIP-conditioned FLUX decoder | 1152², 5184 AR tokens + 28 decoder steps |
| `star_7b` | MM-MVR/STAR-7B | AR (576 VQ tokens) + Lumina2 DiT decoder | 1024², 40 decoder steps |
| `sensenova_u15` | sensenova/SenseNova-U1.5-8B-MoT-Preview | pixel-space flow matching | 2048², 50 steps, CFG 4.0 |

Each `sample.py` docstring and `Spec.notes` record the exact settings, where they come
from, and any workaround applied to the upstream code.

## Usage

```bash
# 1) build the model's env + download weights (no GPU needed; safe to re-run)
python run.py z_image --download          # same as: bash z_image/download.sh

# 2) benchmark: single GPU, batch 1, one warmup, then every prompt in prompts.json
CUDA_VISIBLE_DEVICES=0 python run.py z_image
#    -> results/z_image.json + results/images/z_image__<prompt>.png
python run.py z_image onecat --prompts sign_cafe,count_birds   # several models / prompt subset
python run.py all
```

`run.py` re-launches itself inside `<model>/.venv`, so the calling `python` needs no torch.
`z_image_turbo` and `glm_image` share `z_image`'s env through a symlink; their
`download.sh` builds that env if it is missing.

Requirements: Linux, an NVIDIA driver that supports CUDA 12.8, and
[`uv`](https://github.com/astral-sh/uv) on `PATH` (or `UV=/path/to/uv`).

### Output

`results/<model>.json` has the model spec, parameter counts (text encoder / backbone /
decoder), GPU name, median latency, median per-stage latency (`text_encode`, `backbone`,
`decode`), peak VRAM, and per-prompt timings. How each adapter splits the stages is
described in its `sample.py` docstring.

## Environments

- Every model has its own venv, built from `requirements.lock.txt`, a `uv pip freeze` of
  the env that was smoke-tested. The upstream repo is cloned into the model folder at a
  pinned commit.
- All envs use **torch 2.8.0 + cu128**, which ships `sm_100` kernels, so they run on
  **B200**. This holds even where upstream pins an older torch, since torch < 2.7 has no
  Blackwell support.
- Where flash-attn is used (BAGEL, X-Omni, STAR, SenseNova), it is the prebuilt 2.8.3
  wheel for torch 2.8, which includes `sm_100`. No source builds are needed.

## Status

Every model was tested end to end from this repo alone: `python run.py <model> --download`
built the env from scratch, then `python run.py <model> --prompts sign_cafe` ran a warmup
plus that prompt. This was done on an L40S 48GB node, and every output image was checked
by eye.

| model | runs | total params (B) | peak VRAM (GB) |
|---|---|---|---|
| z_image | ✅ | 10.26 | 23.3 |
| z_image_turbo | ✅ | 10.26 | 23.3 |
| glm_image | ✅ | 17.67 | 40.5 |
| lumina_dimoo | ✅ | 8.23 | 20.5 |
| bagel | ✅ | 14.25 | 34.0 |
| onecat | ✅ | 9.58 | 34.0 |
| star_7b | ✅ | 17.53 | 40.1 |
| sensenova_u15 | ✅ | 17.53 | 37.3 |
| x_omni | ✅ | 27.76 | 25.1 (sequential CPU offload) |

Notes:
- **Latencies from that node are not reported.** They come from a different GPU, some
  runs used CPU offload, and the GPU was shared. Measure on B200.
- **x_omni:** the decoder needs text encoders + VAE from the gated
  `black-forest-labs/FLUX.1-dev`. Accept the license on HF, then
  `export HF_TOKEN=...` (or `hf auth login`) and rerun `x_omni/download.sh`. A
  fine-grained token needs "read access to public gated repos" enabled.
  Offload: `XOMNI_OFFLOAD=auto` (default) keeps everything on GPU when more than 50GB is
  free (B200) and uses sequential CPU offload otherwise.
- **bagel:** like the official notebook, it loads with a 40GiB per-GPU `max_memory` cap.
  The model fits entirely on one GPU; the placement is printed at load time.
- **sensenova_u15:** defaults to the `-Preview` weights. Set
  `SENSENOVA_REPO=sensenova/SenseNova-U1.5-8B-MoT` for the official release. The checkpoint
  holds 17.5B params: two full experts per layer, one for understanding and one for
  generation, ~8.2B per path.
- **Parameter counts** above are what `param_groups()` measures, including text encoders
  and decoders. Decide one counting rule before quoting sizes in a paper.
