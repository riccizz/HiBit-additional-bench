#!/usr/bin/env python
"""Download or benchmark one or more models.

    python run.py z_image --download          # build env + fetch weights (runs z_image/download.sh)
    python run.py z_image                      # benchmark all prompts
    python run.py z_image onecat --prompts sign_cafe
    python run.py all

If `<model>/.venv` exists, the benchmark is re-launched inside that env, so the Python
that calls this script does not need torch.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = ["lumina_dimoo", "sensenova_u15", "bagel", "z_image", "z_image_turbo",
          "x_omni", "glm_image", "star_7b", "onecat"]


def load_adapter(model):
    path = os.path.join(ROOT, model, "sample.py")
    spec = importlib.util.spec_from_file_location(f"{model}_sample", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Adapter()


def run_in_process(args, model):
    sys.path.insert(0, ROOT)
    from harness import DEFAULT_OUT, benchmark
    keys = args.prompts.split(",") if args.prompts else None
    benchmark(load_adapter(model), device=args.device, seed=args.seed,
              prompt_keys=keys, out_dir=args.out or DEFAULT_OUT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", help=f"model folder(s) or 'all': {', '.join(MODELS)}")
    ap.add_argument("--download", action="store_true", help="run <model>/download.sh instead of benchmarking")
    ap.add_argument("--prompts", default="", help="comma-separated prompt keys (default: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="", help="output dir (default: ./results)")
    ap.add_argument("--in-env", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    models = MODELS if args.models == ["all"] else args.models
    for model in models:
        if not os.path.isfile(os.path.join(ROOT, model, "sample.py")):
            sys.exit(f"unknown model '{model}'")
        if args.download:
            subprocess.run(["bash", os.path.join(ROOT, model, "download.sh")], check=True)
            continue
        env_python = os.path.join(ROOT, model, ".venv", "bin", "python")
        if not args.in_env and os.path.exists(env_python):
            cmd = [env_python, os.path.abspath(__file__), model, "--in-env",
                   "--seed", str(args.seed), "--device", args.device]
            if args.prompts:
                cmd += ["--prompts", args.prompts]
            if args.out:
                cmd += ["--out", args.out]
            subprocess.run(cmd, check=True)
        else:
            run_in_process(args, model)


if __name__ == "__main__":
    main()
