"""Run several models one after another, in phases, on one GPU.

Phase 1 brings every model to the same first N records (stratified order); phase 2
extends every model to the whole subset. Each model runs in its own process, so its
memory is released before the next one loads. Finished records are skipped, so the
queue can be stopped and restarted at any time.

Usage (Mac):
  caffeinate -i uv run python scripts/run_queue.py --device mps --models-config configs/models_mac.toml \\
      --models internvl3-2b,smolvlm2-2b,qwen25vl-3b,qwen3vl-4b --phases 1000,all
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="comma-separated model keys, in run order")
    ap.add_argument("--phases", default="1000,all", help="comma-separated record counts; 'all' = whole subset")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--models-config", default="configs/models_mac.toml")
    ap.add_argument("--subset", default="core")
    ap.add_argument("--dataset", default="data/faultline_v0.jsonl")
    ap.add_argument("--images", default="data/images")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    results = []
    for phase in phases:
        for model in models:
            cmd = [sys.executable, str(ROOT / "scripts" / "run_eval.py"),
                   "--model", model, "--device", args.device,
                   "--models-config", args.models_config, "--subset", args.subset,
                   "--dataset", args.dataset, "--images", args.images]
            if phase != "all":
                cmd += ["--first", phase]
            print(f"\n=== phase {phase}: {model}  ({time.strftime('%Y-%m-%d %H:%M')})", flush=True)
            t0 = time.time()
            code = subprocess.run(cmd, cwd=ROOT).returncode
            hours = (time.time() - t0) / 3600
            results.append((phase, model, code, hours))
            print(f"=== phase {phase}: {model} exited with {code} after {hours:.1f} h", flush=True)

    print("\nsummary")
    for phase, model, code, hours in results:
        status = "ok" if code == 0 else f"FAILED ({code})"
        print(f"  phase {phase:<5} {model:<16} {status:<12} {hours:5.1f} h")


if __name__ == "__main__":
    main()
