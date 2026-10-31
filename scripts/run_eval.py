"""Run all probes for one model.

Examples:
  python scripts/run_eval.py --model qwen25vl-3b --device cuda:0 --limit 5
  python scripts/run_eval.py --model internvl3-2b --device cuda:1 --subset core

Outputs go to outputs/<model>/<subset>.jsonl and are resumed automatically.
"""

import argparse
import json
import os
import platform
import sys
import time
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# model outputs can contain any unicode; the Windows console default codec cannot print all of it
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from faultline.models import HFBackend, load_image, mps_version_problem  # noqa: E402
from faultline.prompts import PROMPT_VERSION  # noqa: E402
from faultline.runner import run, select  # noqa: E402
from faultline.schema import read_jsonl  # noqa: E402


def device_name(torch, device):
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_name(device)
    if device == "mps":
        return "Apple GPU (mps), " + platform.machine()
    return device


def parse_shard(s):
    if not s:
        return None
    k, n = s.split("/")
    return int(k), int(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--models-config", default="configs/models.toml")
    ap.add_argument("--dataset", default="data/faultline_v0.jsonl")
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--subset", choices=["core", "main"], default="core")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--shard", help="k/n: run every n-th record starting at k")
    ap.add_argument("--first", type=int,
                    help="only the first N records in stratified order (same file as the full run)")
    ap.add_argument("--only-probes",
                    help="comma-separated probe names; writes outputs/<model>/extra.jsonl by default")
    ap.add_argument("--out")
    args = ap.parse_args()

    with open(args.models_config, "rb") as f:
        mcfg = tomllib.load(f)
    if args.model not in mcfg["models"]:
        sys.exit(f"unknown model {args.model}; known: {sorted(mcfg['models'])}")
    cfg = {**mcfg["defaults"], **mcfg["models"][args.model]}
    if args.device == "mps" and cfg["quant"] != "none":
        sys.exit(f"{args.models_config} uses quant={cfg['quant']}, which needs CUDA; "
                 "use --models-config configs/models_mac.toml on a Mac")

    records = select(read_jsonl(args.dataset), args.subset, parse_shard(args.shard), args.limit,
                     first=args.first)
    tag = args.subset if not args.limit else f"{args.subset}_limit{args.limit}"
    if args.shard:
        tag += "_shard" + args.shard.replace("/", "of")
    only = [p.strip() for p in args.only_probes.split(",")] if args.only_probes else None
    if only and not args.out:
        tag = "extra" if not args.limit else f"extra_limit{args.limit}"
    out = Path(args.out or f"outputs/{args.model}/{tag}.jsonl")

    if args.device == "mps":
        # let the few operators without an Apple GPU kernel run on the CPU instead of failing;
        # must be set before torch is imported
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch
    import transformers

    if args.device == "mps":
        problem = mps_version_problem(torch.__version__, platform.mac_ver()[0])
        if problem:
            sys.exit(problem)

    meta = {"model": args.model, **cfg, "prompt_version": PROMPT_VERSION,
            "transformers": transformers.__version__, "torch": torch.__version__,
            "gpu": device_name(torch, args.device),
            "platform": platform.platform(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": args.dataset, "subset": args.subset, "first": args.first, "only_probes": only,
            "n_records": len(records)}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))

    t0 = time.perf_counter()
    backend = HFBackend(cfg["hf_id"], device=args.device, quant=cfg["quant"], dtype=cfg["dtype"])
    mem = backend.memory_gib()
    mem_txt = f"; gpu memory {mem:.2f} GiB" if mem is not None else ""
    print(f"loaded in {time.perf_counter() - t0:.0f}s{mem_txt}")

    run(backend, records, args.images, out, args.model, cfg["max_side"], load_image, only=only)


if __name__ == "__main__":
    main()
