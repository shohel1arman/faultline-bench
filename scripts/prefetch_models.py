"""Download model weights into the Hugging Face cache before a long run.

Downloading the large models while the small ones are evaluated keeps the GPU busy.
Already downloaded files are skipped, so the script can be stopped and rerun.

  python scripts/prefetch_models.py --models-config configs/models_spark.toml
  python scripts/prefetch_models.py --models-config configs/models_spark.toml --models qwen3vl-32b
"""

import argparse
import sys
import time
import tomllib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-config", default="configs/models_spark.toml")
    ap.add_argument("--models", help="comma-separated model keys; default: all in the config")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    with open(args.models_config, "rb") as f:
        cfg = tomllib.load(f)["models"]
    keys = [k.strip() for k in args.models.split(",")] if args.models else list(cfg)
    unknown = [k for k in keys if k not in cfg]
    if unknown:
        sys.exit(f"unknown models {unknown}; known: {sorted(cfg)}")

    for k in keys:
        t0 = time.time()
        print(f"{k}: {cfg[k]['hf_id']}", flush=True)
        path = snapshot_download(cfg[k]["hf_id"], allow_patterns=["*.json", "*.safetensors", "*.txt",
                                                                  "*.model", "*.jinja"])
        print(f"  done in {(time.time() - t0) / 60:.0f} min: {path}", flush=True)


if __name__ == "__main__":
    main()
