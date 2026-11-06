"""How far each model has got: finished records per output file, and the time left.

  python scripts/progress.py                     (Spark outputs)
  python scripts/progress.py --out-root outputs --models-config configs/models_mac.toml
"""

import argparse
import json
import sys
import time
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faultline.runner import select  # noqa: E402
from faultline.schema import read_jsonl  # noqa: E402

# measured on the DGX Spark, seconds per record (bfloat16)
SPARK_SPEED = {"qwen3vl-2b": 1.3, "qwen25vl-3b": 3.0, "smolvlm2-2b": 3.9, "qwen3vl-4b": 4.3,
               "internvl3-2b": 4.5, "qwen3vl-30b-a3b": 7.2, "qwen3vl-8b": 7.3, "qwen25vl-7b": 7.5,
               "internvl3-8b": 12.5, "qwen3vl-32b": 25.9, "qwen25vl-32b": 26.7}


def count(path):
    ok, errors, bad = set(), 0, 0
    if not path.exists():
        return ok, errors, bad
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if r.get("error") is None:
                ok.add(r["id"])
            else:
                errors += 1
    return ok, errors, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="outputs/spark")
    ap.add_argument("--models-config", default="configs/models_spark.toml")
    ap.add_argument("--dataset", default="data/faultline_v0.jsonl")
    args = ap.parse_args()

    with open(args.models_config, "rb") as f:
        models = list(tomllib.load(f)["models"])
    core = {r["id"] for r in select(read_jsonl(args.dataset), "core")}
    first = {r["id"] for r in select(read_jsonl(args.dataset), "core", first=1000)}
    left_h = 0.0
    print(f"{'model':<17}{'first 1000':>11}{'all 2001':>10}{'errors':>8}{'bad lines':>10}  last write")
    for m in models:
        p = Path(args.out_root) / m / "core.jsonl"
        ok, errors, bad = count(p)
        when = time.strftime("%m-%d %H:%M", time.localtime(p.stat().st_mtime)) if p.exists() else "-"
        print(f"{m:<17}{len(ok & first):>11}{len(ok & core):>10}{errors:>8}{bad:>10}  {when}")
        left_h += (len(core) - len(ok & core)) * SPARK_SPEED.get(m, 10) / 3600
    print(f"\nabout {left_h:.1f} h of running left (Spark speeds; plus model loading)")


if __name__ == "__main__":
    main()
