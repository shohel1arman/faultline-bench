"""Build the FaultLine JSONL from GQA validation scene graphs.

Usage:
  python scripts/build_dataset.py --config configs/dataset.toml
  python scripts/build_dataset.py --config configs/dataset.toml --limit-images 500 --out data/smoke.jsonl
"""

import argparse
import json
import sys
import tomllib
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faultline.generate import SEVERITIES, TYPES, build  # noqa: E402
from faultline.scene import load_image_meta, load_scene_graphs  # noqa: E402
from faultline.schema import validate, write_jsonl  # noqa: E402


def report(records, supply, cfg):
    q = cfg["quota"]
    need = q["per_cell"] + q["reserve_per_cell"]
    print(f"\n{'cell':<16}{'supply':>8}{'main':>7}{'reserve':>9}{'core':>6}")
    short = []
    for t in TYPES:
        for s in SEVERITIES:
            rows = [r for r in records if r["conflict_type"] == t and r["severity"] == s]
            main = sum(r["split"] == "main" for r in rows)
            res = len(rows) - main
            core = sum(r["core"] for r in rows)
            print(f"{t + ' ' + s:<16}{supply.get((t, s), 0):>8}{main:>7}{res:>9}{core:>6}")
            if len(rows) < need:
                short.append((t, s, len(rows), need))
    ctrl = [r for r in records if not r["is_conflict"]]
    by = Counter((r["question_type"], r["split"]) for r in ctrl)
    for t in TYPES:
        print(f"{'control ' + t:<16}{'':>8}{by[(t, 'main')]:>7}{by[(t, 'reserve')]:>9}"
              f"{sum(r['core'] for r in ctrl if r['question_type'] == t):>6}")
    imgs = len({r["image_id"] for r in records})
    print(f"\nrecords {len(records)}  images {imgs}  "
          f"core {sum(r['core'] for r in records)}  "
          f"with coco_id {sum(r['coco_id'] is not None for r in records)}")
    return short


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dataset.toml")
    ap.add_argument("--out")
    ap.add_argument("--limit-images", type=int)
    args = ap.parse_args()

    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    out = Path(args.out or cfg["output"])

    graphs = load_scene_graphs(cfg["scene_graphs"])
    meta = load_image_meta(cfg["image_data"]) if Path(cfg["image_data"]).exists() else {}
    print(f"scene graphs: {len(graphs)} images; image metadata: {len(meta)} rows")

    records, supply = build(graphs, cfg, image_meta=meta, limit_images=args.limit_images)

    bad = [(r["id"], e) for r in records for e in validate(r)]
    if bad:
        for b in bad[:20]:
            print("INVALID", b)
        sys.exit(f"{len(bad)} validation errors")

    short = report(records, supply, cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, records)
    stats = {"records": len(records), "supply": {f"{t}_{s}": n for (t, s), n in supply.items()},
             "shortfalls": short, "config": cfg}
    with open(out.with_suffix(".stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"wrote {out}")
    if short:
        print(f"WARNING: cells below quota: {short}")


if __name__ == "__main__":
    main()
