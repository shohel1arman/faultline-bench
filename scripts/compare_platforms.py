"""Cross-platform check: the same models on the Mac (float16) and the Spark (bfloat16).

  python scripts/compare_platforms.py

For each model run on both, on the records both finished: how often the two runs give
the same multiple-choice letter, the same detection decision and the same attribution
fields, and how far the headline metrics move. Writes results/platform_agreement.csv.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faultline import parse  # noqa: E402
from faultline.schema import read_jsonl  # noqa: E402
from faultline.score import outcome, summarize  # noqa: E402

MCQ = ["image_only", "text_only", "mcq", "mcq_focus", "delay", "check_silent", "verbalize"]


def load(path):
    rows = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("error") is None and r.get("probes"):
                rows[r["id"]] = r
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="outputs", help="first output root (Mac)")
    ap.add_argument("--b", default="outputs/spark", help="second output root (Spark)")
    ap.add_argument("--models", default="internvl3-2b,qwen25vl-3b,qwen3vl-4b,smolvlm2-2b")
    ap.add_argument("--dataset", default="data/faultline_v0.jsonl")
    ap.add_argument("--out", default="results/platform_agreement.csv")
    args = ap.parse_args()

    recs = {r["id"]: r for r in read_jsonl(args.dataset)}
    table = []
    for m in args.models.split(","):
        a, b = load(Path(args.a) / m / "core.jsonl"), load(Path(args.b) / m / "core.jsonl")
        ids = sorted(set(a) & set(b) & set(recs))
        row = {"model": m, "n": len(ids)}
        for p in MCQ:
            pairs = [(a[i]["probes"][p].get("choice"), b[i]["probes"][p].get("choice"))
                     for i in ids if p in a[i]["probes"] and p in b[i]["probes"]]
            row[f"same_{p}"] = sum(x == y for x, y in pairs) / len(pairs) if pairs else None
        row["same_detect"] = sum(parse.detection(a[i]["probes"]["detect"]["output"]) ==
                                 parse.detection(b[i]["probes"]["detect"]["output"]) for i in ids) / len(ids)
        for f in ["error", "input", "type", "obvious"]:
            row[f"same_attr_{f}"] = sum(parse.attribution(a[i]["probes"]["attribute"]["output"])[f] ==
                                        parse.attribution(b[i]["probes"]["attribute"]["output"])[f]
                                        for i in ids) / len(ids)
        sa = summarize([outcome(recs[i], a[i]["probes"]) for i in ids])
        sb = summarize([outcome(recs[i], b[i]["probes"]) for i in ids])
        for k in ["perceive", "DetR", "FFR", "DetR_adj", "ResA_mcq", "ResA_verbalize", "AttA"]:
            if sa.get(k) is not None and sb.get(k) is not None:
                row[f"diff_{k}"] = sb[k] - sa[k]
        table.append(row)

    keys = list(table[0])
    for r in table:
        print(f"\n{r['model']} ({r['n']} records)")
        for k in keys[2:]:
            v = r.get(k)
            if v is None:
                continue
            print(f"  {k:<22} {100 * v:+6.1f}" if k.startswith("diff_") else f"  {k:<22} {100 * v:6.1f}%")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        w.writerows(table)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
