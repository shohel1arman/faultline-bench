"""Score raw outputs and print a summary per model.

Usage:
  python scripts/score_eval.py outputs/qwen25vl-3b/core_limit5.jsonl
  python scripts/score_eval.py outputs/*/core.jsonl --by-cell --show 3
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# model outputs can contain any unicode; the Windows console default codec cannot print all of it
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from faultline.schema import read_jsonl  # noqa: E402
from faultline.score import by_cell, outcome, summarize  # noqa: E402


def load_outputs(path):
    """Last successful row per id (a resumed run may contain retries)."""
    rows = {}
    for r in read_jsonl(path):
        if r.get("error") is None and r.get("probes"):
            rows[r["id"]] = r
    return rows


def fmt(v):
    if v is None:
        return "  -  "
    return f"{v:5.3f}" if isinstance(v, float) else str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outputs", nargs="+")
    ap.add_argument("--dataset", default="data/faultline_v0.jsonl")
    ap.add_argument("--by-cell", action="store_true")
    ap.add_argument("--show", type=int, default=0, help="print raw outputs for N records")
    ap.add_argument("--save", help="write per-record outcomes to this JSONL")
    args = ap.parse_args()

    recs = {r["id"]: r for r in read_jsonl(args.dataset)}
    saved = []
    for path in args.outputs:
        rows = load_outputs(path)
        outs = [outcome(recs[i], r["probes"]) for i, r in rows.items()]
        model = next(iter(rows.values()))["model"] if rows else path
        print(f"\n== {model}  ({path}, {len(outs)} records)")
        s = summarize(outs)
        joint = s.pop("joint")
        for k, v in s.items():
            print(f"  {k:<24}{fmt(v)}")
        print("  joint D/R/A: " + "  ".join(f"{k}={v:.3f}" for k, v in joint.items()))

        if args.by_cell:
            print(f"\n  {'cell':<14}{'DetR':>7}{'ResA':>7}{'AttA':>7}{'strict':>8}{'perc':>7}")
            for (t, sev), c in by_cell(outs).items():
                print(f"  {t + ' ' + sev:<14}{fmt(c['DetR']):>7}{fmt(c['ResA_mcq']):>7}"
                      f"{fmt(c['AttA']):>7}{fmt(c['AttA_strict']):>8}{fmt(c['perceive']):>7}")

        for i, (rid, r) in enumerate(rows.items()):
            if i >= args.show:
                break
            rec = recs[rid]
            print(f"\n  --- {rid} {rec['conflict_type']} {rec['severity']}: {rec['text']}")
            print(f"  Q: {rec['question']}  image={rec['answer_image']} text={rec['answer_text']}")
            for name, res in r["probes"].items():
                val = res.get("output", res.get("choice"))
                print(f"    {name:<11} {val!r}  ({res.get('sec')}s)")

        for o in outs:
            o["model"] = model
        saved.extend(outs)

    if args.save:
        with open(args.save, "w", encoding="utf-8", newline="\n") as f:
            for o in saved:
                f.write(json.dumps(o) + "\n")


if __name__ == "__main__":
    main()
