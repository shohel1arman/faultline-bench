"""Turn raw model outputs into the paper's numbers.

Usage:
  python scripts/analyze.py                          # every outputs/*/core.jsonl
  python scripts/analyze.py --common                 # only records every model finished
  python scripts/analyze.py --first 1000 --common    # the phase-1 stratified subset

Writes results/: metrics.csv (estimate and 95% interval per model and metric),
cells.csv (per type x severity), joint.csv (detect/resolve/attribute patterns),
tests.csv (paired McNemar tests), outcomes.jsonl (per-record labels), main_table.tex.
"""

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faultline.analysis import bootstrap, common_ids, joint, mcnemar  # noqa: E402
from faultline.runner import select  # noqa: E402
from faultline.schema import read_jsonl  # noqa: E402
from faultline.score import outcome  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TABLE_METRICS = [
    ("perceive", "Perc."),
    ("DetR_adj", "Det."),
    ("ResA_mcq", "Res."),
    ("ResA_if_perceived", "Res.|Perc."),
    ("AttA", "Att."),
    ("AttA_strict", "Att.(strict)"),
    ("TPR_mcq", "TPR"),
]


def load_outputs(path):
    rows, bad = {}, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if r.get("error") is None and r.get("probes"):
                rows[r["id"]] = r
    return rows, bad


def pct(v):
    return "--" if v is None else f"{100 * v:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outputs", nargs="*")
    ap.add_argument("--dataset", default="data/faultline_v0.jsonl")
    ap.add_argument("--out", default="results")
    ap.add_argument("--common", action="store_true", help="restrict to records every model finished")
    ap.add_argument("--first", type=int, help="restrict to the first N core records in stratified order")
    ap.add_argument("--reps", type=int, default=2000)
    args = ap.parse_args()

    paths = args.outputs or sorted(glob.glob("outputs/*/core.jsonl"))
    if not paths:
        sys.exit("no output files found")
    records = read_jsonl(args.dataset)
    recs = {r["id"]: r for r in records}
    image_of = {r["id"]: r["image_id"] for r in records}
    keep = None
    if args.first:
        keep = {r["id"] for r in select(records, subset="core", first=args.first)}

    by_model = {}
    for path in paths:
        rows, bad = load_outputs(path)
        extra = Path(path).with_name("extra.jsonl")
        if extra.exists():                       # probes added later to a finished run
            more, bad_more = load_outputs(extra)
            bad += bad_more
            for i, r in more.items():
                if i in rows:
                    rows[i]["probes"] = {**rows[i]["probes"], **r["probes"]}
            print(f"merged {len(more)} records of extra probes from {extra}")
        model = next(iter(rows.values()))["model"] if rows else Path(path).parent.name
        outs = [outcome(recs[i], r["probes"]) for i, r in rows.items() if keep is None or i in keep]
        for o in outs:
            o["model"] = model
        by_model[model] = outs
        print(f"{model}: {len(outs)} records from {path}" + (f" ({bad} malformed lines skipped)" if bad else ""))

    if args.common:
        ids = common_ids(by_model)
        by_model = {m: [o for o in outs if o["id"] in ids] for m, outs in by_model.items()}
        print(f"common records: {len(ids)}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    with open(out / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["model", "metric", "estimate", "ci_low", "ci_high", "n"])
        for model, outs in by_model.items():
            stats[model] = bootstrap(outs, image_of, reps=args.reps)
            for name, (est, lo, hi, n) in stats[model].items():
                w.writerow([model, name, est, lo, hi, n])

    with open(out / "cells.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["model", "type", "severity", "metric", "estimate", "ci_low", "ci_high", "n"])
        for model, outs in by_model.items():
            for t in ("object", "attribute", "relation"):
                ctrl = [o for o in outs if not o["is_conflict"] and o["type"] == t]
                for sev in ("S1", "S2", "S3"):
                    cell = [o for o in outs if o["is_conflict"] and o["type"] == t and o["severity"] == sev]
                    if not cell:
                        continue
                    b = bootstrap(cell + ctrl, image_of, reps=max(200, args.reps // 4))
                    for name in ("perceive", "DetR", "DetR_adj", "ResA_mcq", "ResA_if_perceived",
                                 "AttA", "AttA_strict", "verbalize_gain"):
                        est, lo, hi, n = b[name]
                        w.writerow([model, t, sev, name, est, lo, hi, n])

    with open(out / "joint.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["model", "pattern", "count", "share"])
        for model, outs in by_model.items():
            for k, (c, share) in joint(outs).items():
                w.writerow([model, k, c, share])

    with open(out / "tests.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["model", "probe_a", "probe_b", "a_only", "b_only", "p_value"])
        for model, outs in by_model.items():
            for a, b in (("verbalize", "mcq"), ("mcq_focus", "mcq"), ("open", "mcq"),
                         ("delay", "mcq"), ("check_silent", "delay"), ("verbalize", "check_silent")):
                a_only, b_only, p = mcnemar(outs, a, b)
                w.writerow([model, a, b, a_only, b_only, p])

    with open(out / "outcomes.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for outs in by_model.values():
            for o in outs:
                f.write(json.dumps(o) + "\n")

    lines = ["\\begin{tabular}{l" + "c" * len(TABLE_METRICS) + "}", "\\toprule",
             "Model & " + " & ".join(h for _, h in TABLE_METRICS) + " \\\\", "\\midrule"]
    for model, st in stats.items():
        cells = []
        for name, _ in TABLE_METRICS:
            est, lo, hi, _ = st[name]
            cells.append(pct(est) if lo is None else f"{pct(est)}\\,{{\\scriptsize$\\pm${100 * (hi - lo) / 2:.1f}}}")
        lines.append(f"{model} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (out / "main_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    width = max(len(m) for m in stats) + 2
    print("\n" + " " * width + "".join(f"{h:>14}" for _, h in TABLE_METRICS))
    for model, st in stats.items():
        row = []
        for name, _ in TABLE_METRICS:
            est, lo, hi, _ = st[name]
            row.append(f"{pct(est):>6} [{pct(lo)},{pct(hi)}]" if lo is not None else f"{pct(est):>14}")
        print(f"{model:<{width}}" + "".join(f"{x:>14}" for x in row))
    print(f"\nwrote {out}/metrics.csv, cells.csv, joint.csv, tests.csv, outcomes.jsonl, main_table.tex")


if __name__ == "__main__":
    main()
