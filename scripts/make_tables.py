"""LaTeX tables for the paper (booktabs), from the analysis outputs.

  python scripts/make_tables.py

Reads results/spark/{metrics,cells,tests,extra,pairs}.csv and
results/platform_agreement.csv; writes results/tables/*.tex:
  dataset    design of the core set (cells, counts, severity rules)
  main       eleven models: perception, detection, resolution, attribution
  probes     where the verbalize gain comes from (structure, check, report)
  pairs      paired comparisons between sizes within a family
  severity   S1 minus S3, pooled over models, per conflict type
  cells      (supplement) perceive / detect / resolve per cell for every model
  platform   (supplement) Mac float16 vs Spark bfloat16 agreement
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faultline.runner import cell_of, select  # noqa: E402
from faultline.schema import read_jsonl  # noqa: E402

FAMILIES = [
    ("Qwen3-VL", [("qwen3vl-2b", "2B"), ("qwen3vl-4b", "4B"), ("qwen3vl-8b", "8B"),
                  ("qwen3vl-30b-a3b", "30B-A3B"), ("qwen3vl-32b", "32B")]),
    ("Qwen2.5-VL", [("qwen25vl-3b", "3B"), ("qwen25vl-7b", "7B"), ("qwen25vl-32b", "32B")]),
    ("InternVL3", [("internvl3-2b", "2B"), ("internvl3-8b", "8B")]),
    ("SmolVLM2", [("smolvlm2-2b", "2.2B")]),
]
NAME = {k: f"{fam}-{size}" for fam, ms in FAMILIES for k, size in ms}


def read_metrics(path, key=("model", "metric")):
    out = {}
    for r in csv.DictReader(open(path)):
        def num(x):
            return None if x in ("", "None") else 100 * float(x)
        out[tuple(r[k] for k in key)] = (num(r["estimate"]), num(r.get("ci_low", "")), num(r.get("ci_high", "")))
    return out


def fmt(v, sign=False, nd=1):
    if v is None:
        return "--"
    return tex_minus(f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}")


def tex_minus(s):
    """A typographic minus sign in LaTeX (a hyphen sets as a short dash)."""
    return s.replace("-", "$-$")


def bold_best(values, higher=True):
    """Indexes of the best value in a column (ties all bold)."""
    vals = [v for v in values if v is not None]
    if not vals:
        return set()
    best = max(vals) if higher else min(vals)
    return {i for i, v in enumerate(values) if v is not None and abs(v - best) < 1e-9}


def model_rows(body_fn, ncols):
    lines = []
    for fi, (fam, ms) in enumerate(FAMILIES):
        if fi:
            lines.append("\\midrule")
        for k, size in ms:
            lines.append(body_fn(k, fam, size))
    return lines


def write(out, name, lines):
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_dataset(records, out):
    """Compact enough for one column: severity rule per type and severity, record counts."""
    core = select(records, "core")
    cells = Counter(cell_of(r) for r in core)
    imgs = len({r["image_file"] for r in core})
    rows = [
        ("Attribute", "attribute", "far colour, $\\geq$10\\%", "far colour, 2--10\\%", "near colour"),
        ("Object", "object", "other group, $\\geq$10\\%", "other group, 2--10\\%", "same group"),
        ("Relation", "relation", "gap $\\geq$15\\%", "gap 6--15\\%", "gap 2--6\\%"),
    ]
    lines = ["\\begin{tabular}{lcccc}", "\\toprule",
             "Type & S1 (salient) & S2 (moderate) & S3 (subtle) & Controls \\\\", "\\midrule"]
    for label, t, s1, s2, s3 in rows:
        lines.append(f"{label} & {s1} & {s2} & {s3} & -- \\\\")
        lines.append(f" & {cells[(t, 'S1')]} & {cells[(t, 'S2')]} & {cells[(t, 'S3')]} & {cells[('control', t)]} \\\\")
    n_conf = sum(v for c, v in cells.items() if c[0] != "control")
    n_ctrl = sum(v for c, v in cells.items() if c[0] == "control")
    lines += ["\\midrule",
              f"\\multicolumn{{5}}{{l}}{{Core: {len(core):,} records ({n_conf:,} conflicts, {n_ctrl} controls), "
              f"{imgs:,} images; full release {len(records):,}.}} \\\\",
              "\\bottomrule", "\\end{tabular}"]
    write(out, "dataset", lines)


def table_main(m, x, out):
    cols = [("perceive", "Perc."), ("DetR", "Flag"), ("FFR", "FA$\\downarrow$"), ("DetR_adj", "Det."),
            ("ResA_mcq", "Res."), ("ResA_if_perceived", "Res.$|$P"), ("arbitration_failure", "ArbF$\\downarrow$"),
            ("ResA_focus", "Focus"), ("A_span", "Span"), ("A_input", "Input"), ("AttA_strict", "Strict")]
    lower = {"FFR", "arbitration_failure"}
    keys = [k for _, ms in FAMILIES for k, _ in ms]

    def val(k, c):
        return (x if c == "arbitration_failure" else m).get((k, c), (None,))[0]
    best = {c: {keys[i] for i in bold_best([val(k, c) for k in keys], higher=c not in lower)} for c, _ in cols}
    lines = ["\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
             " & \\multicolumn{1}{c}{Perception} & \\multicolumn{3}{c}{Detection} & "
             "\\multicolumn{4}{c}{Resolution} & \\multicolumn{3}{c}{Attribution} \\\\",
             "\\cmidrule(lr){2-2}\\cmidrule(lr){3-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-12}",
             "Model & " + " & ".join(h for _, h in cols) + " \\\\", "\\midrule"]

    def row(k, fam, size):
        cells = []
        for c, _ in cols:
            s = fmt(val(k, c))
            cells.append(f"\\textbf{{{s}}}" if k in best[c] else s)
        return f"{NAME[k]} & " + " & ".join(cells) + " \\\\"
    lines += model_rows(row, len(cols))
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(out, "main", lines)


def table_probes(m, tests, out):
    cols = [("ResA_mcq", "Direct"), ("ResA_delay", "Delay"), ("ResA_check", "Check"), ("ResA_verbalize", "Verbalize"),
            ("structure_gain", "Structure"), ("check_gain", "Check"), ("report_gain", "Report")]
    test_of = {"structure_gain": ("delay", "mcq"), "check_gain": ("check_silent", "delay"),
               "report_gain": ("verbalize", "check_silent")}
    lines = ["\\begin{tabular}{lcccc|ccc}", "\\toprule",
             " & \\multicolumn{4}{c|}{Answers from the image (\\%)} & \\multicolumn{3}{c}{Gain (points)} \\\\",
             "Model & " + " & ".join(h for _, h in cols) + " \\\\", "\\midrule"]

    def row(k, fam, size):
        cells = []
        for c, _ in cols:
            v = m.get((k, c), (None,))[0]
            s = fmt(v, sign=c.endswith("gain"))
            if c in test_of:
                p = tests.get((k, *test_of[c]))
                if p is not None and p < 0.001:
                    s += "$^{\\ast}$"
            cells.append(s)
        return f"{NAME[k]} & " + " & ".join(cells) + " \\\\"
    lines += model_rows(row, len(cols))
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(out, "probes", lines)


def table_pairs(pairs, out):
    lines = ["\\begin{tabular}{llccc}", "\\toprule",
             "Smaller & Larger & $\\Delta$ Res. (points) & Discordant (L/S) & $p$ \\\\", "\\midrule"]
    for r in pairs:
        d, lo, hi = (100 * float(r[k]) for k in ("diff", "low", "high"))
        p = float(r["p_value"])
        ps = "$<10^{-10}$" if p < 1e-10 else (f"{p:.2g}" if p >= 1e-3 else
                                              "${:.1f}\\times10^{{{}}}$".format(*_sci(p)))
        lines.append(f"{NAME[r['smaller']]} & {NAME[r['larger']]} & " + tex_minus(f"{d:+.1f} [{lo:+.1f}, {hi:+.1f}]") + " & "
                     f"{r['larger_only']}/{r['smaller_only']} & {ps} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(out, "pairs", lines)


def _sci(p):
    e = int(f"{p:e}".split("e")[1])
    return p / 10 ** e, e


def table_severity(x, out):
    lines = ["\\begin{tabular}{lcc}", "\\toprule", "Conflict type & Detection & Resolution \\\\", "\\midrule"]
    for t, label in [("attribute", "Attribute"), ("object", "Object"), ("relation", "Relation"), ("all", "All")]:
        cells = []
        for kind in ("detect", "resolve"):
            v, lo, hi = x[("all models", f"s1_minus_s3_{kind}_{t}")]
            cells.append(tex_minus(f"{v:+.1f} [{lo:+.1f}, {hi:+.1f}]"))
        if t == "all":
            lines.append("\\midrule")
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(out, "severity", lines)


def table_cells(cells, out):
    cols = [(t, s) for t in ("attribute", "object", "relation") for s in ("S1", "S2", "S3")]
    lines = ["\\begin{tabular}{l" + "c" * 9 + "}", "\\toprule",
             " & \\multicolumn{3}{c}{Attribute} & \\multicolumn{3}{c}{Object} & \\multicolumn{3}{c}{Relation} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
             "Model & " + " & ".join(s for _, s in cols) + " \\\\", "\\midrule"]

    def row(k, fam, size):
        out_cells = []
        for t, s in cols:
            vals = [cells.get((k, t, s, mt), (None,))[0] for mt in ("perceive", "DetR", "ResA_mcq")]
            out_cells.append("/".join(fmt(v, nd=0) for v in vals))
        return f"{NAME[k]} & " + " & ".join(out_cells) + " \\\\"
    lines += model_rows(row, 9)
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(out, "cells", lines)


def table_platform(path, out):
    rows = list(csv.DictReader(open(path)))
    cols = [("same_mcq", "MC letter"), ("same_verbalize", "Verbalize"), ("same_detect", "Detection"),
            ("same_attr_type", "Error type"), ("diff_DetR_adj", "$\\Delta$ Det."), ("diff_ResA_mcq", "$\\Delta$ Res.")]
    lines = ["\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
             " & \\multicolumn{4}{c}{Identical outputs (\\%)} & \\multicolumn{2}{c}{Metric change (points)} \\\\",
             "Model & " + " & ".join(h for _, h in cols) + " \\\\", "\\midrule"]
    for r in rows:
        cells = []
        for c, _ in cols:
            v = r.get(c)
            cells.append("--" if v in (None, "") else fmt(100 * float(v), sign=c.startswith("diff")))
        lines.append(f"{NAME[r['model']]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(out, "platform", lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/spark")
    ap.add_argument("--dataset", default="data/faultline_v0.jsonl")
    ap.add_argument("--out", default="results/tables")
    args = ap.parse_args()
    res, out = Path(args.results), Path(args.out)

    m = read_metrics(res / "metrics.csv")
    x = read_metrics(res / "extra.csv")
    cells = read_metrics(res / "cells.csv", key=("model", "type", "severity", "metric"))
    tests = {(r["model"], r["probe_a"], r["probe_b"]): float(r["p_value"]) for r in csv.DictReader(open(res / "tests.csv"))}
    pairs = list(csv.DictReader(open(res / "pairs.csv")))

    table_dataset(read_jsonl(args.dataset), out)
    table_main(m, x, out)
    table_probes(m, tests, out)
    table_pairs(pairs, out)
    table_severity(x, out)
    table_cells(cells, out)
    table_platform(Path("results/platform_agreement.csv"), out)
    print(f"wrote dataset, main, probes, pairs, severity, cells and platform tables to {out}/")


if __name__ == "__main__":
    main()
