"""Paper figures from the analysis outputs (results/spark by default).

  python scripts/make_figures.py

Writes PDF (for LaTeX) and PNG (for checking) to results/figures/:
  scaling      three panels against model size: false-alarm-corrected detection,
               answers from the image by default, and when told to rely on the image
  outcomes     share of conflicts by joint outcome (noticed or not x image or text answer)
  cells        answers from the image per cell (type x severity) for every model
  probes       answers from the image under each probe: direct question, delayed question,
               after the check prompt, after the model's own error report, with "rely on the image"
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# reference categorical slots in fixed order (validated: adjacent CVD dE >= 9.1)
FAMILY = {
    "Qwen3-VL": ("#2a78d6", "o"),
    "Qwen2.5-VL": ("#eb6834", "s"),
    "InternVL3": ("#1baf7a", "^"),
    "SmolVLM2": ("#eda100", "D"),
}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# model key -> (family, display name, billions of parameters, mixture of experts)
MODELS = {
    "qwen3vl-2b": ("Qwen3-VL", "Qwen3-VL-2B", 2, False),
    "qwen3vl-4b": ("Qwen3-VL", "Qwen3-VL-4B", 4, False),
    "qwen3vl-8b": ("Qwen3-VL", "Qwen3-VL-8B", 8, False),
    "qwen3vl-30b-a3b": ("Qwen3-VL", "Qwen3-VL-30B-A3B", 30, True),
    "qwen3vl-32b": ("Qwen3-VL", "Qwen3-VL-32B", 32, False),
    "qwen25vl-3b": ("Qwen2.5-VL", "Qwen2.5-VL-3B", 3, False),
    "qwen25vl-7b": ("Qwen2.5-VL", "Qwen2.5-VL-7B", 7, False),
    "qwen25vl-32b": ("Qwen2.5-VL", "Qwen2.5-VL-32B", 32, False),
    "internvl3-2b": ("InternVL3", "InternVL3-2B", 2, False),
    "internvl3-8b": ("InternVL3", "InternVL3-8B", 8, False),
    "smolvlm2-2b": ("SmolVLM2", "SmolVLM2-2.2B", 2.2, False),
}


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "legend.frameon": False, "pdf.fonttype": 42, "savefig.dpi": 300,
    })


def load_metrics(path):
    m = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        def num(x):
            return None if x in ("", "None") else 100 * float(x)
        m[r["model"]][r["metric"]] = (num(r["estimate"]), num(r["ci_low"]), num(r["ci_high"]))
    return m


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None})  # reproducible files
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def fig_scaling(m, out):
    panels = [("DetR_adj", "Detection minus false alarms (%)"),
              ("ResA_mcq", "Answers from the image (%)"),
              ("ResA_focus", "... when told to rely on the image (%)")]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35), sharey=True)
    for ax, (metric, title) in zip(axes, panels):
        for fam, (color, marker) in FAMILY.items():
            dense = sorted((p, k) for k, (f, _, p, moe) in MODELS.items() if f == fam and not moe and k in m)
            xs = [p for p, _ in dense]
            ys = [m[k][metric][0] for _, k in dense]
            lo = [m[k][metric][0] - m[k][metric][1] for _, k in dense]
            hi = [m[k][metric][2] - m[k][metric][0] for _, k in dense]
            ax.errorbar(xs, ys, yerr=[lo, hi], color=color, marker=marker, ms=4.5, lw=1.6,
                        elinewidth=0.8, capsize=0, mec="white", mew=0.8, label=fam, zorder=3)
            for k, (f, _, p, moe) in MODELS.items():
                if f == fam and moe and k in m:
                    v, lo1, hi1 = m[k][metric]
                    ax.errorbar([p], [v], yerr=[[v - lo1], [hi1 - v]], color=color, marker=marker, ms=5,
                                mfc="white", mew=1.2, elinewidth=0.8, capsize=0, ls="none", zorder=4)
        ax.set_xscale("log")
        ax.set_xticks([2, 4, 8, 16, 32])
        ax.set_xticklabels(["2B", "4B", "8B", "16B", "32B"])
        ax.minorticks_off()
        ax.set_ylim(0, 100)
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        ax.set_title(title, fontsize=8, color=INK, loc="left")
        ax.set_xlabel("Parameters (log scale)", color=INK2)
    axes[0].set_ylabel("% of conflicts")
    handles, labels = axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles.append(Line2D([], [], color=INK2, marker="o", mfc="white", ls="none", ms=5))
    labels.append("Qwen3-VL-30B-A3B (mixture of experts)")
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.1), fontsize=7.5,
               handletextpad=0.4, columnspacing=1.2)
    fig.tight_layout(w_pad=1.2)
    save(fig, out, "scaling")


def fig_outcomes(joint_path, out):
    shares = defaultdict(lambda: defaultdict(float))
    for r in csv.DictReader(open(joint_path)):
        p, s = r["pattern"], float(r["share"])
        noticed = p[0] == "D"
        image = p[1] == "R"
        shares[r["model"]][(noticed, image)] += 100 * s
    order = sorted((k for k in MODELS if k in shares), key=lambda k: shares[k][(True, True)])
    cats = [((True, True), "Noticed, answered from image", FAMILY["Qwen3-VL"][0]),
            ((True, False), "Noticed, answered from text", FAMILY["Qwen2.5-VL"][0]),
            ((False, True), "Not noticed, answered from image", FAMILY["InternVL3"][0]),
            ((False, False), "Not noticed, answered from text", FAMILY["SmolVLM2"][0])]
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    left = [0.0] * len(order)
    for key, label, color in cats:
        vals = [shares[k][key] for k in order]
        ax.barh(range(len(order)), vals, left=left, color=color, height=0.72, label=label,
                edgecolor="white", linewidth=1.0)
        for i, (v, l0) in enumerate(zip(vals, left)):
            if v >= 12:
                ax.text(l0 + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if key[0] else INK)
        left = [a + b for a, b in zip(left, vals)]
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([MODELS[k][1] for k in order])
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of conflicts")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.38, -0.14), ncol=2, fontsize=6.5, handlelength=1,
              columnspacing=0.8)
    save(fig, out, "outcomes")


def fig_cells(cells_path, m, out):
    v = {}
    for r in csv.DictReader(open(cells_path)):
        if r["metric"] == "ResA_mcq" and r["estimate"] not in ("", "None"):
            v[(r["model"], r["type"], r["severity"])] = 100 * float(r["estimate"])
    order = sorted((k for k in MODELS if k in m), key=lambda k: m[k]["ResA_mcq"][0], reverse=True)
    cols = [(t, s) for t in ("attribute", "object", "relation") for s in ("S1", "S2", "S3")]
    grid = [[v.get((k, t, s)) for t, s in cols] for k in order]
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(3.4, 3.1))
    im = ax.imshow([[x if x is not None else 0 for x in row] for row in grid], cmap=cmap, vmin=0, vmax=100,
                   aspect="auto")
    for i, row in enumerate(grid):
        for j, x in enumerate(row):
            if x is not None:
                ax.text(j, i, f"{x:.0f}", ha="center", va="center", fontsize=6.3,
                        color="white" if x >= 50 else INK)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([s for _, s in cols])
    for j, t in enumerate(("attribute", "object", "relation")):
        ax.text(3 * j + 1, -0.9, t, ha="center", va="bottom", fontsize=7.5, color=INK)
    for x in (2.5, 5.5):
        ax.axvline(x, color="white", lw=2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([MODELS[k][1] for k in order])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Answers from the image (%)", fontsize=7)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.5, length=0)
    save(fig, out, "cells")


def fig_probes(m, out):
    order = ["qwen3vl-2b", "qwen3vl-4b", "qwen3vl-8b", "qwen3vl-30b-a3b", "qwen3vl-32b",
             "qwen25vl-3b", "qwen25vl-7b", "qwen25vl-32b", "internvl3-2b", "internvl3-8b", "smolvlm2-2b"]
    order = [k for k in order if k in m][::-1]
    probes = [("ResA_mcq", "Direct question", "#2a78d6", "o"),
              ("ResA_delay", "Question in a later turn", "#eb6834", "s"),
              ("ResA_check", "After the check prompt", "#1baf7a", "^"),
              ("ResA_verbalize", "After its own error report", "#eda100", "D"),
              ("ResA_focus", "Told to rely on the image", "#e87ba4", "P")]
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    for i, k in enumerate(order):
        vals = [m[k][p][0] for p, *_ in probes if m[k].get(p, (None,))[0] is not None]
        ax.plot([min(vals), max(vals)], [i, i], color=GRID, lw=2.2, zorder=1, solid_capstyle="round")
        for p, label, color, marker in probes:
            v = m[k].get(p, (None,))[0]
            if v is not None:
                ax.scatter([v], [i], color=color, marker=marker, s=26, zorder=3, edgecolors="white",
                           linewidths=0.6, label=label if i == 0 else None)
    for i in range(1, len(order)):
        fam_prev, fam = MODELS[order[i - 1]][0], MODELS[order[i]][0]
        if fam != fam_prev:
            ax.axhline(i - 0.5, color=GRID, lw=0.6)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([MODELS[k][1] for k in order])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Answers from the image (% of conflicts)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.35, -0.13), ncol=2, fontsize=6.5, handletextpad=0.2,
              columnspacing=0.8, scatterpoints=1)
    save(fig, out, "probes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/spark")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()
    style()
    res, out = Path(args.results), Path(args.out)
    m = load_metrics(res / "metrics.csv")
    fig_scaling(m, out)
    fig_outcomes(res / "joint.csv", out)
    fig_cells(res / "cells.csv", m, out)
    fig_probes(m, out)
    print(f"wrote scaling, outcomes, cells and probes to {out}/ (pdf and png)")


if __name__ == "__main__":
    main()
