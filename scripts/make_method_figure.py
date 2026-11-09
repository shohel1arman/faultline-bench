"""Method overview figure: how a FaultLine pair is built and how every model is probed.

  python scripts/make_method_figure.py

Writes results/figures/method.pdf and .png. The running example is the teaser record
(fl_01827, a white airplane described as green), so Figures 1 and 2 tell one story.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK, INK2, LINE, FILL, PANEL = "#0b0b0b", "#52514e", "#bdbbb4", "#f4f3f0", "#fbfbfa"
IMAGE_C, TEXT_C = "#2a78d6", "#eb6834"


def box(ax, x, y, w, h, title, body=None, fill=FILL, title_color=INK, size=6.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012",
                                fc=fill, ec=LINE, lw=0.7))
    ax.text(x + 0.008, y + h - 0.012, title, ha="left", va="top", fontsize=size + 0.4, color=title_color,
            fontweight="bold")
    if body:
        ax.text(x + 0.008, y + h - 0.05, body, ha="left", va="top", fontsize=size, color=INK2, linespacing=1.3)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, color=INK2, lw=0.8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/faultline_v0.jsonl")
    ap.add_argument("--id", default="fl_01827")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()
    rec = next(json.loads(l) for l in open(args.dataset, encoding="utf-8") if json.loads(l)["id"] == args.id)

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(7.0, 2.9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    # panel labels
    ax.text(0.0, 0.985, "(a) Building a pair", fontsize=8, fontweight="bold", va="top", color=INK)
    ax.text(0.385, 0.985, "(b) Probing every model on the same pair", fontsize=8, fontweight="bold", va="top",
            color=INK)

    # (a) construction, top to bottom
    box(ax, 0.0, 0.73, 0.35, 0.2, "Scene graph + photo",
        "GQA objects, attributes, boxes (Visual Genome)\nrelations recomputed from the boxes\n"
        "people, hypernyms, ambiguous words excluded")
    arrow(ax, 0.175, 0.73, 0.175, 0.69)
    box(ax, 0.0, 0.47, 0.35, 0.22, "Edit exactly one span",
        "object | attribute (colour) | relation\nS1 salient, S2 moderate, S3 subtle\n"
        "(object size, edit distance, spatial gap)\nfair allocation over 9 cells and controls")
    arrow(ax, 0.175, 0.47, 0.175, 0.43)
    a, b = rec["edited_span_char"]
    t = rec["text"]
    box(ax, 0.0, 0.05, 0.35, 0.38, "Pair (example)", None)
    ax.text(0.008, 0.36, "Faithful:", fontsize=6.6, color=INK2, va="top")
    ax.text(0.085, 0.36, rec["text_clean"], fontsize=6.6, color=INK, va="top")
    ax.text(0.008, 0.31, "Edited:", fontsize=6.6, color=INK2, va="top")
    pre = ax.text(0.085, 0.31, t[:a], fontsize=6.6, color=INK, va="top")
    fig.canvas.draw()
    x1 = pre.get_window_extent().x1
    xs = ax.transAxes.inverted().transform((x1, 0))[0]
    span = ax.text(xs, 0.31, t[a:b], fontsize=6.6, color=TEXT_C, va="top", fontweight="bold")
    fig.canvas.draw()
    xe = ax.transAxes.inverted().transform((span.get_window_extent().x1, 0))[0]
    ax.text(xe, 0.31, t[b:], fontsize=6.6, color=INK, va="top")
    ax.text(0.008, 0.25, "Question:", fontsize=6.6, color=INK2, va="top")
    ax.text(0.085, 0.25, rec["question"], fontsize=6.6, color=INK, va="top")
    ax.text(0.008, 0.20, "Options:", fontsize=6.6, color=INK2, va="top")
    opts = []
    for o in rec["mcq_options"]:
        tag = " (image)" if o == rec["answer_image"] else " (text)" if o == rec["answer_text"] else ""
        opts.append(o + tag)
    ax.text(0.085, 0.20, ",  ".join(opts), fontsize=6.6, color=INK, va="top")
    ax.text(0.008, 0.13, "One edited span per record (checked); faithful\ncontrols use the unedited "
            "description.", fontsize=6.2, color=INK2, va="top", linespacing=1.3)

    arrow(ax, 0.355, 0.24, 0.385, 0.24)

    # (b) probes, a 3 x 2 grid of probe groups
    col_x = [0.39, 0.595, 0.80]
    w = 0.195
    groups = [
        (col_x[0], 0.6, "Perception controls", "image only: no description\ntext only: no image",
         "perceive, read"),
        (col_x[1], 0.6, "Detection", '"Does the description\naccurately describe\nthe image?" yes / no',
         "flags; false alarms"),
        (col_x[2], 0.6, "Resolution", "multiple choice\nopen answer\n" + '"rely on the image"',
         "which input it follows"),
        (col_x[0], 0.22, "Attribution", "Error | Wrong input\nError type | Wrong words\nHow obvious",
         "each part; all four"),
        (col_x[1], 0.22, "Turn-structure controls", 'delay: text, "OK.", question\ncheck: report prompt,\n'
         '"OK.", question', "structure; check"),
        (col_x[2], 0.22, "Verbalize", "own report as the\nassistant turn, then\nthe question",
         "report gain"),
    ]
    for x, y, title, body, measures in groups:
        box(ax, x, y, w, 0.33, title, body, fill=PANEL)
        ax.text(x + 0.008, y + 0.03, "measures: " + measures, fontsize=5.9, color=IMAGE_C, va="bottom")
    ax.text(0.39, 0.165, "Per record: detected / resolved / attributed; resolution among perceived facts;\n"
            "arbitration failure (perceived and flagged, still answered from the text).\n"
            "95% intervals from a bootstrap over images.", fontsize=6.0, color=INK2, va="top", linespacing=1.35)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "method.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(out / "method.png", bbox_inches="tight", dpi=300)
    print(f"wrote {out}/method.pdf and .png")


if __name__ == "__main__":
    main()
