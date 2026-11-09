"""Figures with real photos: the teaser (Figure 1) and one example per conflict type.

Run where the photos are (data/images, on the Mac or the Spark):

  python scripts/make_examples.py candidates     contact sheets to choose from
  python scripts/make_examples.py teaser [--id fl_xxxxx]
  python scripts/make_examples.py examples [--ids attribute/S1=fl_..,object/S3=fl_..]

The teaser shows one conflict and how four models answer it: whether they flag the
description, which option they pick when asked directly, when told to rely on the image,
and after writing their own error report. Answers that agree with the photo are blue,
answers that follow the description are orange (the same colours as the outcome figure).
Outputs go to results/figures/ (teaser.pdf/png, examples.pdf/png); contact sheets go to
results/figures/candidates/ (not tracked).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker  # noqa: E402

from faultline.runner import cell_of, select  # noqa: E402
from faultline.schema import read_jsonl  # noqa: E402
from faultline.score import outcome  # noqa: E402

IMAGE_C, TEXT_C, INK, INK2, BOX = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#eda100"
TEASER_MODELS = [("qwen3vl-8b", "Qwen3-VL-8B"), ("qwen3vl-32b", "Qwen3-VL-32B"),
                 ("qwen25vl-7b", "Qwen2.5-VL-7B"), ("qwen25vl-32b", "Qwen2.5-VL-32B")]


def load_outputs(root, models):
    out = {}
    for m in models:
        rows = {}
        with open(Path(root) / m / "core.jsonl", encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("error") is None and r.get("probes"):
                    rows[r["id"]] = r
        out[m] = rows
    return out


def picked(rec, probes, name):
    """The option word a model chose in a multiple-choice probe."""
    c = probes.get(name, {}).get("choice")
    return None if c is None else rec["mcq_options"][ord(c) - ord("A")]


def show_photo(ax, rec, images):
    from PIL import Image
    im = Image.open(Path(images) / rec["image_file"]).convert("RGB")
    ax.imshow(im)
    sx, sy = im.width / rec["width"], im.height / rec["height"]   # boxes are in original pixels
    boxes = [rec["target"]] + ([rec["anchor"]] if rec["conflict_type"] == "relation" and rec.get("anchor") else [])
    for b, style in zip(boxes, ["-", "--"]):
        x, y, w, h = b["bbox"]
        ax.add_patch(patches.Rectangle((x * sx, y * sy), w * sx, h * sy, fill=False, ec=BOX, lw=1.6, ls=style))
    ax.set_axis_off()


def words_with_edit(rec, highlight=TEXT_C):
    """[(word, color, weight, space_before)] for the description, edited span highlighted.

    Punctuation right after the span ("brown.") stays attached to it.
    """
    t = rec["text"]
    a, b = rec["edited_span_char"]
    out, prev_space = [], False
    for piece, col, wt in [(t[:a], INK, "normal"), (t[a:b], highlight, "bold"), (t[b:], INK, "normal")]:
        if not piece:
            continue
        for j, w in enumerate(piece.split(" ")):
            if w:
                out.append((w, col, wt, bool(out) and (j > 0 or prev_space)))
        prev_space = piece.endswith(" ")
    return out


def plain(words, color=INK, weight="normal"):
    return [(w, color, weight, True) for w in words]


def wrapped_lines(tokens, width):
    lines, cur, n = [], [], 0
    for tok in tokens:
        add = len(tok[0]) + (1 if cur and tok[3] else 0)
        if cur and tok[3] and n + add > width:
            lines.append(cur)
            cur, n = [], 0
            add = len(tok[0])
        cur.append(tok)
        n += add
    if cur:
        lines.append(cur)
    return lines


def text_block(lines, size):
    rows = []
    for line in lines:
        parts = []
        for k, (w, col, wt, space) in enumerate(line):
            parts.append(TextArea((" " if k and space else "") + w,
                                  textprops=dict(color=col, fontweight=wt, fontsize=size)))
        rows.append(HPacker(children=parts, align="baseline", pad=0, sep=0))
    return VPacker(children=rows, align="left", pad=0, sep=2)


def teaser_candidates(recs, outs):
    """Conflicts that show the pattern clearly: every shown model sees the fact alone, the
    32B models flag the conflict yet answer from the text, and follow the image when told."""
    cands = []
    for r in recs:
        if not r["is_conflict"] or r["severity"] != "S1" or r["conflict_type"] == "relation":
            continue
        if any(r["id"] not in outs[m] for m, _ in TEASER_MODELS):
            continue
        o = {m: outcome(r, outs[m][r["id"]]["probes"]) for m, _ in TEASER_MODELS}
        if not all(o[m]["perceives"] for m in o):
            continue
        big = [m for m in o if m.endswith("32b")]
        if all(o[m]["flagged"] and o[m]["mcq"] == "text" and o[m]["mcq_focus"] == "image" for m in big) \
                and any(o[m]["mcq"] == "image" for m in o if not m.endswith("32b")):
            cands.append((r["target"]["rel_area"], r))
    return [r for _, r in sorted(cands, key=lambda t: -t[0])]


def fig_teaser(rec, outs, images, out):
    fig = plt.figure(figsize=(7.0, 2.45))
    ax_img = fig.add_axes([0.0, 0.02, 0.30, 0.96])
    show_photo(ax_img, rec, images)
    ax = fig.add_axes([0.32, 0.0, 0.68, 1.0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    words = words_with_edit(rec)
    words[0] = words[0][:3] + (True,)          # a space after the "Description:" label
    desc = wrapped_lines(plain(["Description:"], INK2) + words, 70)
    q = wrapped_lines(plain(["Question:"], INK2) + plain(rec["question"].split()) +
                      plain(["(options: " + ", ".join(rec["mcq_options"]) + ")"], INK2), 70)
    block = text_block(desc + q, 8)
    ax.add_artist(AnchoredOffsetbox(loc="upper left", child=block, frameon=False,
                                    bbox_to_anchor=(0.0, 1.0), bbox_transform=ax.transAxes, borderpad=0.3))
    cols = ["Flags the\ndescription?", "Answer,\nasked directly", "Answer, told to\nrely on the image",
            "Answer after its\nown error report"]
    xs = [0.30, 0.47, 0.66, 0.86]
    y0 = 0.56
    for x, c in zip(xs, cols):
        ax.text(x, y0, c, ha="center", va="bottom", fontsize=7, color=INK2)
    ax.plot([0, 1], [y0 - 0.02, y0 - 0.02], color="#d9d8d3", lw=0.6)
    for i, (m, label) in enumerate(TEASER_MODELS):
        y = y0 - 0.1 - i * 0.12
        probes = outs[m][rec["id"]]["probes"]
        o = outcome(rec, probes)
        ax.text(0.0, y, label, ha="left", va="center", fontsize=8, color=INK)
        ax.text(xs[0], y, "yes" if o["flagged"] else "no", ha="center", va="center", fontsize=8,
                color=INK, fontweight="bold" if o["flagged"] else "normal")
        for x, p in zip(xs[1:], ["mcq", "mcq_focus", "verbalize"]):
            w = picked(rec, probes, p)
            col = IMAGE_C if w == rec["answer_image"] else TEXT_C if w == rec["answer_text"] else INK2
            ax.text(x, y, w or "-", ha="center", va="center", fontsize=8, color=col, fontweight="bold")
    ax.text(0.0, 0.0, f"blue: the photo's answer ({rec['answer_image']})", fontsize=7, color=IMAGE_C, va="bottom")
    ax.text(0.42, 0.0, f"orange: the description's answer ({rec['answer_text']})", fontsize=7, color=TEXT_C,
            va="bottom")
    save(fig, out, "teaser")


def examples_default(recs):
    pick = {}
    for r in recs:
        c = cell_of(r)
        if c[0] != "control" and c[1] in ("S1", "S3") and c not in pick:
            pick[c] = r
    return pick


def fig_examples(chosen, images, out):
    order = [(t, s) for s in ("S1", "S3") for t in ("attribute", "object", "relation")]
    fig = plt.figure(figsize=(7.0, 5.0))
    for k, cell in enumerate(order):
        r = chosen[cell]
        col, row = k % 3, k // 3
        ax = fig.add_axes([col / 3 + 0.01, 0.53 - row * 0.5 + 0.12, 1 / 3 - 0.02, 0.33])
        show_photo(ax, r, images)
        ax.set_anchor("S")
        ax.set_title(f"{cell[0]}, {cell[1]}", fontsize=8, color=INK, loc="left")
        tax = fig.add_axes([col / 3 + 0.01, 0.53 - row * 0.5, 1 / 3 - 0.02, 0.11])
        tax.set_axis_off()
        lines = wrapped_lines(words_with_edit(r), 42)
        lines.append(plain(["photo:"], INK2) + plain([r["original_span"]], IMAGE_C, "bold"))
        tax.add_artist(AnchoredOffsetbox(loc="upper left", child=text_block(lines, 7), frameon=False,
                                         bbox_to_anchor=(0, 1), bbox_transform=tax.transAxes, borderpad=0))
    save(fig, out, "examples")


def contact_sheet(recs, images, path, title):
    n = len(recs)
    if not n:
        print(f"no candidates for {title}")
        return
    cols = 4
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(cols * 2.6, rows * 2.9))
    for k, r in enumerate(recs):
        ax = fig.add_subplot(rows, cols, k + 1)
        show_photo(ax, r, images)
        ax.set_title(f"{r['id']}  {r['original_span']} -> {r['edited_span']}", fontsize=7)
    fig.suptitle(title, fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(out / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {out}/{name}.pdf and .png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["candidates", "teaser", "examples"])
    ap.add_argument("--dataset", default="data/faultline_v0.jsonl")
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--outputs", default="outputs/spark")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--id", help="teaser record")
    ap.add_argument("--ids", help="examples: comma-separated type/severity=id, e.g. object/S1=fl_00008")
    args = ap.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})

    recs = select(read_jsonl(args.dataset), "core")
    by_id = {r["id"]: r for r in recs}
    out = Path(args.out)
    if args.what in ("candidates", "teaser"):
        outs = load_outputs(args.outputs, [m for m, _ in TEASER_MODELS])
        cands = teaser_candidates(recs, outs)
        print(f"{len(cands)} teaser candidates")
    if args.what == "candidates":
        contact_sheet(cands[:16], args.images, out / "candidates" / "teaser.png",
                      "Teaser candidates (largest target first): pick one with --id")
        for t in ("attribute", "object", "relation"):
            for s in ("S1", "S3"):
                cell = [r for r in recs if cell_of(r) == (t, s)][:12]
                contact_sheet(cell, args.images, out / "candidates" / f"{t}_{s}.png",
                              f"{t} {s}: pick one with --ids {t}/{s}=<id>")
    elif args.what == "teaser":
        rec = by_id[args.id] if args.id else (cands[0] if cands else None)
        if rec is None:
            sys.exit("no candidate found; pass --id")
        print(f"teaser record {rec['id']}")
        fig_teaser(rec, outs, args.images, out)
    else:
        chosen = examples_default(recs)
        for item in (args.ids or "").split(","):
            if item.strip():
                cell, rid = item.split("=")
                t, s = cell.split("/")
                chosen[(t, s)] = by_id[rid.strip()]
        print("examples: " + ", ".join(f"{t}/{s}={r['id']}" for (t, s), r in sorted(chosen.items())))
        fig_examples(chosen, args.images, out)


if __name__ == "__main__":
    main()
