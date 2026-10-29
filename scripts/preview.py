"""Write an HTML page showing a few samples per cell with their images and boxes.

Usage: python scripts/preview.py --dataset data/faultline_v0.jsonl --per-cell 2
Then open data/preview.html in a browser. Images are read from data/images/.
"""

import argparse
import html
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faultline.schema import read_jsonl  # noqa: E402

CELLS = [(t, s) for t in ("object", "attribute", "relation") for s in ("S1", "S2", "S3")]

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>FaultLine preview</title>
<style>
body {{ font: 14px/1.45 system-ui, sans-serif; margin: 16px; background: #fafafa; color: #222; }}
h2 {{ margin: 28px 0 8px; font-size: 16px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 16px; }}
.card {{ background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 10px; }}
.img {{ position: relative; }}
.img img {{ width: 100%; display: block; }}
.box {{ position: absolute; border: 2px solid; box-sizing: border-box; }}
.t {{ border-color: #e00; }} .a {{ border-color: #06f; }}
.edit {{ background: #ffe08a; font-weight: 600; }}
.meta {{ color: #666; font-size: 12px; }}
.legend span {{ padding: 0 6px; border-left: 4px solid; margin-right: 10px; }}
</style>
<h1>FaultLine preview ({n} samples)</h1>
<p class="legend"><span style="border-color:#e00">target</span>
<span style="border-color:#06f">anchor</span> highlighted text = edited span</p>
{body}
"""


def box(obj, rec, cls):
    if not obj:
        return ""
    x, y, w, h = obj["bbox"]
    W, H = rec["width"], rec["height"]
    return (f'<div class="box {cls}" style="left:{100 * x / W:.2f}%;top:{100 * y / H:.2f}%;'
            f'width:{100 * w / W:.2f}%;height:{100 * h / H:.2f}%"></div>')


def text_html(rec):
    t = rec["text"]
    if not rec["is_conflict"]:
        return html.escape(t)
    s0, s1 = rec["edited_span_char"]
    return (html.escape(t[:s0]) + '<span class="edit">' + html.escape(t[s0:s1]) + "</span>"
            + html.escape(t[s1:]))


def card(rec, image_dir):
    img = f"{image_dir}/{rec['image_file']}"
    return f"""<div class="card">
<div class="img"><img src="{html.escape(img)}" loading="lazy">{box(rec['target'], rec, 't')}{box(rec['anchor'], rec, 'a')}</div>
<p>{text_html(rec)}</p>
<p><b>Q:</b> {html.escape(rec['question'])}<br>
<b>options:</b> {html.escape(' / '.join(rec['mcq_options']))}<br>
<b>image says:</b> {html.escape(rec['answer_image'])} &nbsp; <b>text says:</b> {html.escape(rec['answer_text'])}</p>
<p class="meta">{rec['id']} · image {rec['image_id']} · target area {rec['target']['rel_area']:.3f}
{'· gap ' + format(rec['relation_gap'], '.3f') if rec.get('relation_gap') is not None else ''}</p>
</div>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/faultline_v0.jsonl")
    ap.add_argument("--images", default="images", help="image folder, relative to the output page")
    ap.add_argument("--out", default="data/preview.html")
    ap.add_argument("--per-cell", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    recs = [r for r in read_jsonl(args.dataset) if r["split"] == "main"]
    rng = random.Random(args.seed)
    sections = []
    n = 0
    groups = [((t, s), f"{t} {s}") for t, s in CELLS] + \
             [(("control", t), f"control ({t} question)") for t in ("object", "attribute", "relation")]
    for key, title in groups:
        if key[0] == "control":
            pool = [r for r in recs if not r["is_conflict"] and r["question_type"] == key[1]]
        else:
            pool = [r for r in recs if r["conflict_type"] == key[0] and r["severity"] == key[1]]
        pick = rng.sample(pool, min(args.per_cell, len(pool)))
        n += len(pick)
        cards = "\n".join(card(r, args.images) for r in pick)
        sections.append(f"<h2>{html.escape(title)}</h2>\n<div class=\"grid\">\n{cards}\n</div>")
    Path(args.out).write_text(PAGE.format(n=n, body="\n".join(sections)), encoding="utf-8")
    print(f"wrote {args.out} with {n} samples")


if __name__ == "__main__":
    main()
