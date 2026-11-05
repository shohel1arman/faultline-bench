"""Build the human-check page: one self-contained HTML file with the 500 sampled photos.

  python scripts/make_annotation.py --images data/images

Writes annotation/faultline_check.html (images embedded; send this one file to each
annotator) and annotation/sample_ids.json (which records are in the sample, tracked in
git). The page shows no cell labels and keeps answers in the browser until exported.
"""

import argparse
import base64
import hashlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faultline.annotation import item_for_page, sample  # noqa: E402
from faultline.runner import cell_of  # noqa: E402
from faultline.schema import read_jsonl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def data_uri(path, max_side, quality):
    from PIL import Image

    im = Image.open(path).convert("RGB")
    scale = max_side / max(im.size)
    if scale < 1:
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/faultline_v0.jsonl")
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--out", default="annotation/faultline_check.html")
    ap.add_argument("--n-per-cell", type=int, default=50)
    ap.add_argument("--n-controls", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-side", type=int, default=640)
    ap.add_argument("--quality", type=int, default=80)
    args = ap.parse_args()

    recs = sample(read_jsonl(args.dataset), args.n_per_cell, args.n_controls, args.seed)
    ids = [r["id"] for r in recs]
    version = hashlib.sha1(",".join(ids).encode()).hexdigest()[:10]

    items = []
    for k, r in enumerate(recs, 1):
        items.append(item_for_page(r, data_uri(Path(args.images) / r["image_file"],
                                               args.max_side, args.quality)))
        if k % 100 == 0:
            print(f"{k}/{len(recs)} images", flush=True)

    # "</" inside the embedded JSON must not end the script element
    payload = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    page = (ROOT / "scripts" / "annotation_template.html").read_text(encoding="utf-8")
    page = page.replace("/*ITEMS*/", payload).replace("/*PAGE_VERSION*/", version)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8", newline="\n")

    cells = {}
    for r in recs:
        cells["/".join(cell_of(r))] = cells.get("/".join(cell_of(r)), 0) + 1
    meta = {"page_version": version, "seed": args.seed, "n": len(ids), "cells": cells, "ids": ids}
    with open(out.parent / "sample_ids.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(ids)} items, version {version})")


if __name__ == "__main__":
    main()
