"""Build the hosted version of the human-check page: index.html plus one JPEG per photo.

  python scripts/make_annotation_site.py --images data/images
  python scripts/make_annotation_site.py --from-page annotation/faultline_check.html

The second form takes the photos out of the single-file page built by make_annotation.py,
so the hosted page shows exactly the same images. Writes annotation/site/ (not tracked).
The hosted page saves answers to the signed-in annotator's account when the host offers a
shared store, and otherwise in the browser with an Export button, like the single file.
"""

import argparse
import base64
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faultline.annotation import sample  # noqa: E402
from faultline.schema import read_jsonl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def photos_from_page(path):
    """{record id: JPEG bytes} from the embedded items of the single-file page."""
    html = Path(path).read_text(encoding="utf-8")
    m = re.search(r'<script id="items" type="application/json">(.*?)</script>', html, re.S)
    if not m:
        sys.exit(f"no embedded items in {path}")
    items = json.loads(m.group(1).replace("<\\/", "</"))
    out = {}
    for it in items:
        head, b64 = it["image"].split(",", 1)
        if not head.startswith("data:image/jpeg"):
            sys.exit(f"unexpected image type for {it['id']}: {head}")
        out[it["id"]] = base64.b64decode(b64)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/faultline_v0.jsonl")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", help="data/images, resized here like make_annotation.py does")
    src.add_argument("--from-page", help="a faultline_check.html built by make_annotation.py")
    ap.add_argument("--out", default="annotation/site")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-side", type=int, default=640)
    ap.add_argument("--quality", type=int, default=80)
    args = ap.parse_args()

    recs = sample(read_jsonl(args.dataset), seed=args.seed)
    ids = [r["id"] for r in recs]
    version = hashlib.sha1(",".join(ids).encode()).hexdigest()[:10]
    out = Path(args.out)
    (out / "img").mkdir(parents=True, exist_ok=True)

    if args.from_page:
        photos = photos_from_page(args.from_page)
        missing = [i for i in ids if i not in photos]
        if missing:
            sys.exit(f"{len(missing)} sampled records are not in {args.from_page} (different sample?)")
    else:
        import io

        from PIL import Image
        photos = {}
        for r in recs:
            im = Image.open(Path(args.images) / r["image_file"]).convert("RGB")
            scale = args.max_side / max(im.size)
            if scale < 1:
                im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=args.quality, optimize=True)
            photos[r["id"]] = buf.getvalue()

    items, written = [], set()
    for r in recs:
        name = Path(r["image_file"]).stem + ".jpg"   # two records can share a photo
        if name not in written:
            (out / "img" / name).write_bytes(photos[r["id"]])
            written.add(name)
        items.append({"id": r["id"], "image": "img/" + name, "question": r["question"],
                      "options": r["mcq_options"], "text": r["text"]})

    payload = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    page = (ROOT / "scripts" / "annotation_site_template.html").read_text(encoding="utf-8")
    page = page.replace("/*ITEMS*/", payload).replace("/*PAGE_VERSION*/", version)
    (out / "index.html").write_text(page, encoding="utf-8", newline="\n")
    size = sum(p.stat().st_size for p in (out / "img").glob("*.jpg"))
    print(f"wrote {out}/index.html and {len(written)} photos ({size / 1e6:.1f} MB), "
          f"{len(items)} items, version {version}")


if __name__ == "__main__":
    main()
