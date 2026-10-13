"""Download GQA scene graphs and Visual Genome image metadata (about 60 MB in total).

Usage: python scripts/download_data.py --out data/raw
"""

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

SOURCES = {
    "sceneGraphs.zip": "https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip",
    "image_data.json.zip": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip",
}


def fetch(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"have {dest.name}")
        return
    print(f"get  {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                sys.stdout.write(f"\r     {done / 1e6:.1f} / {total / 1e6:.1f} MB")
                sys.stdout.flush()
    print()
    tmp.rename(dest)


def extract(zpath, out, wanted):
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        print(f"{zpath.name}: {names}")
        for n in names:
            if Path(n).name in wanted and not (out / Path(n).name).exists():
                with z.open(n) as src, open(out / Path(n).name, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                print(f"     extracted {Path(n).name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for fname, url in SOURCES.items():
        fetch(url, out / fname)
    extract(out / "sceneGraphs.zip", out, {"val_sceneGraphs.json"})
    extract(out / "image_data.json.zip", out, {"image_data.json"})

    missing = [n for n in ("val_sceneGraphs.json", "image_data.json") if not (out / n).exists()]
    if missing:
        sys.exit(f"missing after extraction: {missing} (check the archive listing above)")
    print("ok")


if __name__ == "__main__":
    main()
