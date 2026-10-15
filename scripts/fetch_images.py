"""Download only the Visual Genome images referenced by a FaultLine JSONL.

Usage: python scripts/fetch_images.py --dataset data/faultline_v0.jsonl --out data/images
Re-running skips images already on disk; failures are listed in <out>/failed.txt.
"""

import argparse
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faultline.schema import read_jsonl  # noqa: E402


def get(url, dest, tries=3):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = r.read()
            if len(data) < 1000:
                raise IOError(f"suspiciously small ({len(data)} bytes)")
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.rename(dest)
            return None
        except Exception as e:  # network errors vary by platform
            err = e
            time.sleep(1 + 2 * k)
    return f"{url}\t{err}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/faultline_v0.jsonl")
    ap.add_argument("--out", default="data/images")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = {}
    for r in read_jsonl(args.dataset):
        dest = out / r["image_file"]
        if not dest.exists() and r.get("image_url"):
            jobs[r["image_file"]] = (r["image_url"], dest)
        elif not r.get("image_url"):
            print(f"no url for {r['image_id']}")
    print(f"{len(jobs)} images to fetch")

    failed = []
    with ThreadPoolExecutor(args.workers) as pool:
        futs = [pool.submit(get, url, dest) for url, dest in jobs.values()]
        for i, f in enumerate(as_completed(futs), 1):
            err = f.result()
            if err:
                failed.append(err)
            if i % 100 == 0 or i == len(futs):
                print(f"{i}/{len(futs)} done, {len(failed)} failed")
    if failed:
        (out / "failed.txt").write_text("\n".join(failed) + "\n", encoding="utf-8")
        print(f"{len(failed)} failures written to {out / 'failed.txt'}")


if __name__ == "__main__":
    main()
