"""Download only the images referenced by a FaultLine JSONL.

Default source: the GQA images.zip (the Visual Genome images, named by image id). Only
the zip index and the needed members are fetched, with HTTP range requests, so about
0.5 GB is transferred instead of the whole archive. `--source vg` uses the per-image
Visual Genome URLs instead.

Usage: python scripts/fetch_images.py --dataset data/faultline_v0.jsonl --out data/images
Re-running skips images already on disk; failures are listed in <out>/failed.txt.
"""

import argparse
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faultline.remote_zip import http_fetcher, http_size, index, read_member  # noqa: E402
from faultline.schema import read_jsonl  # noqa: E402

GQA_IMAGES_ZIP = "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip"


def save(dest, data):
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(dest)


def with_retries(fn, tries=4):
    for k in range(tries):
        try:
            return fn(), None
        except Exception as e:  # network errors vary by platform
            err = e
            time.sleep(1 + 2 * k)
    return None, err


def from_zip(names, out, workers, url):
    fetch = http_fetcher(url)
    size = http_size(url)
    print(f"{url}: {size / 1e9:.1f} GB; reading the index")
    infos = {PurePosixPath(i.filename).name: i for i in index(fetch, size) if not i.is_dir()}
    print(f"index has {len(infos)} files")
    missing = [n for n in names if n not in infos]
    jobs = [n for n in names if n in infos]

    def one(name):
        data, err = with_retries(lambda: read_member(fetch, infos[name]))
        if err:
            return f"{name}\t{err}"
        save(out / name, data)
        return None

    return run_jobs(jobs, one, workers), [f"{n}\tnot in archive" for n in missing]


def from_vg(records, names, out, workers):
    urls = {r["image_file"]: r.get("image_url") for r in records}

    def one(name):
        url = urls.get(name)
        if not url:
            return f"{name}\tno url"

        def get():
            with urllib.request.urlopen(url, timeout=30) as r:
                data = r.read()
            if len(data) < 1000:
                raise IOError(f"suspiciously small ({len(data)} bytes)")
            return data
        data, err = with_retries(get, tries=3)
        if err:
            return f"{name}\t{url}\t{err}"
        save(out / name, data)
        return None

    return run_jobs(names, one, workers), []


def run_jobs(jobs, one, workers):
    failed = []
    print(f"{len(jobs)} images to fetch")
    with ThreadPoolExecutor(workers) as pool:
        futs = [pool.submit(one, n) for n in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            err = f.result()
            if err:
                failed.append(err)
                if len(failed) <= 3:
                    print(f"failed: {err}")
            if i % 100 == 0 or i == len(futs):
                print(f"{i}/{len(futs)} done, {len(failed)} failed", flush=True)
            if i == 50 and len(failed) == 50:
                print("the first 50 downloads all failed; stopping")
                pool.shutdown(cancel_futures=True)
                break
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/faultline_v0.jsonl")
    ap.add_argument("--out", default="data/images")
    ap.add_argument("--source", choices=["gqa-zip", "vg"], default="gqa-zip")
    ap.add_argument("--zip-url", default=GQA_IMAGES_ZIP)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(args.dataset)
    names = sorted({r["image_file"] for r in records if not (out / r["image_file"]).exists()})
    print(f"{len(names)} images not on disk yet")
    if not names:
        return

    if args.source == "gqa-zip":
        failed, missing = from_zip(names, out, args.workers, args.zip_url)
    else:
        failed, missing = from_vg(records, names, out, args.workers)
    problems = failed + missing
    if problems:
        (out / "failed.txt").write_text("\n".join(problems) + "\n", encoding="utf-8")
        print(f"{len(problems)} problems written to {out / 'failed.txt'}")
    else:
        print("all images present")


if __name__ == "__main__":
    main()
