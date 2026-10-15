"""Run every probe on every record and append raw outputs to a JSONL file.

Each line is one record for one model. Lines are written as soon as a record is
finished, so a crash loses at most the record in progress. Records already in the
output file without an error are skipped, which is how a stopped run is resumed;
records that failed are retried.
"""

import json
import os
import time
from pathlib import Path

from .prompts import LETTERS, PROMPT_VERSION, probe_turns, verbalize_turns


def done_ids(path):
    ids = set()
    if not os.path.exists(path):
        return ids
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a half-written last line from a killed run; the record is redone
            if row.get("error") is None:
                ids.add(row["id"])
    return ids


def run_record(backend, rec, image):
    out = {}
    for name, kind, turns, max_new in probe_turns(rec, image):
        t0 = time.perf_counter()
        if kind == "gen":
            res = backend.generate(turns, max_new)
        else:
            res = backend.letter_probs(turns)
        res["sec"] = round(time.perf_counter() - t0, 3)
        out[name] = res

    attr = out["attribute"].get("output", "")
    t0 = time.perf_counter()
    res = backend.letter_probs(verbalize_turns(rec, image, attr))
    res["sec"] = round(time.perf_counter() - t0, 3)
    out["verbalize"] = res

    for res in out.values():
        if "probs" in res:
            res["choice"] = max(LETTERS, key=lambda L: res["probs"][L])
    return out


def select(records, subset=None, shard=None, limit=None):
    rows = [r for r in records if r["split"] == "main"]
    if subset == "core":
        rows = [r for r in rows if r["core"]]
    if shard:
        k, n = shard
        rows = [r for i, r in enumerate(rows) if i % n == k]
    if limit:
        rows = rows[:limit]
    return rows


def run(backend, records, image_dir, out_path, model_key, max_side, load_image, log_every=10):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    skip = done_ids(out_path)
    todo = [r for r in records if r["id"] not in skip]
    print(f"{len(records)} selected, {len(skip & {r['id'] for r in records})} already done, {len(todo)} to run")

    # a killed run can leave a last line without a newline; start the next record on a fresh line
    if out_path.exists() and out_path.stat().st_size > 0:
        with open(out_path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            needs_nl = f.read(1) != b"\n"
        if needs_nl:
            with open(out_path, "a") as f:
                f.write("\n")

    t_start = time.perf_counter()
    with open(out_path, "a") as f:
        for i, rec in enumerate(todo, 1):
            image = load_image(Path(image_dir) / rec["image_file"], max_side)
            try:
                probes = run_record(backend, rec, image)
                err = None
            except Exception as e:  # keep going; failed records are retried on the next run
                probes, err = None, f"{type(e).__name__}: {e}"
                if hasattr(backend, "cleanup"):
                    backend.cleanup()
            row = {"id": rec["id"], "model": model_key, "prompt_version": PROMPT_VERSION,
                   "probes": probes, "error": err}
            f.write(json.dumps(row) + "\n")
            f.flush()
            if err:
                print(f"[{rec['id']}] {err}")
            if i % log_every == 0 or i == len(todo):
                el = time.perf_counter() - t_start
                eta = el / i * (len(todo) - i)
                print(f"{i}/{len(todo)}  {el / i:.1f}s/record  eta {eta / 3600:.1f}h", flush=True)
