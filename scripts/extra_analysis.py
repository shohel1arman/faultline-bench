"""Analyses beyond scripts/analyze.py, from its per-record outcomes.

  python scripts/extra_analysis.py            (reads results/spark/outcomes.jsonl)

Writes results/spark/extra.csv with, per model (95% bootstrap intervals that resample
images, 2,000 replicates):
  arbitration_failure  answers from the text although it perceived the fact and flagged
                       the conflict (P(mcq = text | perceives, flagged))
  report_contradiction its attribution report says "Error: no" although it flagged the
                       conflict in the detection probe
  steerability         answers from the image with "rely on the image" minus without
  s1_minus_s3_detect, s1_minus_s3_resolve   severity effect, per conflict type and pooled
and results/spark/pairs.csv: paired differences between models on the same records
(smaller minus larger), with an exact McNemar test on answering from the image.
"""

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faultline.analysis import mcnemar  # noqa: E402
from faultline.schema import read_jsonl  # noqa: E402

PAIRS = [("qwen3vl-2b", "qwen3vl-4b"), ("qwen3vl-4b", "qwen3vl-8b"), ("qwen3vl-8b", "qwen3vl-32b"),
         ("qwen3vl-8b", "qwen3vl-30b-a3b"), ("qwen3vl-30b-a3b", "qwen3vl-32b"),
         ("qwen25vl-3b", "qwen25vl-7b"), ("qwen25vl-7b", "qwen25vl-32b"), ("internvl3-2b", "internvl3-8b")]


def ratio_stat(num, den):
    """A statistic as (numerator predicate, denominator predicate) over outcome dicts."""
    return num, den


STATS = {
    "arbitration_failure": ratio_stat(lambda o: o["mcq"] == "text",
                                      lambda o: o["is_conflict"] and o["perceives"] and o["flagged"]),
    "report_contradiction": ratio_stat(lambda o: o.get("att_error") == "no",
                                       lambda o: o["is_conflict"] and o["flagged"] and o.get("att_parsed")),
    "resolve": ratio_stat(lambda o: o["mcq"] == "image", lambda o: o["is_conflict"]),
    "focus": ratio_stat(lambda o: o["mcq_focus"] == "image", lambda o: o["is_conflict"]),
}
for t in ("attribute", "object", "relation", None):
    for s in ("S1", "S3"):
        suffix = f"_{t}_{s}" if t else f"_all_{s}"
        STATS["detect" + suffix] = ratio_stat(
            lambda o: o["flagged"],
            lambda o, t=t, s=s: o["is_conflict"] and o["severity"] == s and (t is None or o["type"] == t))
        STATS["resolve" + suffix] = ratio_stat(
            lambda o: o["mcq"] == "image",
            lambda o, t=t, s=s: o["is_conflict"] and o["severity"] == s and (t is None or o["type"] == t))

CONTRASTS = {"steerability": ("focus", "resolve")}
for t in ("attribute", "object", "relation", "all"):
    CONTRASTS[f"s1_minus_s3_detect_{t}"] = (f"detect_{t}_S1", f"detect_{t}_S3")
    CONTRASTS[f"s1_minus_s3_resolve_{t}"] = (f"resolve_{t}_S1", f"resolve_{t}_S3")


def per_image(outcomes, image_of, stats):
    sums = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for o in outcomes:
        img = image_of[o["id"]]
        for name, (num, den) in stats.items():
            if den(o):
                sums[img][name][1] += 1
                sums[img][name][0] += bool(num(o))
    return sums


def value(sums, images, name):
    n = d = 0
    for i in images:
        a, b = sums[i][name]
        n += a
        d += b
    return n / d if d else None


def interval(draws, alpha=0.05):
    d = sorted(x for x in draws if x is not None)
    if not d:
        return None, None
    return d[int(math.floor(alpha / 2 * (len(d) - 1)))], d[int(math.ceil((1 - alpha / 2) * (len(d) - 1)))]


def bootstrap(outcomes, image_of, reps, seed=0):
    sums = per_image(outcomes, image_of, STATS)
    images = sorted(sums)
    point = {n: value(sums, images, n) for n in STATS}
    for c, (a, b) in CONTRASTS.items():
        point[c] = None if point[a] is None or point[b] is None else point[a] - point[b]
    draws = defaultdict(list)
    rng = random.Random(seed)
    for _ in range(reps):
        sample = [images[rng.randrange(len(images))] for _ in images]
        v = {n: value(sums, sample, n) for n in STATS}
        for n in STATS:
            draws[n].append(v[n])
        for c, (a, b) in CONTRASTS.items():
            draws[c].append(None if v[a] is None or v[b] is None else v[a] - v[b])
    return {n: (point[n], *interval(draws[n])) for n in point}


def paired(a_out, b_out, image_of, reps, seed=0):
    """Difference in answering from the image (b minus a) on records both models answered."""
    a = {o["id"]: o for o in a_out if o["is_conflict"]}
    b = {o["id"]: o for o in b_out if o["is_conflict"]}
    ids = sorted(set(a) & set(b))
    sums = defaultdict(lambda: [0, 0, 0])
    for i in ids:
        s = sums[image_of[i]]
        s[0] += a[i]["mcq"] == "image"
        s[1] += b[i]["mcq"] == "image"
        s[2] += 1
    images = sorted(sums)

    def diff(sample):
        x = y = n = 0
        for im in sample:
            p, q, m = sums[im]
            x, y, n = x + p, y + q, n + m
        return (y - x) / n if n else None
    point = diff(images)
    rng = random.Random(seed)
    lo, hi = interval([diff([images[rng.randrange(len(images))] for _ in images]) for _ in range(reps)])
    joined = [{"is_conflict": True, "larger": b[i]["mcq"], "smaller": a[i]["mcq"]} for i in ids]
    b_only, a_only, p = mcnemar(joined, "larger", "smaller")
    return {"n": len(ids), "diff": point, "low": lo, "high": hi, "larger_only": b_only,
            "smaller_only": a_only, "p_value": p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/spark")
    ap.add_argument("--dataset", default="data/faultline_v0.jsonl")
    ap.add_argument("--reps", type=int, default=2000)
    args = ap.parse_args()

    image_of = {r["id"]: r["image_file"] for r in read_jsonl(args.dataset)}
    by_model = defaultdict(list)
    with open(Path(args.results) / "outcomes.jsonl", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            by_model[o["model"]].append(o)

    rows = []
    for m in sorted(by_model):
        res = bootstrap(by_model[m], image_of, args.reps)
        for name in ["arbitration_failure", "report_contradiction", "steerability"] + \
                [c for c in CONTRASTS if c.startswith("s1_minus")]:
            est, lo, hi = res[name]
            rows.append({"model": m, "metric": name, "estimate": est, "ci_low": lo, "ci_high": hi})
        print(f"{m:<17} arbitration failure {100 * res['arbitration_failure'][0]:5.1f}  "
              f"report says no {100 * res['report_contradiction'][0]:5.1f}  "
              f"steerability {100 * res['steerability'][0]:+5.1f}  "
              f"S1-S3 detect {100 * res['s1_minus_s3_detect_all'][0]:+5.1f}  "
              f"resolve {100 * res['s1_minus_s3_resolve_all'][0]:+5.1f}")
    # pooled over all models: severity effects on the same records
    pooled = [dict(o, id=o["model"] + "/" + o["id"]) for os_ in by_model.values() for o in os_]
    pooled_images = {o["id"]: image_of[o["id"].split("/", 1)[1]] for o in pooled}
    res = bootstrap(pooled, pooled_images, args.reps)
    for name in [c for c in CONTRASTS if c.startswith("s1_minus")]:
        est, lo, hi = res[name]
        rows.append({"model": "all models", "metric": name, "estimate": est, "ci_low": lo, "ci_high": hi})
        print(f"all models  {name:<28} {100 * est:+5.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]")
    out = Path(args.results)
    with open(out / "extra.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    prows = []
    for a, b in PAIRS:
        if a in by_model and b in by_model:
            r = paired(by_model[a], by_model[b], image_of, args.reps)
            prows.append({"smaller": a, "larger": b, **r})
            print(f"{a} -> {b}: from image {100 * r['diff']:+5.1f} [{100 * r['low']:+.1f}, {100 * r['high']:+.1f}], "
                  f"McNemar {r['larger_only']} vs {r['smaller_only']}, p = {r['p_value']:.2g}")
    with open(out / "pairs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(prows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(prows)
    print(f"wrote {out}/extra.csv and {out}/pairs.csv")


if __name__ == "__main__":
    main()
