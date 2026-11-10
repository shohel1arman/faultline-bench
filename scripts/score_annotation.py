"""Score the human check from the files the annotators exported.

  python scripts/score_annotation.py annotation/results/faultline_check_A1_500.json \\
      annotation/results/faultline_check_A2_500.json

Prints per-annotator accuracy, per-cell results, agreement (Cohen's kappa), the records
where all annotators agree the label is wrong, and the severity correlation; writes
results/human_check.json.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faultline.annotation import summarize  # noqa: E402
from faultline.schema import read_jsonl  # noqa: E402


def complete(a):
    """Same rule as the page: an answer, a description judgement, and obviousness if 'no'."""
    return bool(a.get("choice")) and bool(a.get("matches")) and (a["matches"] != "no" or bool(a.get("obvious")))


def pct(v):
    if isinstance(v, tuple):
        return "  -  " if v[0] is None else f"{100 * v[0]:5.1f} (n={v[1]})"
    return "  -  " if v is None else f"{100 * v:5.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--dataset", default="data/faultline_v0.jsonl")
    ap.add_argument("--out", default="results/human_check.json")
    args = ap.parse_args()

    recs = {r["id"]: r for r in read_jsonl(args.dataset)}
    anns, versions = {}, set()
    for p in args.files:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        done = {k: v for k, v in d["answers"].items() if not k.startswith("_") and complete(v)}
        print(f"{p}: annotator {d['annotator']}, {len(done)} complete answers")
        anns[d["annotator"]] = done
        versions.add(d.get("page_version"))
    if len(versions) > 1:
        sys.exit(f"files come from different page versions: {versions}")

    res = summarize(recs, anns)
    print(f"items answered by every annotator: {res['n_items']}")
    for n, s in res["per_annotator"].items():
        print(f"\n{n}")
        print(f"  picks the labelled image answer      {pct(s['picks_image'])}")
        print(f"  can't tell (question)                {pct(s['cant_tell_question'])}")
        print(f"  says edited description is wrong     {pct(s['desc_correct_conflict'])}")
        print(f"  says faithful description is right   {pct(s['desc_correct_control'])}")
    print("\nper cell (all annotators): picks image answer / description judged correctly / mean obviousness 1-3")
    for c, s in res["per_cell"].items():
        ob = s["obvious_mean"][0]
        print(f"  {c:<20} {pct(s['picks_image']):>14} {pct(s['desc_correct']):>14}   "
              f"{'-' if ob is None else f'{ob:.2f}'}")
    if "kappa_choice" in res:
        def k(v):
            return "-" if v is None else f"{v:.3f}"
        print(f"\nagreement on the answer: {pct(res['agree_choice'])}; Cohen's kappa: answer "
              f"{k(res['kappa_choice'])}, description {k(res['kappa_matches'])}, "
              f"obviousness {k(res['kappa_obvious'])}")
    rho, n = res["severity_spearman"]
    print(f"severity (S1 > S3) vs rated obviousness: Spearman {'-' if rho is None else f'{rho:.3f}'} (n={n})")
    print(f"all annotators pick another answer than the label: {len(res['all_agree_other_answer'])}")
    print(f"all annotators say an edited description matches: {len(res['all_agree_conflict_matches'])}")
    print(f"all annotators say a faithful description is wrong: {len(res['all_agree_control_wrong'])}")
    print(f"records valid by these checks: {pct(res['valid_rate'])}")

    # Anchoring check: on conflicts the description argues for one wrong option. A step-1
    # answer is given before the description is shown, so it should pick that option no
    # more often than the other wrong ones (see docs/ANNOTATION.md, round 1).
    anchor = {}
    print("\nstep-1 answers on conflicts: picks the label / picks the description's answer / other wrong")
    for n, done in anns.items():
        for t in ("object", "attribute", "relation"):
            c = {"label": 0, "description": 0, "other": 0}
            for rid, a in done.items():
                r = recs[rid]
                if not r["is_conflict"] or r["conflict_type"] != t or a["choice"] == "cant_tell":
                    continue
                key = ("label" if a["choice"] == r["answer_image"] else
                       "description" if a["choice"] == r["answer_text"] else "other")
                c[key] += 1
            anchor[f"{n}/{t}"] = c
            tot = sum(c.values()) or 1
            print(f"  {n} {t:<10} {100 * c['label'] / tot:5.1f} {100 * c['description'] / tot:5.1f} "
                  f"{100 * c['other'] / tot:5.1f}  (n={sum(c.values())})")
    res["step1_on_conflicts"] = anchor

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
