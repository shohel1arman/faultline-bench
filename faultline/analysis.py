"""Paper statistics: metrics with image-clustered bootstrap intervals and paired tests.

Every metric is a ratio of sums over records (numerator / denominator), so it can be
recomputed on a bootstrap sample from per-image sums. Records from the same image are
resampled together, because an image can contribute up to two records.
Standard library only.
"""

import math
import random
from collections import defaultdict

def _has_controls(o):
    return o["is_conflict"] and o.get("delay") is not None and o.get("check_silent") is not None


# name -> (numerator(o), denominator(o)); o is a per-record outcome from score.outcome
METRICS = {
    "perceive": (lambda o: o["perceives"], lambda o: o["is_conflict"]),
    "read": (lambda o: o["reads"], lambda o: o["is_conflict"]),
    "DetR": (lambda o: o["flagged"], lambda o: o["is_conflict"]),
    "FFR": (lambda o: o["flagged"], lambda o: not o["is_conflict"]),
    "ResA_mcq": (lambda o: o["mcq"] == "image", lambda o: o["is_conflict"]),
    "ResA_open": (lambda o: o["open"] == "image", lambda o: o["is_conflict"]),
    "ResA_focus": (lambda o: o["mcq_focus"] == "image", lambda o: o["is_conflict"]),
    "ResA_verbalize": (lambda o: o["verbalize"] == "image", lambda o: o["is_conflict"]),
    "TextA_mcq": (lambda o: o["mcq"] == "text", lambda o: o["is_conflict"]),
    "TPR_mcq": (lambda o: o["mcq"] == "text",
                lambda o: o["is_conflict"] and o["mcq"] in ("text", "image")),
    "ResA_if_perceived": (lambda o: o["mcq"] == "image",
                          lambda o: o["is_conflict"] and o["perceives"]),
    "AttA": (lambda o: o.get("att_score", 0.0), lambda o: o["is_conflict"]),
    "AttA_strict": (lambda o: o.get("att_strict", False), lambda o: o["is_conflict"]),
    "A_input": (lambda o: o.get("a_input", False), lambda o: o["is_conflict"]),
    "A_type": (lambda o: o.get("a_type", False), lambda o: o["is_conflict"]),
    "A_span": (lambda o: o.get("a_span", False), lambda o: o["is_conflict"]),
    "A_obvious": (lambda o: o.get("a_obvious", False), lambda o: o["is_conflict"]),
    "R_given_A": (lambda o: o["mcq"] == "image",
                  lambda o: o["is_conflict"] and o.get("att_strict", False)),
    "R_given_notA": (lambda o: o["mcq"] == "image",
                     lambda o: o["is_conflict"] and not o.get("att_strict", False)),
    "R_given_D": (lambda o: o["mcq"] == "image", lambda o: o["is_conflict"] and o["flagged"]),
    "R_given_notD": (lambda o: o["mcq"] == "image",
                     lambda o: o["is_conflict"] and not o["flagged"]),
    "R_given_S": (lambda o: o["mcq"] == "image",
                  lambda o: o["is_conflict"] and o.get("a_span", False)),
    "R_given_notS": (lambda o: o["mcq"] == "image",
                     lambda o: o["is_conflict"] and not o.get("a_span", False)),
    # verbalize decomposition, on conflicts that have the two control probes
    "ResA_mcq_c": (lambda o: o["mcq"] == "image", lambda o: _has_controls(o)),
    "ResA_delay": (lambda o: o["delay"] == "image", lambda o: _has_controls(o)),
    "ResA_check": (lambda o: o["check_silent"] == "image", lambda o: _has_controls(o)),
    "ResA_verbalize_c": (lambda o: o["verbalize"] == "image", lambda o: _has_controls(o)),
    "control_acc_mcq": (lambda o: o["mcq"] == "image", lambda o: not o["is_conflict"]),
    "control_foil_mcq": (lambda o: o["mcq"] == "foil", lambda o: not o["is_conflict"]),
    "unparsed_detect": (lambda o: o["detect"] is None, lambda o: True),
    "unparsed_attribute": (lambda o: not o["att_parsed"], lambda o: True),
}

# name -> (metric_a, metric_b): reported as a - b
CONTRASTS = {
    "DetR_adj": ("DetR", "FFR"),
    "verbalize_gain": ("ResA_verbalize", "ResA_mcq"),
    "focus_gain": ("ResA_focus", "ResA_mcq"),
    "coupling_A": ("R_given_A", "R_given_notA"),
    "coupling_D": ("R_given_D", "R_given_notD"),
    "coupling_S": ("R_given_S", "R_given_notS"),
    "structure_gain": ("ResA_delay", "ResA_mcq_c"),        # question in a later turn
    "check_gain": ("ResA_check", "ResA_delay"),            # being asked to check for errors
    "report_gain": ("ResA_verbalize_c", "ResA_check"),     # the model's own error report
    "detect_minus_resolve": ("DetR", "ResA_mcq"),
}


def _sums(outcomes, image_of, names):
    """Per-image (numerator, denominator) sums for each metric."""
    per = defaultdict(lambda: {n: [0.0, 0.0] for n in names})
    for o in outcomes:
        cell = per[image_of[o["id"]]]
        for n in names:
            num, den = METRICS[n]
            if den(o):
                cell[n][0] += float(num(o))
                cell[n][1] += 1.0
    return per


def _ratio(sums, keys, name):
    num = sum(sums[k][name][0] for k in keys)
    den = sum(sums[k][name][1] for k in keys)
    return num / den if den else None


def bootstrap(outcomes, image_of, reps=2000, seed=0, alpha=0.05):
    """{metric: (estimate, low, high, n)} for METRICS and CONTRASTS, clustered by image."""
    names = list(METRICS)
    sums = _sums(outcomes, image_of, names)
    images = sorted(sums)
    point = {n: _ratio(sums, images, n) for n in names}
    for c, (a, b) in CONTRASTS.items():
        point[c] = point[a] - point[b] if point[a] is not None and point[b] is not None else None

    rng = random.Random(seed)
    draws = defaultdict(list)
    for _ in range(reps):
        sample = [images[rng.randrange(len(images))] for _ in images]
        vals = {n: _ratio(sums, sample, n) for n in names}
        for n, v in vals.items():
            if v is not None:
                draws[n].append(v)
        for c, (a, b) in CONTRASTS.items():
            if vals[a] is not None and vals[b] is not None:
                draws[c].append(vals[a] - vals[b])

    out = {}
    for n in names + list(CONTRASTS):
        d = sorted(draws[n])
        if point[n] is None or not d:
            out[n] = (point[n], None, None, 0)
            continue
        lo = d[int(math.floor(alpha / 2 * (len(d) - 1)))]
        hi = d[int(math.ceil((1 - alpha / 2) * (len(d) - 1)))]
        den = METRICS[n][1] if n in METRICS else None
        n_den = sum(1 for o in outcomes if den(o)) if den else None
        out[n] = (point[n], lo, hi, n_den)
    return out


def mcnemar(outcomes, a, b):
    """Exact two-sided McNemar test on conflicts: does probe a resolve differently from b?

    Returns (n_a_only, n_b_only, p_value), where *_only counts records resolved to the
    image by one probe and not the other.
    """
    conf = [o for o in outcomes if o["is_conflict"] and o.get(a) is not None and o.get(b) is not None]
    a_only = sum(1 for o in conf if o[a] == "image" and o[b] != "image")
    b_only = sum(1 for o in conf if o[b] == "image" and o[a] != "image")
    n = a_only + b_only
    if n == 0:
        return a_only, b_only, 1.0
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return a_only, b_only, min(1.0, 2 * tail)


def joint(outcomes):
    """Counts and shares of (detected, resolved, attributed) patterns on conflicts."""
    conf = [o for o in outcomes if o["is_conflict"]]
    counts = defaultdict(int)
    for o in conf:
        key = ("D" if o["flagged"] else "-") + ("R" if o["mcq"] == "image" else "-") + \
              ("A" if o.get("att_strict") else "-")
        counts[key] += 1
    n = len(conf) or 1
    order = ["DRA", "DR-", "D-A", "D--", "-RA", "-R-", "--A", "---"]
    return {k: (counts[k], counts[k] / n) for k in order}


def common_ids(outcomes_by_model):
    """Records that every model has an outcome for."""
    sets = [set(o["id"] for o in outs) for outs in outcomes_by_model.values()]
    return set.intersection(*sets) if sets else set()
