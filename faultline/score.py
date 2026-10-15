"""Per-record outcomes and per-model summary metrics.

Scoring is separate from running so that parser fixes never require rerunning a model.
"""

from collections import defaultdict

from . import parse
from .vocab import SEVERITY_LABEL


def outcome(rec, probes):
    def mcq(name):
        res = probes.get(name, {})
        return parse.mcq_answer(res.get("choice"), rec)

    det = parse.detection(probes["detect"].get("output"))
    att = parse.attribution(probes["attribute"].get("output"))
    o = {
        "id": rec["id"],
        "is_conflict": rec["is_conflict"],
        "type": rec["question_type"],
        "severity": rec["severity"],
        "perceives": mcq("image_only") == "image",
        "reads": mcq("text_only") == ("text" if rec["is_conflict"] else "image"),
        "detect": det,
        "flagged": det == "flagged",
        "mcq": mcq("mcq"),
        "mcq_focus": mcq("mcq_focus"),
        "verbalize": mcq("verbalize"),
        "open": parse.open_answer(probes["open"].get("output"), rec),
        "att_parsed": att["parsed"],
        "att_error": att["error"],
        "mcq_p_image": _p_option(probes.get("mcq"), rec, rec["answer_image"]),
    }
    if rec["is_conflict"]:
        a = {
            "input": att["input"] == "text",
            "type": att["type"] == rec["conflict_type"],
            "span": parse.span_correct(att["span"], rec),
            "obvious": att["obvious"] == SEVERITY_LABEL[rec["severity"]],
        }
        o.update({f"a_{k}": v for k, v in a.items()})
        o["att_score"] = sum(a.values()) / 4
        o["att_strict"] = all(a.values())
    return o


def _p_option(res, rec, option):
    if not res or "probs" not in res:
        return None
    letter = "ABCD"[rec["mcq_options"].index(option)]
    return res["probs"][letter]


def rate(rows, key, cond=None):
    rows = [r for r in rows if cond is None or cond(r)]
    if not rows:
        return None
    return sum(bool(r[key]) if not callable(key) else bool(key(r)) for r in rows) / len(rows)


def summarize(outcomes):
    conf = [o for o in outcomes if o["is_conflict"]]
    ctrl = [o for o in outcomes if not o["is_conflict"]]

    def res(rows, field):
        return rate(rows, lambda r: r[field] == "image")

    def tpr(rows, field):
        t = sum(r[field] == "text" for r in rows)
        i = sum(r[field] == "image" for r in rows)
        return t / (t + i) if t + i else None

    det = rate(conf, "flagged")
    ffr = rate(ctrl, "flagged")
    s = {
        "n_conflict": len(conf),
        "n_control": len(ctrl),
        "perceive": rate(conf, "perceives"),
        "read": rate(conf, "reads"),
        "DetR": det,
        "FFR": ffr,
        "DetR_adj": det - ffr if det is not None and ffr is not None else None,
        "ResA_mcq": res(conf, "mcq"),
        "ResA_open": res(conf, "open"),
        "ResA_focus": res(conf, "mcq_focus"),
        "ResA_verbalize": res(conf, "verbalize"),
        "TPR_mcq": tpr(conf, "mcq"),
        "ResA_mcq_if_perceived": rate(conf, lambda r: r["mcq"] == "image", lambda r: r["perceives"]),
        "AttA": (sum(o["att_score"] for o in conf) / len(conf)) if conf else None,
        "AttA_strict": rate(conf, "att_strict"),
        "control_acc_mcq": res(ctrl, "mcq"),
        "unparsed_detect": rate(outcomes, lambda r: r["detect"] is None),
        "unparsed_attribute": rate(outcomes, lambda r: not r["att_parsed"]),
    }
    r_given_a = rate(conf, lambda r: r["mcq"] == "image", lambda r: r["att_strict"])
    r_given_not_a = rate(conf, lambda r: r["mcq"] == "image", lambda r: not r["att_strict"])
    s["R_given_A"] = r_given_a
    s["R_given_notA"] = r_given_not_a
    s["joint"] = joint(conf)
    return s


def joint(conf):
    """Distribution over (detected, resolved, attributed) on conflict records."""
    counts = defaultdict(int)
    for o in conf:
        key = ("D" if o["flagged"] else "-") + ("R" if o["mcq"] == "image" else "-") + \
              ("A" if o["att_strict"] else "-")
        counts[key] += 1
    n = len(conf) or 1
    return {k: counts[k] / n for k in sorted(counts)}


def by_cell(outcomes):
    """Summaries per (type, severity); controls are attached to every cell of their type."""
    out = {}
    for t in ("object", "attribute", "relation"):
        ctrl = [o for o in outcomes if not o["is_conflict"] and o["type"] == t]
        for sev in ("S1", "S2", "S3"):
            conf = [o for o in outcomes if o["is_conflict"] and o["type"] == t and o["severity"] == sev]
            if conf:
                out[(t, sev)] = summarize(conf + ctrl)
    return out
