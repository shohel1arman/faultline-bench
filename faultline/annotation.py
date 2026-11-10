"""Human check of the dataset: sample selection and scoring.

Two annotators see the same 500 core records, in a shuffled order, without the cell
labels. For each record they first answer the multiple-choice question from the image
alone (they do not see the description yet), then judge whether the description matches
the image and, if not, how obvious the mismatch is. This measures:

- label validity: do people pick the scene-graph answer from the image?
- conflict validity: do people see the edited description as wrong, and the faithful
  control descriptions as right?
- severity: do people rate S1 edits as more obvious than S3 edits?
- agreement between the two annotators (Cohen's kappa).
"""

import random
from collections import Counter

from .runner import cell_of, select, stratified_order

CANT_TELL = "cant_tell"
OBVIOUS_SCORE = {"obvious": 3, "moderate": 2, "subtle": 1}
SEVERITY_SCORE = {"S1": 3, "S2": 2, "S3": 1}


def sample(records, n_per_cell=50, n_controls=50, seed=0, first=1000, exclude=()):
    """The annotation sample, taken from the first `first` core records (every model has them).

    Within each cell the records keep their stratified order (ids are a seeded shuffle), so
    the sample is the first n of each cell. Controls are split evenly across question types.
    `exclude` drops records an earlier round already showed. Returns records in a seeded
    random presentation order.
    """
    rows = select(records, "core", first=first)
    if not first:
        rows = stratified_order(rows)
    if exclude:
        exclude = set(exclude)
        rows = [r for r in rows if r["id"] not in exclude]
    by_cell = {}
    for r in rows:
        by_cell.setdefault(cell_of(r), []).append(r)
    control_cells = sorted(c for c in by_cell if c[0] == "control")
    quota = {c: n_controls // len(control_cells) for c in control_cells}
    for c in control_cells[: n_controls % len(control_cells)]:
        quota[c] += 1
    picked = []
    for cell in sorted(by_cell):
        n = quota.get(cell, n_per_cell)
        if len(by_cell[cell]) < n:
            raise ValueError(f"cell {cell} has {len(by_cell[cell])} records, needs {n}")
        picked.extend(by_cell[cell][:n])
    random.Random(seed).shuffle(picked)
    return picked


def item_for_page(rec, image_data_uri):
    """What the annotator sees: nothing that reveals the cell, the edit or the answer."""
    return {"id": rec["id"], "image": image_data_uri, "question": rec["question"],
            "options": rec["mcq_options"], "text": rec["text"]}


# ---------------------------------------------------------------- scoring

def judge(rec, ann):
    """Per-record outcomes for one annotator's answer (None where not applicable)."""
    choice = ann.get("choice")
    desc = ann.get("matches")
    out = {
        "answered": choice not in (None, CANT_TELL),
        "picks_image": None, "desc_answered": desc not in (None, CANT_TELL), "desc_correct": None,
        "obvious": OBVIOUS_SCORE.get(ann.get("obvious")),
    }
    if out["answered"]:
        out["picks_image"] = choice == rec["answer_image"]
    if out["desc_answered"]:
        out["desc_correct"] = (desc == "no") if rec["is_conflict"] else (desc == "yes")
    return out


def cohen_kappa(a, b):
    """Cohen's kappa for two equal-length label lists (None where undefined)."""
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)


def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x, y):
    """Spearman rank correlation with average ranks for ties (None if undefined)."""
    if len(x) < 3:
        return None
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    return None if vx == 0 or vy == 0 else cov / (vx * vy) ** 0.5


def _rate(vals):
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)


def summarize(recs_by_id, annotations):
    """Summary for one or more annotators.

    `annotations` maps annotator name -> {record id: answer dict}. Only records every
    annotator answered are used, so the annotators are compared on the same set.
    """
    names = sorted(annotations)
    ids = sorted(set.intersection(*(set(a) for a in annotations.values())) & set(recs_by_id))
    res = {"n_items": len(ids), "annotators": names, "per_annotator": {}, "per_cell": {}}
    judged = {n: {i: judge(recs_by_id[i], annotations[n][i]) for i in ids} for n in names}

    for n in names:
        j = judged[n]
        res["per_annotator"][n] = {
            "cant_tell_question": 1 - _rate([j[i]["answered"] for i in ids])[0],
            "picks_image": _rate([j[i]["picks_image"] for i in ids]),
            "desc_correct_conflict": _rate([j[i]["desc_correct"] for i in ids
                                            if recs_by_id[i]["is_conflict"]]),
            "desc_correct_control": _rate([j[i]["desc_correct"] for i in ids
                                           if not recs_by_id[i]["is_conflict"]]),
        }

    cells = sorted({cell_of(recs_by_id[i]) for i in ids})
    for cell in cells:
        cid = [i for i in ids if cell_of(recs_by_id[i]) == cell]
        res["per_cell"]["/".join(cell)] = {
            "n": len(cid),
            "picks_image": _rate([judged[n][i]["picks_image"] for n in names for i in cid]),
            "desc_correct": _rate([judged[n][i]["desc_correct"] for n in names for i in cid]),
            "obvious_mean": _rate([judged[n][i]["obvious"] for n in names for i in cid]),
        }

    # label problems: every annotator answered, and all agree on something other than the label
    def agreed(field):
        out = []
        for i in ids:
            vals = [annotations[n][i].get(field) for n in names]
            if all(v not in (None, CANT_TELL) for v in vals) and len(set(vals)) == 1:
                out.append((i, vals[0]))
        return out
    wrong_label = [i for i, v in agreed("choice") if v != recs_by_id[i]["answer_image"]]
    conflict_unseen = [i for i, v in agreed("matches") if recs_by_id[i]["is_conflict"] and v == "yes"]
    control_rejected = [i for i, v in agreed("matches") if not recs_by_id[i]["is_conflict"] and v == "no"]
    res["all_agree_other_answer"] = wrong_label
    res["all_agree_conflict_matches"] = conflict_unseen
    res["all_agree_control_wrong"] = control_rejected
    res["valid_rate"] = 1 - len(set(wrong_label) | set(conflict_unseen) | set(control_rejected)) / len(ids) \
        if ids else None

    # severity: higher annotator obviousness for S1 than S3
    sev, obv = [], []
    for n in names:
        for i in ids:
            r = recs_by_id[i]
            if r["is_conflict"] and judged[n][i]["obvious"] is not None:
                sev.append(SEVERITY_SCORE[r["severity"]])
                obv.append(judged[n][i]["obvious"])
    res["severity_spearman"] = (spearman(sev, obv), len(sev))

    if len(names) == 2:
        a, b = names
        res["kappa_choice"] = cohen_kappa([annotations[a][i].get("choice") for i in ids],
                                          [annotations[b][i].get("choice") for i in ids])
        res["kappa_matches"] = cohen_kappa([annotations[a][i].get("matches") for i in ids],
                                           [annotations[b][i].get("matches") for i in ids])
        both = [i for i in ids if recs_by_id[i]["is_conflict"]
                and annotations[a][i].get("obvious") and annotations[b][i].get("obvious")]
        res["kappa_obvious"] = cohen_kappa([annotations[a][i]["obvious"] for i in both],
                                           [annotations[b][i]["obvious"] for i in both])
        res["agree_choice"] = sum(annotations[a][i].get("choice") == annotations[b][i].get("choice")
                                  for i in ids) / len(ids) if ids else None
    return res
