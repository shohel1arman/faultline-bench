"""Turn scene graphs into typed, graded conflict samples and clean controls.

Pipeline: extract every valid candidate edit per image -> assign candidates to
(type, severity) cells under quotas and a per-image cap -> render text, question
and options -> attach ids and the core-subset flag.
"""

import random
from collections import defaultdict

from . import vocab
from .scene import parse_scene, spatial_relation

TYPES = ["object", "attribute", "relation"]
SEVERITIES = ["S1", "S2", "S3"]


# ---------------------------------------------------------------- candidates

def _anchors_for(scene, t, cfg):
    rcfg = cfg["relation"]
    out = []
    for a in scene.objects:
        if a.oid == t.oid or a.name == t.name:
            continue
        if a.rel_area < cfg["target"]["anchor_min_area"] or not scene.unique_name(a):
            continue
        if a.canon is None and a.name not in vocab.EXTRA_ANCHORS:
            continue
        if a.canon is not None and a.canon == t.canon:
            continue
        r = spatial_relation(t, a, scene.width, scene.height, rcfg["min_gap"], rcfg["min_overlap"])
        if r:
            out.append((a, r[0], r[1]))
    return out


def _only_target_has_relation(scene, t, a, rel, cfg):
    """No other in-vocabulary object stands in the same relation to the anchor."""
    for o in scene.objects:
        if o.oid in (t.oid, a.oid) or o.canon is None:
            continue
        r = spatial_relation(o, a, scene.width, scene.height, 0.0, cfg["relation"]["min_overlap"])
        if r and r[0] == rel:
            return False
    return True


def _filler(scene, used_ids, rng):
    pool = [o for o in scene.objects
            if o.oid not in used_ids and o.canon and scene.unique_canon(o) and len(o.colors) == 1]
    return rng.choice(pool) if pool else None


def _relation_severity(t, a, gap, cfg):
    r = cfg["relation"]
    if gap >= r["salient_gap"] and min(t.rel_area, a.rel_area) >= r["salient_min_area"]:
        return "S1"
    if gap < r["subtle_gap"]:
        return "S3"
    return "S2"


def image_candidates(image_id, graph, cfg, rng):
    scene = parse_scene(image_id, graph)
    if not scene.width or not scene.height:
        return []
    min_area = cfg["target"]["min_area"]
    salient = cfg["severity"]["salient_area"]
    cands = []

    for t in scene.objects:
        if t.canon is None or not scene.unique_canon(t) or t.rel_area < min_area:
            continue
        anchors = _anchors_for(scene, t, cfg)
        anchor = rng.choice(anchors) if anchors else None
        color = t.colors[0] if len(t.colors) == 1 else None
        base = {"image_id": scene.image_id, "scene": scene, "target": t,
                "anchor": anchor[0] if anchor else None,
                "rel": anchor[1] if anchor else None,
                "gap": anchor[2] if anchor else None,
                "color": color}

        # attribute: exactly one colour on the target
        if color:
            far = vocab.far_colors(color)
            if far:
                sev = "S1" if t.rel_area >= salient else "S2"
                cands.append({**base, "type": "attribute", "severity": sev, "edit": "far",
                              "orig": color, "new": rng.choice(far)})
            near = [c for c in vocab.NEAR_COLORS[color] if c not in t.colors]
            if near:
                cands.append({**base, "type": "attribute", "severity": "S3", "edit": "near",
                              "orig": color, "new": rng.choice(near)})

        if anchor is None:
            continue
        a, rel, gap = anchor

        # relation: geometrically verified, replaced by its antonym
        sev = _relation_severity(t, a, gap, cfg)
        cands.append({**base, "type": "relation", "severity": sev, "edit": "antonym",
                      "orig": rel, "new": vocab.ANTONYM[rel]})

        # object: swap the target noun for one that is absent from the image
        if not _only_target_has_relation(scene, t, a, rel, cfg):
            continue
        group = vocab.GROUP_OF[t.canon]
        absent = [n for n in vocab.OBJECT_NAMES if n != t.canon and scene.is_absent(n)
                  and n != a.name and n != (a.canon or "")]
        far = [n for n in absent if vocab.GROUP_OF[n] != group]
        near = [n for n in absent if vocab.GROUP_OF[n] == group]
        if far:
            sev = "S1" if t.rel_area >= salient else "S2"
            cands.append({**base, "type": "object", "severity": sev, "edit": "far",
                          "orig": t.canon, "new": rng.choice(far)})
        if near:
            cands.append({**base, "type": "object", "severity": "S3", "edit": "near",
                          "orig": t.canon, "new": rng.choice(near)})
    return cands


# ---------------------------------------------------------------- rendering

def _target_sentence(c, value):
    """Return (sentence, start, end) where [start:end] is the editable slot filled with value."""
    t, a, rel = c["target"], c["anchor"], c["rel"]
    kind = c["type"]
    if kind == "attribute":
        if a is None:
            pre = f"The {t.canon} is "
            return pre + value + ".", len(pre), len(pre) + len(value)
        pre = "The "
        s = f"{pre}{value} {t.canon} is {rel} the {a.name}."
        return s, len(pre), len(pre) + len(value)
    if kind == "object":
        pre = "The "
        s = f"{pre}{value} is {rel} the {a.name}."
        return s, len(pre), len(pre) + len(value)
    if kind == "relation":
        subject = f"The {c['color']} {t.canon}" if c["color"] else f"The {t.canon}"
        pre = f"{subject} is "
        s = f"{pre}{value} the {a.name}."
        return s, len(pre), len(pre) + len(value)
    raise ValueError(kind)


def _question(c):
    t, a = c["target"], c["anchor"]
    if c["type"] == "attribute":
        return f"What color is the {t.canon}?"
    if c["type"] == "object":
        return f"What is {c['rel']} the {a.name}?"
    return f"Where is the {t.canon} relative to the {a.name}?"


def _options(c, rng):
    true, new = c["orig"], c["new"]
    if c["type"] == "relation":
        opts = list(vocab.RELATIONS)
    elif c["type"] == "attribute":
        blocked = {true, new, *c["target"].colors, *vocab.NEAR_COLORS[true]}
        pool = [x for x in vocab.FAR_COLOR_POOL + vocab.COLORS if x not in blocked]
        pool = list(dict.fromkeys(pool))
        opts = [true, new] + rng.sample(pool, 2)
    else:
        scene = c["scene"]
        group_t = vocab.GROUP_OF[true]
        group_n = vocab.GROUP_OF[new]
        absent = [n for n in vocab.OBJECT_NAMES if n not in (true, new) and scene.is_absent(n)]
        picks = []
        for g in (group_t, group_n):
            same = [n for n in absent if vocab.GROUP_OF[n] == g and n not in picks]
            if same:
                picks.append(rng.choice(same))
        rest = [n for n in absent if n not in picks]
        while len(picks) < 2:
            picks.append(rng.choice(rest))
            rest.remove(picks[-1])
        opts = [true, new] + picks[:2]
    rng.shuffle(opts)
    return opts


def render(c, conflict, rng):
    value = c["new"] if conflict else c["orig"]
    sent, s0, s1 = _target_sentence(c, value)
    clean_sent, c0, c1 = _target_sentence(c, c["orig"])

    used = {c["target"].oid} | ({c["anchor"].oid} if c["anchor"] else set())
    filler = _filler(c["scene"], used, rng)
    filler_sent = f"The {filler.canon} is {filler.colors[0]}." if filler else None
    filler_first = bool(filler) and rng.random() < 0.5

    if filler_first:
        offset = len(filler_sent) + 1
        text = f"{filler_sent} {sent}"
        text_clean = f"{filler_sent} {clean_sent}"
    else:
        offset = 0
        text = sent + (f" {filler_sent}" if filler else "")
        text_clean = clean_sent + (f" {filler_sent}" if filler else "")

    t, a = c["target"], c["anchor"]
    scene = c["scene"]
    rec = {
        "image_id": c["image_id"],
        "is_conflict": conflict,
        "conflict_type": c["type"] if conflict else "none",
        "question_type": c["type"],
        "severity": c["severity"] if conflict else None,
        "edit_kind": c["edit"] if conflict else None,
        "text": text,
        "text_clean": text_clean,
        "edited_span": value if conflict else None,
        "edited_span_char": [offset + s0, offset + s1] if conflict else None,
        "original_span": c["orig"] if conflict else None,
        "original_span_char": [offset + c0, offset + c1] if conflict else None,
        "question": _question(c),
        "answer_image": c["orig"],
        "answer_text": c["new"] if conflict else c["orig"],
        "foil": c["new"],
        "mcq_options": _options(c, rng),
        "target": {"object_id": t.oid, "name": t.canon, "bbox": t.bbox(),
                   "rel_area": round(t.rel_area, 4)},
        "anchor": ({"object_id": a.oid, "name": a.name, "bbox": a.bbox(),
                    "rel_area": round(a.rel_area, 4)} if a else None),
        "relation": c["rel"],
        "relation_gap": round(c["gap"], 4) if c["gap"] is not None else None,
        "width": scene.width,
        "height": scene.height,
    }
    return rec


# ---------------------------------------------------------------- assignment

def _cell(c):
    return (c["type"], c["severity"])


def assign(cands, cfg, rng):
    """Greedy fill of every cell under the per-image cap, scarcest cells first."""
    q = cfg["quota"]
    need_conf = q["per_cell"] + q["reserve_per_cell"]
    n_ctrl_types = len(TYPES)
    need_ctrl = (q["controls"] + q["reserve_controls"]) // n_ctrl_types

    by_cell = defaultdict(list)
    for c in cands:
        by_cell[_cell(c)].append(c)
    for lst in by_cell.values():
        rng.shuffle(lst)

    per_image = defaultdict(int)
    used_targets = set()
    picked = defaultdict(list)

    def take(c):
        key = (c["image_id"], c["target"].oid)
        if per_image[c["image_id"]] >= q["max_per_image"] or key in used_targets:
            return False
        per_image[c["image_id"]] += 1
        used_targets.add(key)
        return True

    cells = [(t, s) for t in TYPES for s in SEVERITIES]
    cells.sort(key=lambda k: len(by_cell.get(k, [])) / need_conf)
    for cell in cells:
        for c in by_cell.get(cell, []):
            if len(picked[cell]) >= need_conf:
                break
            if take(c):
                picked[cell].append(c)

    # controls reuse the candidate machinery but render the faithful text
    for t in TYPES:
        pool = [c for s in SEVERITIES for c in by_cell.get((t, s), [])]
        rng.shuffle(pool)
        for c in pool:
            if len(picked[(t, "control")]) >= need_ctrl:
                break
            if take(c):
                picked[(t, "control")].append(c)
    return picked


def build(scene_graphs, cfg, image_meta=None, limit_images=None):
    rng = random.Random(cfg["seed"])
    q = cfg["quota"]
    ids = sorted(scene_graphs)
    if limit_images:
        ids = ids[:limit_images]

    cands = []
    for iid in ids:
        cands.extend(image_candidates(iid, scene_graphs[iid], cfg, rng))
    supply = defaultdict(int)
    for c in cands:
        supply[_cell(c)] += 1

    picked = assign(cands, cfg, rng)

    records = []
    per_ctrl_main = q["controls"] // len(TYPES)
    per_ctrl_core = q["core_controls"] // len(TYPES)
    for cell, lst in sorted(picked.items()):
        is_ctrl = cell[1] == "control"
        n_main = per_ctrl_main if is_ctrl else q["per_cell"]
        n_core = per_ctrl_core if is_ctrl else q["core_per_cell"]
        for i, c in enumerate(lst):
            rec = render(c, conflict=not is_ctrl, rng=rng)
            rec["split"] = "main" if i < n_main else "reserve"
            rec["core"] = i < n_core
            meta = (image_meta or {}).get(c["image_id"], {})
            rec["coco_id"] = meta.get("coco_id")
            rec["image_url"] = meta.get("url")
            rec["image_file"] = f"{c['image_id']}.jpg"
            records.append(rec)

    rng.shuffle(records)
    for n, rec in enumerate(records, 1):
        rec["id"] = f"fl_{n:05d}"
    records.sort(key=lambda r: r["id"])
    return records, dict(supply)
