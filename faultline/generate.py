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

def stated_color(obj):
    """The object's colour if it can be stated reliably, else None.

    Requires exactly one colour label. Animals are excluded: coats are often patterned
    (a cow labelled only "white" may be black and white), so a single scene-graph colour
    can be false for the image even before any edit.
    """
    if len(obj.colors) != 1 or obj.canon is None or vocab.GROUP_OF[obj.canon] == "animal":
        return None
    return obj.colors[0]


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
            if o.oid not in used_ids and o.canon and scene.unique_canon(o) and stated_color(o)]
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
        color = stated_color(t)
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
    filler_sent = f"The {filler.canon} is {stated_color(filler)}." if filler else None
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


def _target(c):
    return (c["image_id"], c["target"].oid)


def assign(cands, cfg, rng):
    """Allocate target objects to cells: each object at most once, at most
    max_per_image objects per image, and no cell above its quota.

    One object can usually serve several cells (an object with a valid anchor yields
    object, relation and attribute candidates, and any candidate can be a control of its
    type), so cells compete for the same objects. Cells grow one sample at a time, always
    the least-filled cell first. A cell takes a free object if one exists; otherwise an
    augmenting path moves already-assigned objects between cells to make room, so earlier
    choices never block later cells. A cell is closed only when no such path exists, in
    which case it cannot grow without shrinking another cell. When all quotas fit, all
    cells are filled; when they do not, the shortfall is spread so that the least-filled
    cell is as full as possible. Main quotas (controls included) are filled before
    reserves, and the main/reserve split within a cell is random.

    Returns {cell: (main_list, reserve_list)}.
    """
    q = cfg["quota"]
    cap = q["max_per_image"]
    main_need = {(t, s): q["per_cell"] for t in TYPES for s in SEVERITIES}
    full_need = {(t, s): q["per_cell"] + q["reserve_per_cell"] for t in TYPES for s in SEVERITIES}
    for t in TYPES:
        main_need[(t, "control")] = q["controls"] // len(TYPES)
        full_need[(t, "control")] = (q["controls"] + q["reserve_controls"]) // len(TYPES)
    cells = list(full_need)

    # one candidate per (cell, object); a control uses the first candidate of its type
    cand = {}
    for c in cands:
        for cell in (_cell(c), (c["type"], "control")):
            if cell in full_need:
                cand.setdefault((cell, _target(c)), c)
    eligible = defaultdict(list)
    n_cells_of = defaultdict(int)
    in_image = defaultdict(list)
    for cell, t in cand:
        eligible[cell].append(t)
        n_cells_of[t] += 1
    for t in n_cells_of:
        in_image[t[0]].append(t)
    for cell in cells:
        rng.shuffle(eligible[cell])

    where = {}                                  # object -> cell
    members = {cell: {} for cell in cells}      # cell -> ordered set of objects
    img_count = defaultdict(int)

    # objects ordered least flexible last, so pop() prefers objects with fewer other uses
    free = {cell: sorted(eligible[cell], key=lambda t: n_cells_of[t], reverse=True) for cell in cells}

    def free_target(cell):
        lst = free[cell]
        while lst:
            t = lst[-1]
            if t not in where and img_count[t[0]] < cap:
                return t
            lst.pop()                            # never usable again: assigned or image full
        return None

    def put(t, cell):
        where[t] = cell
        members[cell][t] = None
        img_count[t[0]] += 1

    def take_out(t):
        cell = where.pop(t)
        del members[cell][t]
        img_count[t[0]] -= 1

    def augment(k):
        """Add one object to cell k, rearranging other cells without shrinking them."""
        t = free_target(k)
        if t is not None:
            put(t, k)
            return True
        parent = {k: None}
        touched = set()
        queue = [k]
        while queue:
            x = queue.pop(0)
            if x != k:
                t = free_target(x)
                if t is not None:
                    put(t, x)
                    cur = x
                    while parent[cur] is not None:
                        px, move = parent[cur]
                        if move[0] == "move":
                            take_out(move[1])
                            put(move[1], px)
                        else:                    # swap: a free object in a full image replaces u
                            take_out(move[2])
                            put(move[1], px)
                        cur = px
                    return True
            for t in eligible[x]:
                if t in touched or where.get(t) == x:
                    continue
                y = where.get(t)
                if y is not None:
                    if y not in parent:
                        parent[y] = (x, ("move", t))
                        touched.add(t)
                        queue.append(y)
                elif img_count[t[0]] >= cap:
                    for u in in_image[t[0]]:
                        y = where.get(u)
                        if y is not None and y != x and y not in parent and u not in touched:
                            parent[y] = (x, ("swap", t, u))
                            touched.update((t, u))
                            queue.append(y)
                            break
        return False

    closed = set()
    for need in (main_need, full_need):
        while True:
            open_cells = [k for k in cells if k not in closed and len(members[k]) < need[k]]
            if not open_cells:
                break
            k = min(open_cells, key=lambda c: (len(members[c]) / need[c], c))
            if not augment(k):
                closed.add(k)

    out = {}
    for cell in cells:
        chosen = [cand[(cell, t)] for t in sorted(members[cell])]
        rng.shuffle(chosen)
        n_main = min(len(chosen), main_need[cell])
        if chosen:
            out[cell] = (chosen[:n_main], chosen[n_main:])
    return out


def target_supply(cands):
    """Distinct target objects available per conflict type, and for object or relation."""
    per_type = defaultdict(set)
    for c in cands:
        per_type[c["type"]].add(_target(c))
    out = {(t, "targets"): len(per_type[t]) for t in TYPES}
    out[("object_or_relation", "targets")] = len(per_type["object"] | per_type["relation"])
    return out


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
    supply.update(target_supply(cands))

    picked = assign(cands, cfg, rng)

    records = []
    per_ctrl_core = q["core_controls"] // len(TYPES)
    for cell, (main, reserve) in sorted(picked.items()):
        is_ctrl = cell[1] == "control"
        n_core = per_ctrl_core if is_ctrl else q["core_per_cell"]
        for i, c in enumerate(main + reserve):
            rec = render(c, conflict=not is_ctrl, rng=rng)
            rec["split"] = "main" if i < len(main) else "reserve"
            rec["core"] = i < min(n_core, len(main))
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
