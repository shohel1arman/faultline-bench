"""Loading GQA scene graphs and the geometry used to verify spatial relations."""

import json
from collections import Counter
from dataclasses import dataclass, field

from . import vocab


@dataclass
class Obj:
    oid: str
    name: str            # raw scene-graph name, lowercased
    canon: str | None    # canonical vocabulary name, or None if out of vocabulary
    x: float
    y: float
    w: float
    h: float
    rel_area: float
    colors: list = field(default_factory=list)

    @property
    def x2(self):
        return self.x + self.w

    @property
    def y2(self):
        return self.y + self.h

    def bbox(self):
        return [int(self.x), int(self.y), int(self.w), int(self.h)]


@dataclass
class Scene:
    image_id: str
    width: int
    height: int
    objects: list
    canon_counts: Counter
    name_counts: Counter

    def present_names(self):
        """Every raw and canonical name seen in the image."""
        names = set(self.name_counts)
        names.update(c for c in self.canon_counts)
        return names

    def is_absent(self, canonical):
        present = self.present_names()
        return not (vocab.aliases_of(canonical) & present)

    def unique_canon(self, obj):
        return obj.canon is not None and self.canon_counts[obj.canon] == 1

    def unique_name(self, obj):
        return self.name_counts[obj.name] == 1


def parse_scene(image_id, graph):
    width = int(graph["width"])
    height = int(graph["height"])
    area = float(width * height) if width and height else 0.0
    objects = []
    for oid, o in graph.get("objects", {}).items():
        name = str(o.get("name", "")).strip().lower()
        if not name:
            continue
        w, h = float(o.get("w", 0)), float(o.get("h", 0))
        if w <= 0 or h <= 0 or area == 0:
            continue
        colors = []
        for a in o.get("attributes", []) or []:
            c = vocab.canonical_color(str(a))
            if c and c not in colors:
                colors.append(c)
        objects.append(Obj(
            oid=str(oid),
            name=name,
            canon=vocab.canonical_object(name),
            x=float(o.get("x", 0)),
            y=float(o.get("y", 0)),
            w=w,
            h=h,
            rel_area=(w * h) / area,
            colors=colors,
        ))
    objects.sort(key=lambda o: o.oid)
    canon_counts = Counter(o.canon for o in objects if o.canon)
    name_counts = Counter(o.name for o in objects)
    return Scene(str(image_id), width, height, objects, canon_counts, name_counts)


def load_scene_graphs(path):
    with open(path) as f:
        raw = json.load(f)
    return {str(k): v for k, v in raw.items()}


def load_image_meta(path):
    """Visual Genome image_data.json -> {image_id: record}. Handles v1.0 ('id') and v1.2 ('image_id')."""
    with open(path) as f:
        rows = json.load(f)
    meta = {}
    for r in rows:
        iid = r.get("image_id", r.get("id"))
        if iid is None:
            continue
        meta[str(iid)] = {
            "url": r.get("url"),
            "width": r.get("width"),
            "height": r.get("height"),
            "coco_id": r.get("coco_id"),
        }
    return meta


def _overlap(a1, a2, b1, b2):
    inter = min(a2, b2) - max(a1, b1)
    if inter <= 0:
        return 0.0
    return inter / min(a2 - a1, b2 - b1)


def spatial_relation(t, a, width, height, min_gap, min_overlap):
    """Return (relation of t w.r.t. a, normalised gap) if unambiguous, else None.

    t is "to the left of" a when t ends before a starts horizontally by at least
    min_gap * width and the boxes overlap vertically by at least min_overlap of the
    shorter box. Same idea for above/below. Diagonal or overlapping pairs give None.
    """
    v_overlap = _overlap(t.y, t.y2, a.y, a.y2)
    h_overlap = _overlap(t.x, t.x2, a.x, a.x2)
    found = []
    if v_overlap >= min_overlap:
        gap = (a.x - t.x2) / width
        if gap >= min_gap:
            found.append(("to the left of", gap))
        gap = (t.x - a.x2) / width
        if gap >= min_gap:
            found.append(("to the right of", gap))
    if h_overlap >= min_overlap:
        gap = (a.y - t.y2) / height
        if gap >= min_gap:
            found.append(("above", gap))
        gap = (t.y - a.y2) / height
        if gap >= min_gap:
            found.append(("below", gap))
    if len(found) != 1:
        return None
    return found[0]


def relation_holds(t, a, rel, width, height, min_gap, min_overlap):
    r = spatial_relation(t, a, width, height, min_gap, min_overlap)
    return r is not None and r[0] == rel
