"""Turn raw model outputs into discrete labels. Pure functions, no model needed."""

import re

from . import vocab

_WORD = re.compile(r"[a-z]+")


def words(s):
    return _WORD.findall((s or "").lower())


# ---------------------------------------------------------------- detection

NEG_PHRASES = ("not accurate", "inaccurate", "incorrect", "does not", "doesn't", "not correct",
               "is wrong", "mismatch")
POS_PHRASES = ("accurately describes", "is accurate", "is correct", "matches")


def detection(output):
    """'flagged' if the model says the description is wrong, 'accepted' if right, else None."""
    head = words(output)[:6]
    for w in head:
        if w == "no":
            return "flagged"
        if w == "yes":
            return "accepted"
    low = (output or "").lower()
    if any(p in low for p in NEG_PHRASES):
        return "flagged"
    if any(p in low for p in POS_PHRASES):
        return "accepted"
    return None


# ---------------------------------------------------------------- answers

REL_KEYWORDS = [
    ("to the left of", ("left",)),
    ("to the right of", ("right",)),
    ("above", ("above", "over", "top")),
    ("below", ("below", "under", "beneath", "underneath", "bottom")),
]


def relation_of(s):
    ws = set(words(s))
    hits = [rel for rel, keys in REL_KEYWORDS if ws & set(keys)]
    return hits[0] if len(hits) == 1 else None


def colors_in(s):
    found = []
    for w in words(s):
        c = vocab.canonical_color(w)
        if c and c not in found:
            found.append(c)
    return found


def objects_in(s):
    text = " " + " ".join(words(s)) + " "
    found = []
    for name in vocab.OBJECT_NAMES:
        for alias in vocab.aliases_of(name):
            for form in (alias, alias + "s", alias + "es"):
                if f" {form} " in text:
                    if name not in found:
                        found.append(name)
    return found


def mentioned(s, qtype):
    if qtype == "attribute":
        return colors_in(s)
    if qtype == "object":
        return objects_in(s)
    rel = relation_of(s)
    return [rel] if rel else []


def open_answer(output, rec):
    """Classify a free-form answer as 'image', 'text' (or 'foil' on controls), or 'other'."""
    hits = set(mentioned(output, rec["question_type"]))
    img = rec["answer_image"] in hits
    foil = rec["foil"] in hits
    if img and not foil:
        return "image"
    if foil and not img:
        return "text" if rec["is_conflict"] else "foil"
    return "other"


def mcq_answer(choice, rec):
    """Same classes as open_answer, from a chosen letter."""
    if choice is None:
        return "other"
    opt = rec["mcq_options"][ord(choice) - ord("A")]
    if opt == rec["answer_image"]:
        return "image"
    if opt == rec["foil"]:
        return "text" if rec["is_conflict"] else "foil"
    return "other"


# ---------------------------------------------------------------- attribution

FIELDS = {
    "error": r"error",
    "input": r"wrong input",
    "type": r"error type",
    "span": r"wrong words",
    "obvious": r"how obvious",
}

TYPE_WORDS = {
    "object": ("object", "noun", "entity"),
    "attribute": ("attribute", "color", "colour", "property"),
    "relation": ("relation", "relationship", "spatial", "position", "location"),
    "none": ("none",),
}
INPUT_WORDS = {
    "text": ("text", "description", "caption"),
    "image": ("image", "picture", "photo"),
    "none": ("none", "neither"),
}
OBVIOUS_WORDS = {"obvious": ("obvious",), "moderate": ("moderate",), "subtle": ("subtle",),
                 "none": ("none",)}


def _field(output, label):
    m = re.search(rf"^\s*[*\-]*\s*{label}\s*[*]*\s*:\s*(.*)$", output or "", re.I | re.M)
    return m.group(1).strip().strip("*").strip() if m else None


def _first_class(value, table):
    if value is None:
        return None
    ws = words(value)
    for w in ws:
        for cls, keys in table.items():
            if w in keys:
                return cls
    return None


def attribution(output):
    # the label must be followed directly by a colon, so "error" does not match "Error type:"
    raw = {k: _field(output, pat) for k, pat in FIELDS.items()}
    return {
        "error": _first_class(raw["error"], {"yes": ("yes",), "no": ("no",)}),
        "input": _first_class(raw["input"], INPUT_WORDS),
        "type": _first_class(raw["type"], TYPE_WORDS),
        "span": raw["span"],
        "obvious": _first_class(raw["obvious"], OBVIOUS_WORDS),
        "parsed": any(v is not None for v in raw.values()),
    }


def span_correct(span, rec):
    """The named wrong words identify the edited value and not the true one."""
    if not span or not rec["is_conflict"]:
        return False
    hits = mentioned(span, rec["question_type"])
    return rec["answer_text"] in hits and rec["answer_image"] not in hits
