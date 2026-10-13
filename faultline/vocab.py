"""Closed vocabularies used to build and corrupt descriptions.

Object names are canonical forms; ALIASES maps scene-graph spellings onto them.
People are deliberately absent: they are never targets or replacements.
Ambiguous words (orange, mouse, bat, glass) and hypernym pairs (taxi/car, duck/bird,
mug/cup, desk/table) are left out so that every swap is unambiguously false.
"""

OBJECT_GROUPS = {
    "vehicle": [
        "car", "bus", "truck", "train", "motorcycle", "bicycle", "boat",
        "airplane", "van", "scooter", "tractor",
    ],
    "animal": [
        "dog", "cat", "horse", "cow", "sheep", "elephant", "giraffe", "zebra",
        "bird", "bear", "goat",
    ],
    "food": [
        "pizza", "sandwich", "banana", "apple", "donut", "cake", "broccoli",
        "carrot", "hot dog", "hamburger", "cookie", "pie",
    ],
    "furniture": [
        "chair", "table", "couch", "bed", "bench", "shelf", "cabinet",
        "stool",
    ],
    "tableware": [
        "cup", "bowl", "plate", "bottle", "vase", "pitcher", "jar",
        "fork", "knife", "spoon",
    ],
    "electronics": [
        "laptop", "phone", "television", "keyboard", "remote", "camera",
    ],
    "sports": [
        "surfboard", "skateboard", "snowboard", "frisbee", "kite", "ball",
        "racket",
    ],
    "household": [
        "clock", "lamp", "pillow", "umbrella", "suitcase", "backpack", "book",
        "towel", "basket", "teddy bear",
    ],
    "street": [
        "stop sign", "fire hydrant", "traffic light", "street light",
        "parking meter",
    ],
}

ALIASES = {
    "bike": "bicycle",
    "motorbike": "motorcycle",
    "plane": "airplane",
    "aeroplane": "airplane",
    "jet": "airplane",
    "puppy": "dog",
    "kitten": "cat",
    "doughnut": "donut",
    "burger": "hamburger",
    "sofa": "couch",
    "tv": "television",
    "cellphone": "phone",
    "cell phone": "phone",
    "mobile phone": "phone",
    "telephone": "phone",
    "racquet": "racket",
    "tennis racket": "racket",
    "teddy": "teddy bear",
    "hydrant": "fire hydrant",
    "streetlight": "street light",
    "lamp post": "street light",
    "street lamp": "street light",
    "traffic signal": "traffic light",
    "coffee cup": "cup",
}

GROUP_OF = {name: group for group, names in OBJECT_GROUPS.items() for name in names}
OBJECT_NAMES = sorted(GROUP_OF)

# Non-target objects that are fine as spatial anchors ("... to the left of the tree").
EXTRA_ANCHORS = {"tree", "building", "window", "door", "fence", "house", "plant", "flower"}

# Colour vocabulary. Families group perceptually close hues; NEAR lists the subtle edits.
COLOR_FAMILY = {
    "white": "light", "silver": "light", "beige": "light",
    "black": "dark", "gray": "dark",
    "red": "warm", "orange": "warm", "pink": "warm",
    "yellow": "yellow", "gold": "yellow",
    "brown": "brown", "tan": "brown",
    "blue": "cool", "purple": "cool",
    "green": "green",
}
COLOR_ALIASES = {"grey": "gray"}
COLORS = sorted(COLOR_FAMILY)

NEAR_COLORS = {
    "red": ["orange", "pink"],
    "orange": ["red", "yellow"],
    "pink": ["red", "purple"],
    "yellow": ["gold", "orange"],
    "gold": ["yellow"],
    "brown": ["tan"],
    "tan": ["brown", "beige"],
    "beige": ["tan", "white"],
    "white": ["silver", "beige", "gray"],
    "silver": ["gray", "white"],
    "gray": ["silver", "black", "white"],
    "black": ["gray"],
    "blue": ["purple"],
    "purple": ["blue", "pink"],
    "green": [],
}

# Far edits draw from common basic colours only, so corrupted text stays natural.
FAR_COLOR_POOL = ["white", "black", "red", "blue", "green", "yellow", "brown"]

RELATIONS = ["to the left of", "to the right of", "above", "below"]
ANTONYM = {
    "to the left of": "to the right of",
    "to the right of": "to the left of",
    "above": "below",
    "below": "above",
}

SEVERITY_LABEL = {"S1": "obvious", "S2": "moderate", "S3": "subtle"}


def canonical_object(name):
    name = name.strip().lower()
    name = ALIASES.get(name, name)
    return name if name in GROUP_OF else None


def canonical_color(attr):
    attr = attr.strip().lower()
    attr = COLOR_ALIASES.get(attr, attr)
    return attr if attr in COLOR_FAMILY else None


def far_colors(color):
    fam = COLOR_FAMILY[color]
    near = set(NEAR_COLORS.get(color, []))
    return [c for c in FAR_COLOR_POOL if COLOR_FAMILY[c] != fam and c not in near]


def aliases_of(canonical):
    return {canonical} | {a for a, c in ALIASES.items() if c == canonical}


def article(phrase):
    return "an" if phrase[:1].lower() in "aeiou" else "a"
