import random

from faultline import vocab


def obj(name, x, y, w, h, attrs=(), rels=()):
    return {"name": name, "x": x, "y": y, "w": w, "h": h,
            "attributes": list(attrs), "relations": list(rels)}


def street_scene():
    """1000x1000 image: big red bus left of a tree, small brown dog below the bus, two men."""
    return {
        "width": 1000, "height": 1000,
        "objects": {
            "1": obj("bus", 50, 300, 400, 300, ["red", "large"]),        # area 0.12
            "2": obj("tree", 600, 250, 200, 400, ["green"]),
            "3": obj("dog", 150, 750, 120, 150, ["brown"]),              # area 0.018 < min_area
            "4": obj("man", 820, 600, 60, 200),
            "5": obj("man", 900, 600, 60, 200),
            "6": obj("sign", 470, 100, 60, 60, ["white"]),
        },
    }


def config(**quota):
    q = {"per_cell": 3, "reserve_per_cell": 1, "controls": 6, "reserve_controls": 3,
         "core_per_cell": 2, "core_controls": 3, "max_per_image": 2}
    q.update(quota)
    return {
        "seed": 7,
        "quota": q,
        "target": {"min_area": 0.02, "anchor_min_area": 0.01},
        "severity": {"salient_area": 0.10},
        "relation": {"min_gap": 0.02, "min_overlap": 0.30, "salient_gap": 0.15,
                     "salient_min_area": 0.03, "subtle_gap": 0.06},
    }


def random_scenes(n, seed=0):
    """Many synthetic images with random in-vocabulary objects laid out on a grid."""
    rng = random.Random(seed)
    names = vocab.OBJECT_NAMES
    graphs = {}
    for i in range(n):
        objs = {}
        k = rng.randint(2, 5)
        chosen = rng.sample(names, k) + ["tree"]
        for j, name in enumerate(chosen):
            col, row = j % 3, j // 3
            size = rng.choice([90, 150, 300])
            x = col * 330 + rng.randint(0, 20)
            y = row * 500 + rng.randint(0, 20)
            attrs = [rng.choice(vocab.COLORS)] if rng.random() < 0.8 else []
            objs[str(j + 1)] = obj(name, x, y, size, size, attrs)
        graphs[str(1000 + i)] = {"width": 1000, "height": 1000, "objects": objs}
    return graphs
