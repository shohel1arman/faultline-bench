import random
from collections import Counter

from faultline import vocab
from faultline.generate import build, image_candidates, render
from faultline.schema import validate

from fixtures import config, obj, random_scenes, street_scene


def cands_for(graph, seed=0):
    return image_candidates("1", graph, config(), random.Random(seed))


def bus_and_tree():
    g = street_scene()
    for k in ("3", "6"):
        del g["objects"][k]
    return g


def test_street_scene_candidates():
    cands = cands_for(bus_and_tree())
    kinds = {(c["type"], c["severity"], c["target"].canon) for c in cands}
    assert ("attribute", "S1", "bus") in kinds
    assert ("attribute", "S3", "bus") in kinds
    assert ("relation", "S1", "bus") in kinds
    assert ("object", "S1", "bus") in kinds
    assert ("object", "S3", "bus") in kinds


def test_small_and_out_of_vocabulary_objects_are_not_targets():
    cands = cands_for(street_scene())
    assert not any(c["target"].canon == "dog" for c in cands)
    assert not any(c["target"].name == "man" for c in cands)


def test_relation_severity_depends_on_anchor_size():
    cands = [c for seed in range(30) for c in cands_for(street_scene(), seed)
             if c["type"] == "relation" and c["target"].canon == "bus"]
    sev = {c["anchor"].name: c["severity"] for c in cands}
    assert sev["tree"] == "S1"
    assert sev["dog"] == "S2"


def test_object_replacements_are_absent_and_correctly_grouped():
    for seed in range(20):
        for c in cands_for(street_scene(), seed):
            if c["type"] != "object":
                continue
            assert c["scene"].is_absent(c["new"])
            same = vocab.GROUP_OF[c["new"]] == vocab.GROUP_OF[c["orig"]]
            assert same == (c["edit"] == "near")
            assert (c["severity"] == "S3") == same


def test_color_edits_respect_families():
    for seed in range(20):
        for c in cands_for(street_scene(), seed):
            if c["type"] != "attribute":
                continue
            if c["edit"] == "near":
                assert c["new"] in vocab.NEAR_COLORS[c["orig"]]
            else:
                assert vocab.COLOR_FAMILY[c["new"]] != vocab.COLOR_FAMILY[c["orig"]]


def test_duplicate_targets_are_skipped():
    g = street_scene()
    g["objects"]["7"] = obj("bus", 500, 700, 300, 250, ["blue"])
    assert not any(c["target"].canon == "bus" for c in cands_for(g))


def test_multi_colored_target_gets_no_attribute_edit():
    g = street_scene()
    g["objects"]["1"]["attributes"] = ["red", "white"]
    assert not any(c["type"] == "attribute" and c["target"].canon == "bus" for c in cands_for(g))


def test_rendered_records_validate():
    rng = random.Random(1)
    for c in cands_for(street_scene()):
        for conflict in (True, False):
            rec = render(c, conflict, rng)
            rec.update(id="x", image_file="1.jpg", split="main", core=True)
            assert validate(rec) == [], (rec, validate(rec))


def test_relation_text_example():
    c = next(c for c in cands_for(bus_and_tree()) if c["type"] == "relation")
    rec = render(c, True, random.Random(0))
    assert "The red bus is to the right of the tree." in rec["text"]
    assert rec["question"] == "Where is the bus relative to the tree?"
    assert rec["answer_image"] == "to the left of"


def test_build_quotas_cap_and_determinism():
    graphs = random_scenes(600, seed=3)
    cfg = config(per_cell=8, reserve_per_cell=2, controls=12, reserve_controls=3,
                 core_per_cell=5, core_controls=6)
    recs, supply = build(graphs, cfg)
    recs2, _ = build(graphs, cfg)
    assert recs == recs2

    for r in recs:
        assert validate(r) == [], validate(r)
    assert len({r["id"] for r in recs}) == len(recs)

    per_image = Counter(r["image_id"] for r in recs)
    assert max(per_image.values()) <= 2
    targets = Counter((r["image_id"], r["target"]["object_id"]) for r in recs)
    assert max(targets.values()) == 1

    cells = Counter((r["conflict_type"], r["severity"], r["split"]) for r in recs if r["is_conflict"])
    core = Counter((r["conflict_type"], r["severity"]) for r in recs if r["is_conflict"] and r["core"])
    for t in ("object", "attribute", "relation"):
        for s in ("S1", "S2", "S3"):
            if supply.get((t, s), 0) >= 30:
                assert cells[(t, s, "main")] == 8, (t, s, cells)
                assert cells[(t, s, "reserve")] == 2
                assert core[(t, s)] == 5

    ctrl = Counter((r["question_type"], r["split"]) for r in recs if not r["is_conflict"])
    for t in ("object", "attribute", "relation"):
        assert ctrl[(t, "main")] == 4
        assert ctrl[(t, "reserve")] == 1


def test_color_distractors_are_not_close_to_the_true_color():
    graphs = random_scenes(400, seed=5)
    recs, _ = build(graphs, config(per_cell=10, reserve_per_cell=0, controls=30,
                                   reserve_controls=0, core_per_cell=5, core_controls=6))
    for r in recs:
        if r["question_type"] != "attribute":
            continue
        distractors = set(r["mcq_options"]) - {r["answer_image"], r["foil"]}
        assert len(distractors) == 2
        assert not distractors & set(vocab.NEAR_COLORS[r["answer_image"]])


def _alloc(n_scenes, per_cell, controls, cap=9, reserve=0):
    from faultline.generate import assign, image_candidates
    graphs = random_scenes(n_scenes, seed=11)
    rng = random.Random(1)
    cands = []
    for iid in sorted(graphs):
        cands.extend(image_candidates(iid, graphs[iid], config(), rng))
    cfg = config(per_cell=per_cell, reserve_per_cell=reserve, controls=controls,
                 reserve_controls=0, core_per_cell=1, core_controls=3, max_per_image=cap)
    return assign(cands, cfg, random.Random(5))


def _check_valid(picked, cap):
    targets, images = Counter(), Counter()
    for cell, (main, reserve) in picked.items():
        for c in main + reserve:
            targets[(c["image_id"], c["target"].oid)] += 1
            images[c["image_id"]] += 1
            if cell[1] == "control":
                assert c["type"] == cell[0]
            else:
                assert (c["type"], c["severity"]) == cell
    assert max(targets.values()) == 1
    assert max(images.values()) <= cap


def test_allocation_fills_every_cell_when_quotas_fit():
    picked = _alloc(800, per_cell=180, controls=150)
    _check_valid(picked, 9)
    for cell, (main, _) in picked.items():
        if cell != ("relation", "S1"):          # this cell has only ~175 candidates here
            assert len(main) == (50 if cell[1] == "control" else 180), cell


def test_allocation_spreads_a_shortage_evenly():
    # object S2 and S3 come from the same objects, which relation cells also want; a
    # cell-by-cell greedy fill left attribute S2 and all controls nearly empty here
    picked = _alloc(800, per_cell=400, controls=180)
    _check_valid(picked, 9)
    frac = {cell: len(main) / (60 if cell[1] == "control" else 400)
            for cell, (main, _) in picked.items() if cell != ("relation", "S1")}
    assert min(frac.values()) > 0.55
    assert max(frac.values()) - min(frac.values()) < 0.05


def test_main_quotas_are_filled_before_reserves():
    picked = _alloc(800, per_cell=200, controls=150, reserve=200)
    for cell, (main, reserve) in picked.items():
        if cell != ("relation", "S1") and reserve:
            assert len(main) == (50 if cell[1] == "control" else 200), cell


def test_build_is_identical_across_hash_seeds(tmp_path):
    import os
    import subprocess
    import sys
    code = (
        "import hashlib, json, sys; sys.path[:0] = ['.', 'tests'];"
        "from faultline.generate import build; from fixtures import config, random_scenes;"
        "recs, _ = build(random_scenes(300, seed=4), config());"
        "print(hashlib.sha256(json.dumps(recs, sort_keys=True).encode()).hexdigest())"
    )
    digests = set()
    for seed in ("1", "2", "3"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
        digests.add(out.stdout.strip())
    assert len(digests) == 1


def test_animal_colours_are_never_stated():
    # a cow labelled only "white" is often black and white: no colour edits, colour
    # mentions or filler sentences about animals
    g = street_scene()
    g["objects"]["7"] = obj("cow", 600, 700, 250, 250, ["white"])
    g["objects"]["8"] = obj("bench", 50, 650, 300, 200, ["brown"])
    animals = set(vocab.OBJECT_GROUPS["animal"])
    for seed in range(30):
        for c in image_candidates("1", g, config(), random.Random(seed)):
            if c["type"] == "attribute":
                assert c["target"].canon not in animals
            rec = render(c, True, random.Random(seed))
            for a in animals:
                for col in vocab.COLORS:
                    assert f"{col} {a}" not in rec["text"] and f"The {a} is {col}." not in rec["text"]


def test_built_dataset_has_no_animal_colour_items():
    recs, _ = build(random_scenes(600, seed=3), config(per_cell=8, reserve_per_cell=2, controls=12,
                                                      reserve_controls=3, core_per_cell=5, core_controls=6))
    animals = set(vocab.OBJECT_GROUPS["animal"])
    assert not [r for r in recs if r["question_type"] == "attribute" and r["target"]["name"] in animals]
