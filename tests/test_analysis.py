import random

import pytest

from faultline.analysis import bootstrap, common_ids, joint, mcnemar


def conflict(i, resolved, flagged=False, att=False, verbal=None):
    return {"id": f"r{i}", "is_conflict": True, "perceives": True, "reads": True,
            "flagged": flagged, "detect": "flagged" if flagged else "accepted",
            "mcq": "image" if resolved else "text", "mcq_focus": "text", "open": "other",
            "verbalize": verbal or ("image" if resolved else "text"),
            "att_parsed": True, "att_score": 1.0 if att else 0.0, "att_strict": att,
            "a_input": att, "a_type": att, "a_span": att, "a_obvious": att}


def control(i, flagged=False):
    return {"id": f"c{i}", "is_conflict": False, "perceives": True, "reads": True,
            "flagged": flagged, "detect": "flagged" if flagged else "accepted",
            "mcq": "image", "mcq_focus": "image", "open": "image", "verbalize": "image",
            "att_parsed": True}


def test_mcnemar_exact_value():
    outs = [conflict(i, resolved=False, verbal="image") for i in range(10)]
    outs += [conflict(100 + i, resolved=True, verbal="text") for i in range(2)]
    a_only, b_only, p = mcnemar(outs, "verbalize", "mcq")
    assert (a_only, b_only) == (10, 2)
    assert p == pytest.approx(2 * (1 + 12 + 66) / 4096)


def test_point_estimates_and_contrasts():
    outs = [conflict(i, resolved=i < 30, flagged=i < 60, att=i < 10) for i in range(100)]
    outs += [control(i, flagged=i < 5) for i in range(50)]
    image_of = {o["id"]: o["id"] for o in outs}
    b = bootstrap(outs, image_of, reps=200)
    assert b["ResA_mcq"][0] == pytest.approx(0.30)
    assert b["DetR"][0] == pytest.approx(0.60)
    assert b["FFR"][0] == pytest.approx(0.10)
    assert b["DetR_adj"][0] == pytest.approx(0.50)
    assert b["R_given_A"][0] == pytest.approx(1.0)
    assert b["R_given_notA"][0] == pytest.approx(20 / 90)
    assert b["coupling_A"][0] == pytest.approx(1 - 20 / 90)
    for name in ("ResA_mcq", "DetR", "DetR_adj"):
        est, lo, hi, _ = b[name]
        assert lo <= est <= hi


def test_intervals_respect_image_clusters():
    rng = random.Random(0)
    base = [conflict(i, resolved=rng.random() < 0.5) for i in range(200)]
    copies = [dict(o, id=o["id"] + "b") for o in base]
    outs = base + copies
    same_image = {o["id"]: o["id"].rstrip("b") for o in outs}
    own_image = {o["id"]: o["id"] for o in outs}
    w_clustered = bootstrap(outs, same_image, reps=1000)["ResA_mcq"]
    w_naive = bootstrap(outs, own_image, reps=1000)["ResA_mcq"]
    single = bootstrap(base, {o["id"]: o["id"] for o in base}, reps=1000)["ResA_mcq"]
    width = lambda t: t[2] - t[1]
    # duplicated records carry no new information: clustering keeps the single-copy width
    assert width(w_clustered) == pytest.approx(width(single), rel=0.2)
    assert width(w_naive) < 0.85 * width(w_clustered)


def test_joint_patterns_sum_to_one():
    outs = [conflict(i, resolved=i % 2 == 0, flagged=i % 3 == 0, att=i % 5 == 0) for i in range(60)]
    j = joint(outs)
    assert sum(c for c, _ in j.values()) == 60
    assert sum(s for _, s in j.values()) == pytest.approx(1.0)
    assert j["DRA"][0] == sum(1 for i in range(60) if i % 2 == 0 and i % 3 == 0 and i % 5 == 0)


def test_common_ids():
    a = [conflict(i, True) for i in range(5)]
    b = [conflict(i, True) for i in range(3, 8)]
    assert common_ids({"a": a, "b": b}) == {"r3", "r4"}


def test_verbalize_gain_decomposes_into_structure_check_and_report():
    outs = []
    for i in range(100):
        o = conflict(i, resolved=i < 10)
        o["delay"] = "image" if i < 20 else "text"         # +10 from the turn structure
        o["check_silent"] = "image" if i < 35 else "text"  # +15 from being asked to check
        o["verbalize"] = "image" if i < 60 else "text"     # +25 from the model's own report
        outs.append(o)
    old = conflict(999, resolved=False)                    # a record without the control probes
    old.update(delay=None, check_silent=None, verbalize="image")
    b = bootstrap(outs + [old], {o["id"]: o["id"] for o in outs + [old]}, reps=200)
    assert b["structure_gain"][0] == pytest.approx(0.10)
    assert b["check_gain"][0] == pytest.approx(0.15)
    assert b["report_gain"][0] == pytest.approx(0.25)
    assert b["ResA_mcq_c"][3] == 100                       # the record without controls is left out
    a_only, b_only, _ = mcnemar(outs + [old], "check_silent", "delay")
    assert (a_only, b_only) == (15, 0)
