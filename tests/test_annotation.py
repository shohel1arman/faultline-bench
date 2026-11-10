from collections import Counter

import pytest

from faultline.annotation import (CANT_TELL, cohen_kappa, item_for_page, judge, sample,
                                  spearman, summarize)
from faultline.runner import cell_of
from faultline.schema import read_jsonl


@pytest.fixture(scope="module")
def records():
    return read_jsonl("data/faultline_v0.jsonl")


def test_sample_is_balanced_fixed_and_inside_the_first_1000(records):
    from faultline.runner import select
    s = sample(records)
    assert len(s) == 500 and len({r["id"] for r in s}) == 500
    cells = Counter(cell_of(r) for r in s)
    assert all(n == 50 for c, n in cells.items() if c[0] != "control")
    assert sum(n for c, n in cells.items() if c[0] == "control") == 50
    first = {r["id"] for r in select(records, "core", first=1000)}
    assert {r["id"] for r in s} <= first
    assert [r["id"] for r in sample(records)] == [r["id"] for r in s]       # same seed, same order


def test_page_item_reveals_nothing_about_the_label(records):
    it = item_for_page(records[0], "data:")
    assert set(it) == {"id", "image", "question", "options", "text"}


def rec(answer="cat", conflict=True, severity="S1"):
    return {"id": "x", "answer_image": answer, "is_conflict": conflict, "severity": severity,
            "conflict_type": "object", "question_type": "object"}


def test_judge():
    j = judge(rec(), {"choice": "cat", "matches": "no", "obvious": "subtle"})
    assert j["picks_image"] is True and j["desc_correct"] is True and j["obvious"] == 1
    j = judge(rec(conflict=False), {"choice": CANT_TELL, "matches": "no"})
    assert j["answered"] is False and j["picks_image"] is None and j["desc_correct"] is False


def test_kappa_and_spearman():
    assert cohen_kappa(list("aabb"), list("aabb")) == pytest.approx(1.0)
    assert cohen_kappa(list("abab"), list("abba")) == pytest.approx(0.0)
    assert spearman([3, 2, 1, 3, 2, 1], [3, 2, 1, 3, 1, 1]) > 0.8
    assert spearman([1, 1, 1], [1, 2, 3]) is None


def test_summarize_flags_agreed_label_problems():
    recs = {f"r{i}": {**rec(severity=["S1", "S2", "S3"][i % 3]), "id": f"r{i}"} for i in range(9)}
    recs["c0"] = {**rec(conflict=False), "id": "c0", "question_type": "object"}
    good = {"choice": "cat", "matches": "no", "obvious": "obvious"}
    a = {i: dict(good) for i in recs}
    b = {i: dict(good) for i in recs}
    a["c0"] = b["c0"] = {"choice": "cat", "matches": "yes"}
    a["r1"] = b["r1"] = {"choice": "dog", "matches": "no", "obvious": "moderate"}   # label wrong
    a["r2"] = b["r2"] = {"choice": "cat", "matches": "yes"}                         # edit not visible
    b["r3"] = {"choice": CANT_TELL, "matches": "no", "obvious": "subtle"}           # disagreement only
    res = summarize(recs, {"A1": a, "A2": b})
    assert res["n_items"] == 10
    assert res["all_agree_other_answer"] == ["r1"]
    assert res["all_agree_conflict_matches"] == ["r2"]
    assert res["valid_rate"] == pytest.approx(0.8)
    assert res["per_annotator"]["A2"]["cant_tell_question"] == pytest.approx(0.1)
    assert res["agree_choice"] == pytest.approx(0.9)


def test_sample_exclude_gives_disjoint_round(records):
    first = sample(records)
    second = sample(records, seed=2, first=None, exclude={r["id"] for r in first})
    assert not {r["id"] for r in first} & {r["id"] for r in second}
    assert len(second) == len(first)
