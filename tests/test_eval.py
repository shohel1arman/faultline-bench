import json
import random

import pytest

from faultline import parse
from faultline.generate import build
from faultline.models import _messages
from faultline.prompts import LETTERS, probe_turns
from faultline.runner import done_ids, run, select
from faultline.score import outcome, summarize

from fixtures import config, random_scenes


@pytest.fixture(scope="module")
def records():
    recs, _ = build(random_scenes(400, seed=2),
                    config(per_cell=6, reserve_per_cell=1, controls=9, reserve_controls=0,
                           core_per_cell=3, core_controls=6))
    return recs


class TextFollower:
    """Fake model: always trusts the text, never sees a conflict, half-follows formats."""

    def __init__(self, recs, fail_ids=()):
        self.recs = recs
        self.fail_ids = set(fail_ids)
        self.calls = 0

    def _rec(self, turns):
        # text-only probes carry no image, but their prompt contains the full description
        img = turns[0]["image"]
        for r in self.recs:
            if img is not None and img.name != r["image_file"]:
                continue
            if r["question"] in turns[-1]["text"] or r["text"] in turns[0]["text"]:
                if img is None and r["text"] not in turns[0]["text"]:
                    continue
                return r
        raise AssertionError("record not found")

    def generate(self, turns, max_new_tokens):
        self.calls += 1
        r = self._rec(turns)
        if r["id"] in self.fail_ids:
            raise RuntimeError("CUDA out of memory")
        t = turns[0]["text"]
        if "Reply in exactly this format" in t:
            out = "Error: no\nWrong input: none\nError type: none\nWrong words: none\nHow obvious: none"
        elif "accurately describe" in t:
            out = "Yes, the description matches the image."
        else:
            out = r["answer_text"]
        return {"output": out, "n_in": 10, "n_out": 5}

    def letter_probs(self, turns):
        self.calls += 1
        r = self._rec(turns)
        has_text = "Description:" in turns[-1]["text"] or len(turns) > 1
        target = r["answer_text"] if has_text else r["answer_image"]
        k = r["mcq_options"].index(target)
        probs = {L: (0.7 if i == k else 0.1) for i, L in enumerate(LETTERS)}
        return {"probs": probs, "letter_mass": 0.9, "n_in": 10}


def load_image_stub(path, max_side):
    return path


# ------------------------------------------------------------------ prompts / messages

def test_probe_turns_shapes(records):
    turns = probe_turns(records[0], image="IMG")
    names = [n for n, *_ in turns]
    assert names == ["image_only", "text_only", "detect", "open", "mcq", "mcq_focus", "attribute"]
    by = {n: t for n, _, t, _ in turns}
    assert by["text_only"][0]["image"] is None
    assert by["image_only"][0]["image"] == "IMG"
    assert records[0]["text"] not in by["image_only"][0]["text"]


def test_messages_place_image_first_and_only_once():
    msgs = _messages([{"role": "user", "text": "q1", "image": "IMG"},
                      {"role": "assistant", "text": "a1", "image": None},
                      {"role": "user", "text": "q2", "image": None}])
    assert msgs[0]["content"][0] == {"type": "image"}
    assert msgs[1]["role"] == "assistant"
    assert all(c["type"] == "text" for c in msgs[2]["content"])


# ------------------------------------------------------------------ parsing

@pytest.mark.parametrize("out,label", [
    ("No, the bus is red.", "flagged"),
    ("Yes.", "accepted"),
    ("**No**. The car is blue", "flagged"),
    ("The description is not accurate.", "flagged"),
    ("The description accurately describes the image.", "accepted"),
    ("I cannot tell.", None),
])
def test_detection(out, label):
    assert parse.detection(out) == label


def test_relation_parsing():
    assert parse.relation_of("To the left of the tree") == "to the left of"
    assert parse.relation_of("on top of it") == "above"
    assert parse.relation_of("underneath") == "below"
    assert parse.relation_of("left and above") is None


def test_open_answer_classes():
    rec = {"question_type": "attribute", "answer_image": "red", "answer_text": "blue",
           "foil": "blue", "is_conflict": True}
    assert parse.open_answer("Red.", rec) == "image"
    assert parse.open_answer("It is blue", rec) == "text"
    assert parse.open_answer("red and blue", rec) == "other"
    assert parse.open_answer("Grey", rec) == "other"
    obj = {"question_type": "object", "answer_image": "bicycle", "answer_text": "car",
           "foil": "car", "is_conflict": True}
    assert parse.open_answer("A bike", obj) == "image"
    assert parse.open_answer("cars", obj) == "text"


def test_attribution_parsing_is_lenient():
    out = ("**Error:** yes\n- Wrong input: the text description\nError type: color\n"
           "Wrong words: \"blue\"\nHow obvious: Subtle")
    a = parse.attribution(out)
    assert a == {"error": "yes", "input": "text", "type": "attribute", "span": '"blue"',
                 "obvious": "subtle", "parsed": True}
    assert parse.attribution("The image shows a bus.")["parsed"] is False


def test_span_correct():
    rec = {"is_conflict": True, "question_type": "relation",
           "answer_text": "to the right of", "answer_image": "to the left of"}
    assert parse.span_correct("to the right of", rec)
    assert parse.span_correct("right", rec)
    assert not parse.span_correct("left", rec)
    assert not parse.span_correct(None, rec)


# ------------------------------------------------------------------ runner

def test_run_writes_resumes_and_retries(tmp_path, records):
    rows = select(records, subset="core")
    out = tmp_path / "o.jsonl"
    bad = rows[1]["id"]

    fake = TextFollower(records, fail_ids=[bad])
    run(fake, rows, "imgs", out, "fake", 672, load_image_stub, log_every=1000)
    lines = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == len(rows)
    assert [l["id"] for l in lines if l["error"]] == [bad]
    assert done_ids(out) == {r["id"] for r in rows} - {bad}

    # simulate a crash that left half a line, then resume: only the failed record runs
    with open(out, "a") as f:
        f.write('{"id": "fl_trunc')
    fake2 = TextFollower(records)
    run(fake2, rows, "imgs", out, "fake", 672, load_image_stub, log_every=1000)
    assert fake2.calls == 8
    assert done_ids(out) == {r["id"] for r in rows}
    for line in out.read_text(encoding="utf-8").splitlines():
        if line.startswith('{"id": "fl_trunc'):
            continue
        json.loads(line)


def test_select_shards_partition(records):
    full = select(records, subset="main")
    parts = [select(records, subset="main", shard=(k, 3)) for k in range(3)]
    assert sorted(r["id"] for p in parts for r in p) == sorted(r["id"] for r in full)
    assert all(r["split"] == "main" for r in full)


# ------------------------------------------------------------------ scoring

def test_text_follower_scores_like_a_text_follower(tmp_path, records):
    rows = select(records, subset="main")
    out = tmp_path / "o.jsonl"
    run(TextFollower(records), rows, "imgs", out, "fake", 672, load_image_stub, log_every=1000)
    recs = {r["id"]: r for r in records}
    outs = [outcome(recs[l["id"]], l["probes"]) for l in map(json.loads, out.read_text(encoding="utf-8").splitlines())]
    s = summarize(outs)
    assert s["perceive"] == 1.0
    assert s["read"] == 1.0
    assert s["DetR"] == 0.0 and s["FFR"] == 0.0
    assert s["ResA_mcq"] == 0.0 and s["ResA_open"] == 0.0
    assert s["TPR_mcq"] == 1.0
    assert s["AttA_strict"] == 0.0
    assert s["control_acc_mcq"] == 1.0
    assert s["joint"] == {"---": 1.0}
    assert s["unparsed_detect"] == 0.0 and s["unparsed_attribute"] == 0.0


# ------------------------------------------------------------------ backend logic with a stub model

torch = pytest.importorskip("torch")


def stub_backend(logit_for):
    """HFBackend with processor and model replaced; checks letter scoring and output slicing."""
    from transformers.feature_extraction_utils import BatchFeature

    from faultline.models import HFBackend

    class Tok:
        vocab = {"A": 5, " A": 6, "B": 7, " B": 8, "C": 9, " C": 10, "D": 11, " D": 12}

        def encode(self, s, add_special_tokens=False):
            return [self.vocab[s]]

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(str(int(i)) for i in ids) + " "

    class Proc:
        tokenizer = Tok()

        def apply_chat_template(self, msgs, add_generation_prompt, tokenize):
            assert add_generation_prompt and not tokenize
            return "prompt"

        def __call__(self, text, return_tensors, images=None):
            return BatchFeature({"input_ids": torch.tensor([[1, 2, 3]]),
                                 "pixel_values": torch.zeros(1, 3, dtype=torch.float32)})

    class Model:
        def __call__(self, input_ids, pixel_values):
            assert pixel_values.dtype == torch.float16
            logits = torch.full((1, 3, 20), -10.0)
            for tid, v in logit_for.items():
                logits[0, -1, tid] = v
            return type("O", (), {"logits": logits})()

        def generate(self, input_ids, pixel_values, max_new_tokens, do_sample):
            return torch.cat([input_ids, torch.tensor([[14, 15]])], dim=1)

    b = object.__new__(HFBackend)
    b.torch, b.device, b.dtype = torch, "cpu", torch.float16
    b.processor, b.tok, b.model = Proc(), Proc.tokenizer, Model()
    b.letter_ids = {L: b._letter_token_ids(L) for L in LETTERS}
    return b


def test_letter_probs_combines_spaced_and_bare_letters():
    b = stub_backend({5: 2.0, 6: 2.0, 7: 2.0, 9: 0.0, 11: 0.0})
    res = b.letter_probs([{"role": "user", "text": "q", "image": "IMG"}])
    p = res["probs"]
    assert abs(sum(p.values()) - 1) < 1e-9
    # A has two variants at logit 2, B one: A should get twice B's probability
    assert p["A"] / p["B"] == pytest.approx(2.0, rel=1e-3)
    assert p["C"] == pytest.approx(p["D"])
    assert 0.9 < res["letter_mass"] <= 1.0


def test_letter_probs_flags_nan():
    b = stub_backend({5: float("nan")})
    assert b.letter_probs([{"role": "user", "text": "q", "image": None}]) == {"error": "non-finite logits"}


def test_generate_returns_only_new_tokens():
    b = stub_backend({})
    res = b.generate([{"role": "user", "text": "q", "image": "IMG"}], 8)
    assert res == {"output": "14 15", "n_in": 3, "n_out": 2}


# ------------------------------------------------------------------ configs and devices

def test_quantisation_is_refused_off_cuda():
    from faultline.models import HFBackend
    with pytest.raises(ValueError, match="CUDA"):
        HFBackend("unused/model", device="mps", quant="nf4")


def test_model_profiles_agree_on_ids():
    import tomllib
    with open("configs/models.toml", "rb") as f:
        cuda = tomllib.load(f)
    with open("configs/models_mac.toml", "rb") as f:
        mac = tomllib.load(f)
    assert mac["defaults"]["quant"] == "none"
    for key, m in mac["models"].items():
        if key in cuda["models"]:
            assert m["hf_id"] == cuda["models"][key]["hf_id"], key


def test_memory_and_cleanup_on_cpu_device():
    b = stub_backend({})
    b.device = "cpu"
    assert b.memory_gib() is None
    b.cleanup()


def test_missing_image_is_recorded_and_retried(tmp_path, records):
    rows = select(records, subset="core")[:3]
    out = tmp_path / "o.jsonl"

    def load(path, max_side):
        if path.name == rows[1]["image_file"]:
            raise FileNotFoundError(path)
        return path

    run(TextFollower(records), rows, "imgs", out, "fake", 672, load, log_every=1000)
    lines = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert [l["id"] for l in lines if l["error"]] == [rows[1]["id"]]
    assert "FileNotFoundError" in lines[1]["error"]
    assert done_ids(out) == {rows[0]["id"], rows[2]["id"]}


def test_mps_version_check():
    from faultline.models import mps_version_problem
    assert "macOS 15" in mps_version_problem("2.15.0", "14.6")
    assert mps_version_problem("2.14.1", "14.6") is None
    assert mps_version_problem("2.15.0", "15.1") is None
    assert mps_version_problem("2.16.0.dev20261020", "14.6.1") is not None
    assert mps_version_problem("2.15.0", "") is None
