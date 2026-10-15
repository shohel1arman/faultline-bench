"""Record schema and invariants. validate() returns a list of problems (empty = valid)."""

import json

REQUIRED = {
    "id": str, "image_id": str, "image_file": str, "is_conflict": bool,
    "conflict_type": str, "question_type": str, "text": str, "text_clean": str,
    "question": str, "answer_image": str, "answer_text": str, "mcq_options": list,
    "target": dict, "split": str, "core": bool, "width": int, "height": int,
}
TYPES = {"object", "attribute", "relation"}


def validate(rec):
    errs = []
    for k, typ in REQUIRED.items():
        if k not in rec:
            errs.append(f"missing {k}")
        elif not isinstance(rec[k], typ):
            errs.append(f"{k} should be {typ.__name__}")
    if errs:
        return errs

    if rec["question_type"] not in TYPES:
        errs.append(f"bad question_type {rec['question_type']}")
    if rec["split"] not in {"main", "reserve"}:
        errs.append(f"bad split {rec['split']}")
    opts = rec["mcq_options"]
    if len(opts) != 4 or len(set(opts)) != 4:
        errs.append("mcq_options must be 4 distinct options")
    if rec["answer_image"] not in opts:
        errs.append("answer_image not among options")
    if rec["answer_text"] not in opts:
        errs.append("answer_text not among options")
    if rec.get("foil") not in opts or rec.get("foil") == rec["answer_image"]:
        errs.append("foil must be a wrong option")

    if not rec["is_conflict"]:
        if rec["conflict_type"] != "none" or rec.get("severity") is not None:
            errs.append("control must have conflict_type none and no severity")
        if rec["text"] != rec["text_clean"]:
            errs.append("control text differs from text_clean")
        if rec["answer_image"] != rec["answer_text"]:
            errs.append("control answers differ")
        return errs

    if rec["conflict_type"] != rec["question_type"] or rec["conflict_type"] not in TYPES:
        errs.append("conflict_type mismatch")
    if rec.get("severity") not in {"S1", "S2", "S3"}:
        errs.append("bad severity")
    if rec["answer_image"] == rec["answer_text"]:
        errs.append("conflict answers are identical")

    s0, s1 = rec["edited_span_char"]
    o0, o1 = rec["original_span_char"]
    text, clean = rec["text"], rec["text_clean"]
    if text[s0:s1] != rec["edited_span"]:
        errs.append("edited_span_char does not point at edited_span")
    if clean[o0:o1] != rec["original_span"]:
        errs.append("original_span_char does not point at original_span")
    if s0 != o0 or text[:s0] != clean[:o0] or text[s1:] != clean[o1:]:
        errs.append("text and text_clean differ outside the edited span")
    if rec["edited_span"] != rec["answer_text"] or rec["original_span"] != rec["answer_image"]:
        errs.append("spans and answers disagree")
    return errs


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
