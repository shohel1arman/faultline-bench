"""Probe prompts. Any change here changes results, so bump PROMPT_VERSION when editing."""

PROMPT_VERSION = "p1"

LETTERS = ["A", "B", "C", "D"]

MCQ_SUFFIX = "Answer with the option's letter from the given choices directly."

DETECT = (
    'Here is a description of the image: "{text}"\n'
    "Does the description accurately describe the image? "
    'Answer "yes" or "no" first, then explain briefly.'
)

OPEN = 'Description: "{text}"\n{question} Answer with a single word or short phrase.'

FOCUS = "If the description conflicts with the image, rely on the image."

ATTRIBUTE = (
    'Here is a description of the image: "{text}"\n'
    "Check the description against the image. Reply in exactly this format:\n"
    "Error: yes or no\n"
    "Wrong input: text, image, or none\n"
    "Error type: object, attribute, relation, or none\n"
    "Wrong words: the exact wrong words from the description, or none\n"
    "How obvious: obvious, moderate, subtle, or none"
)


def options_block(options):
    return "\n".join(f"{L}. {o}" for L, o in zip(LETTERS, options))


def mcq(question, options, text=None, focus=False):
    parts = []
    if text is not None:
        parts.append(f'Description: "{text}"')
    if focus:
        parts.append(FOCUS)
    parts.append(question)
    parts.append(options_block(options))
    parts.append(MCQ_SUFFIX)
    return "\n".join(parts)


def probe_turns(rec, image):
    """All probes for one record, as (name, kind, turns, max_new_tokens).

    kind is "gen" (free generation) or "mcq" (letter scoring). turns is a list of
    {"role", "text", "image"}; the image is attached to the first user turn only.
    P4 ("verbalize") depends on the attribution output and is built separately.
    """
    text, q, opts = rec["text"], rec["question"], rec["mcq_options"]
    user = lambda t, img=image: [{"role": "user", "text": t, "image": img}]
    return [
        ("image_only", "mcq", user(mcq(q, opts)), None),
        ("text_only", "mcq", user(mcq(q, opts, text=text), None), None),
        ("detect", "gen", user(DETECT.format(text=text)), 48),
        ("open", "gen", user(OPEN.format(text=text, question=q)), 12),
        ("mcq", "mcq", user(mcq(q, opts, text=text)), None),
        ("mcq_focus", "mcq", user(mcq(q, opts, text=text, focus=True)), None),
        ("attribute", "gen", user(ATTRIBUTE.format(text=text)), 80),
    ]


def verbalize_turns(rec, image, attribution_output):
    return [
        {"role": "user", "text": ATTRIBUTE.format(text=rec["text"]), "image": image},
        {"role": "assistant", "text": attribution_output, "image": None},
        {"role": "user", "text": mcq(rec["question"], rec["mcq_options"]), "image": None},
    ]
