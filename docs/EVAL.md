# FaultLine evaluation protocol

Prompts are in `faultline/prompts.py` (version `p1`), parsing in `faultline/parse.py`,
metrics in `faultline/score.py`. Every model sees the same prompts with greedy decoding.

## Probes (per record, conflict and control alike)

| Probe | Input | Output used | Purpose |
|---|---|---|---|
| `image_only` | image + question + 4 options | option scores | can the model perceive the fact at all |
| `text_only` | description + question + options, no image | option scores | can it read the fact from the text |
| `detect` | image + description, "does it accurately describe the image?" | yes/no (48 tokens) | detection |
| `open` | image + description + question, short answer | free text (12 tokens) | resolution, open-ended |
| `mcq` | image + description + question + options | option scores | resolution, multiple choice |
| `mcq_focus` | as `mcq`, plus "if they conflict, rely on the image" | option scores | instruction baseline |
| `attribute` | image + description, fixed 5-line report format | free text (80 tokens) | attribution |
| `verbalize` | `attribute` exchange, then the `mcq` question as a follow-up turn | option scores | does stating the error first change the answer |

**Option scoring.** For multiple choice the model is not asked to generate. We read the
next-token distribution after the prompt and take the probability of each option letter
("A" and " A" are summed), renormalised over the four letters. This is deterministic, needs
one forward pass, gives a confidence for calibration analysis, and records `letter_mass`,
the share of total probability on the letters, as a format-compliance check.

**Images** are downscaled so the longer side is at most 672 px, identically for all models.

## Labels

- Detection: first "yes"/"no" in the first six words, otherwise a small list of phrases;
  anything else is unparsed and counted as not flagged (the unparsed rate is reported).
- Open answers: colours, objects (with aliases and plurals) and relations are extracted
  from the answer; it is `image` if only the true value is mentioned, `text` if only the
  edited value is mentioned, else `other`.
- Attribution: the five labelled fields are read leniently (markdown, synonyms such as
  "colour" for attribute, "description" for text). Four components are scored on conflicts:
  wrong input is the text; error type matches; the wrong words name the edited value and
  not the true one; obviousness matches severity (S1 obvious, S2 moderate, S3 subtle).

## Metrics

| Metric | Definition |
|---|---|
| DetR, FFR | share of conflicts / controls flagged in `detect` |
| DetR_adj | DetR minus FFR |
| ResA_mcq, ResA_open | share of conflicts answered with the image value |
| TPR_mcq | text-aligned / (text-aligned + image-aligned) answers |
| ResA_mcq_if_perceived | ResA_mcq on conflicts the model answers correctly from the image alone |
| AttA, AttA_strict | mean attribution score / share with all four components right |
| R_given_A, R_given_notA | ResA_mcq split by strict attribution (coupling) |
| ResA_verbalize - ResA_mcq | effect of verbalising the error first |
| joint | distribution over detected / resolved / attributed on conflicts |

`ResA_mcq_if_perceived` separates arbitration failures from perception failures: a model
that cannot see the bus is red cannot be blamed for trusting a text that says it is blue.

## Precision

All reported models run on one Apple Silicon Mac (16 GB) on the Apple GPU, in float16
without quantisation (`configs/models_mac.toml`), so every row uses the same precision.
The 4-bit NVIDIA profile (`configs/models.toml`) is kept for reproduction on small GPUs
and is not used for reported results.

## Statistics (`faultline/analysis.py`, `scripts/analyze.py`)

- **Intervals.** 95% percentile bootstrap (2,000 replicates) that resamples images, not
  records: an image can contribute two records, and record-level resampling would make
  intervals too narrow. Differences (detection minus false flags, verbalize gain,
  coupling) are bootstrapped jointly from the same replicates.
- **Paired comparisons.** Exact two-sided McNemar tests on conflicts for verbalize vs.
  multiple choice, image-focus instruction vs. multiple choice, and open vs. multiple choice.
- **Comparable sets.** `--common` restricts every model to the records all models finished;
  `--first N` restricts to the first N records of the stratified order (phase 1 uses 1,000).
- **Outputs.** `results/metrics.csv`, `cells.csv`, `joint.csv`, `tests.csv`,
  `outcomes.jsonl` and `main_table.tex`. The final raw outputs
  (`outputs/<model>/core.jsonl`) are committed so every number can be recomputed.
