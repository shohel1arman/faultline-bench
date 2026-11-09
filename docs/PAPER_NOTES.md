# Paper notes

Working notes for writing the CVPR 2027 paper: what was built, why, what has been found,
and what still needs checking. Numbers are copied from `scripts/analyze.py` output; rerun
it before quoting anything in the paper. Last updated 2026-11-09.

## One-line thesis

On identical image-text conflicts, small open VLMs usually notice that the text is wrong,
yet still answer from the text, and can point at the wrong words without saying what kind
of error it is or which input is wrong: detection, resolution and attribution come apart.

## Dataset (FaultLine v0, `data/faultline_v0.jsonl`)

- 4,062 records from GQA validation scene graphs, images from Visual Genome (CC BY 4.0).
  Core subset used for evaluation: 2,001 records = 9 cells (object/attribute/relation x
  S1/S2/S3) x 200 conflicts + 3 x 67 clean controls. Main set: attribute cells 400 each;
  object and relation cells 379 each (limited by the two-samples-per-image cap).
- One edited span per record; text and faithful text differ only inside it (checked for
  every record). Relations are recomputed from bounding boxes, not taken from labels.
- Exclusions and why (each is a sentence in the paper):
  - people never edited or used (identity edits);
  - hypernym pairs removed (taxi/car, duck/bird, mug/cup, desk/table, monitor/TV);
  - ambiguous words removed (orange, mouse, bat, glass);
  - animal colours never edited or stated: coats are patterned, labels unreliable (a cow
    labelled "white" was black and white in the first smoke test);
  - colour distractors exclude near colours of the true answer.
- Assignment of objects to cells is a max-min fair allocation with augmenting paths;
  the greedy fill it replaced left object S3 at 77 of 460 and object controls at 0.
- Records are evaluated in a stratified order, so the first 1,000 are balanced across
  cells; every model covers the same first 1,000 before any model goes further.

## Models and setup

- Decision 2026-11-02: every reported model runs on one NVIDIA DGX Spark (GB10, 128 GB
  unified memory) in bfloat16, no quantisation (`configs/models_spark.toml`, outputs in
  `outputs/spark/`). Torch 2.14.1+cu130, transformers 5.22. Images downscaled to longer
  side 672 px for every model. Greedy decoding.
- Eleven models, run smallest first:
  - Qwen3-VL 2B, 4B, 8B, 32B and 30B-A3B (mixture of experts, about 3B active). This is
    the scaling ladder within one family, the paper's main new question: does the gap
    between noticing a conflict and answering from the image close with scale?
  - Qwen2.5-VL 3B, 7B, 32B; InternVL3 2B, 8B; SmolVLM2 2.2B.
  - Not run: InternVL3-38B (time); 72B models (do not fit in bfloat16).
- The four small models also ran on a Mac mini (16 GB, Apple GPU, float16). That run is
  the cross-platform check (agreement of answers across hardware and precision), not a
  table row. The InternVL3-2B numbers below are from the Mac and will be replaced by the
  Spark run.
- No closed models (no API budget): state as a limitation.
- Multiple choice scored from next-token letter probabilities (no generation).

## Probes (prompt version p1.1, `faultline/prompts.py`)

image_only, text_only (controls); detect (yes/no + reason); open; mcq; mcq_focus
("rely on the image"); attribute (5-line report); verbalize (attribute exchange, then the
question in a new turn); delay and check_silent (same turn structure as verbalize with a
neutral "OK." reply; added p1.1 to separate the verbalize effect into structure, being
asked to check, and the model's own report).

## Statistics

95% percentile bootstrap resampling images (2,000 replicates); exact McNemar tests for
paired probe comparisons. Detection is reported minus the false-flag rate on controls.

## Results so far

### InternVL3-2B, full core set, Mac float16 run (1,800 conflicts, 201 controls)

| Metric | Value [95% CI] |
|---|---|
| Perceives the fact from the image alone | 73.7 [71.6, 75.7] |
| Reads the fact from the text alone | 99.9 |
| Flags conflict / false flags on controls | 71.1 / 19.9 |
| Detection, adjusted | 51.2 [45.2, 56.8] |
| Answers from image (mcq) | 16.4 [14.7, 18.2] |
| Answers from text (mcq); text preference ratio | 82.6; 83.4 |
| Answers from image when it perceived correctly | 21.9 [19.7, 24.4] |
| Open-ended answer from image | 11.6 |
| With "rely on the image" | 21.6 (+5.2; 146 vs 53, p = 3e-11) |
| Verbalize then answer | 52.7 (+36.3; 672 vs 19, p < 1e-100) |
| Attribution: wrong words located | 72.1 [70.0, 74.3] |
| Attribution: error type / wrong input / severity | 40.1 / 19.4 / 20.4 |
| Attribution: all four correct | 0.8 |
| Resolved if detected vs not | 20.4 vs 6.5 (coupling +13.9 [10.8, 17.0]) |
| Resolved if wrong words located vs not | 18.6 vs 10.7 (+7.8 [4.2, 11.1]) |
| Controls answered correctly | 100.0 |

Joint pattern on conflicts: detected but answered from text, 56%; detected and resolved,
14%; neither, 27%; resolved without detecting, 2%.

By cell (perceive / detect / resolve):

| Cell | S1 | S2 | S3 |
|---|---|---|---|
| attribute | 85 / 90 / 33 | 87 / 89 / 28 | 83 / 67 / 12 |
| object | 95 / 97 / 19 | 95 / 91 / 15 | 93 / 83 / 24 |
| relation | 54 / 44 / 7 | 41 / 48 / 6 | 32 / 34 / 5 |

Readings: detection falls with severity for attribute and object, which supports the
severity design; relations are hard to perceive at all (near the 25% guessing rate at
S3), so relation results must be read through the perceived-only metric.

### Final: Spark bfloat16, all 2,001 records, eleven models (2026-11-09)

1,800 conflicts and 201 controls per model; all runs complete, 0 errors. Tables and
figures: `results/spark/` and `results/figures/` (`scripts/analyze.py`,
`scripts/make_figures.py`). 95% image-clustered bootstrap.

| Model | Perceive | Flags | False flags | Det. adj. | From image | If perceived | Open | Focus | Delay | Check | Verbalize |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-2B | 78.1 | 76.3 | 16.9 | 59.4 | 39.4 | 47.4 | 34.2 | 45.5 | 27.9 | 26.3 | 37.1 |
| Qwen3-VL-4B | 83.3 | 94.8 | 37.3 | 57.5 | 41.3 | 47.0 | 38.0 | 76.0 | 69.8 | 71.9 | 80.0 |
| Qwen3-VL-8B | 82.4 | 96.3 | 44.8 | 51.6 | 54.3 | 62.0 | 38.8 | 84.2 | 69.3 | 78.4 | 82.8 |
| Qwen3-VL-30B-A3B | 82.2 | 95.2 | 39.8 | 55.4 | 59.0 | 66.5 | 35.1 | 79.4 | 63.0 | 64.3 | 78.8 |
| Qwen3-VL-32B | 85.7 | 97.7 | 53.7 | 43.9 | 43.1 | 48.2 | 40.7 | 86.3 | 64.3 | 79.2 | 85.9 |
| Qwen2.5-VL-3B | 76.8 | 94.6 | 47.3 | 47.3 | 18.9 | 24.4 | 19.6 | 26.0 | 24.8 | 20.3 | 23.4 |
| Qwen2.5-VL-7B | 79.5 | 98.2 | 67.2 | 31.1 | 65.0 | 73.4 | 39.2 | 84.8 | 71.1 | 75.2 | 80.6 |
| Qwen2.5-VL-32B | 82.4 | 93.8 | 43.3 | 50.5 | 48.8 | 56.2 | 35.4 | 80.3 | 70.6 | 77.4 | 82.0 |
| InternVL3-2B | 73.6 | 72.0 | 19.9 | 52.1 | 16.9 | 22.7 | 11.9 | 22.4 | 30.9 | 39.8 | 52.9 |
| InternVL3-8B | 82.3 | 93.9 | 35.8 | 58.1 | 49.4 | 55.6 | 33.0 | 72.9 | 74.3 | 75.5 | 75.2 |
| SmolVLM2-2.2B | 68.6 | 84.4 | 62.2 | 22.3 | 7.8 | 11.4 | 10.7 | 9.6 | 13.7 | 18.1 | 13.3 |

| Model | Structure | Check | Report | Words located | Type | Wrong input | Severity | All four | Coupling D | Controls |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-2B | -11.6 | -1.6 | +10.8 | 12.3 | 37.2 | 0.0 | 7.8 | 0.0 | +28.1 | 97.0 |
| Qwen3-VL-4B | +28.6 | +2.1 | +8.1 | 87.3 | 82.2 | 1.1 | 32.9 | 0.5 | +39.1 | 99.0 |
| Qwen3-VL-8B | +15.0 | +9.1 | +4.4 | 87.9 | 77.5 | 28.2 | 31.9 | 9.3 | +46.9 | 97.5 |
| Qwen3-VL-30B-A3B | +4.0 | +1.3 | +14.5 | 77.6 | 71.8 | 3.4 | 30.1 | 1.4 | +53.4 | 98.5 |
| Qwen3-VL-32B | +21.2 | +14.9 | +6.7 | 93.3 | 78.9 | 54.2 | 33.3 | 16.4 | +39.3 | 98.0 |
| Qwen2.5-VL-3B | +5.9 | -4.6 | +3.2 | 85.9 | 53.6 | 67.0 | 30.2 | 10.6 | +20.0 | 100.0 |
| Qwen2.5-VL-7B | +6.1 | +4.1 | +5.4 | 81.5 | 67.7 | 21.1 | 28.4 | 5.7 | +56.6 | 97.0 |
| Qwen2.5-VL-32B | +21.8 | +6.8 | +4.6 | 82.7 | 75.6 | 10.1 | 32.9 | 2.6 | +44.5 | 99.0 |
| InternVL3-2B | +14.0 | +8.9 | +13.1 | 70.8 | 39.9 | 0.0 | 20.7 | 0.0 | +14.4 | 100.0 |
| InternVL3-8B | +24.8 | +1.2 | -0.3 | 11.8 | 10.1 | 3.8 | 26.2 | 1.3 | +41.9 | 99.0 |
| SmolVLM2-2.2B | +5.8 | +4.4 | -4.7 | 66.9 | 0.2 | 0.0 | 1.1 | 0.0 | +5.9 | 100.0 |

- Qwen3-VL-2B: adjusted detection 59.4 [53.6, 64.5]; from image 39.4 [37.2, 41.8]; told to rely on image 45.5 [43.3, 47.9]
- Qwen3-VL-4B: adjusted detection 57.5 [50.7, 64.4]; from image 41.3 [38.9, 43.7]; told to rely on image 76.0 [73.9, 78.0]
- Qwen3-VL-8B: adjusted detection 51.6 [44.5, 58.4]; from image 54.3 [52.0, 56.8]; told to rely on image 84.2 [82.4, 85.9]
- Qwen3-VL-30B-A3B: adjusted detection 55.4 [48.2, 62.2]; from image 59.0 [56.6, 61.3]; told to rely on image 79.4 [77.4, 81.3]
- Qwen3-VL-32B: adjusted detection 43.9 [36.1, 51.1]; from image 43.1 [40.8, 45.5]; told to rely on image 86.3 [84.7, 88.0]
- Qwen2.5-VL-3B: adjusted detection 47.3 [40.1, 54.2]; from image 18.9 [17.1, 20.9]; told to rely on image 26.0 [24.0, 28.2]
- Qwen2.5-VL-7B: adjusted detection 31.1 [24.2, 37.8]; from image 65.0 [62.8, 67.3]; told to rely on image 84.8 [83.1, 86.6]
- Qwen2.5-VL-32B: adjusted detection 50.5 [43.5, 57.3]; from image 48.8 [46.4, 51.2]; told to rely on image 80.3 [78.4, 82.2]
- InternVL3-2B: adjusted detection 52.1 [46.3, 57.8]; from image 16.9 [15.2, 18.8]; told to rely on image 22.4 [20.6, 24.4]
- InternVL3-8B: adjusted detection 58.1 [51.1, 65.0]; from image 49.4 [47.0, 51.8]; told to rely on image 72.9 [70.8, 75.0]
- SmolVLM2-2.2B: adjusted detection 22.3 [15.1, 29.0]; from image 7.8 [6.6, 9.2]; told to rely on image 9.6 [8.3, 11.1]

Readings for the paper:

- The gap holds for all eleven: every model flags 72-98% of conflicts and still answers
  from the text on many; "noticed but answered from the text" is the largest outcome for
  6 of 11 models (figure `outcomes`).
- Scale helps up to 7-8B in every family (Qwen3-VL 39 -> 41 -> 54; Qwen2.5-VL 19 -> 65;
  InternVL3 17 -> 49) and the 30B-A3B mixture of experts reaches 59, but both dense 32B
  models fall back (Qwen3-VL 54 -> 43, Qwen2.5-VL 65 -> 49). Told to rely on the image,
  the same 32B models reach 86 and 80. Larger models are able to use the image and do so
  when asked, but do not choose it by default (figure `scaling`).
- False flags on faithful descriptions grow with scale in Qwen3-VL (17 -> 54), so
  false-alarm-corrected detection does not improve with scale (Qwen3-VL 59 -> 44).
- Turn structure explains most of the verbalize gain; the model's own error report adds
  -5 to +15 points (Qwen3-VL-2B is the exception: the delay lowers its image answers).
- Attribution parts come apart: models locate the wrong words (67-93% for most) but
  rarely say which input is wrong (best: Qwen2.5-VL-3B 67, Qwen3-VL-32B 54); InternVL3-8B,
  Qwen3-VL-4B and Qwen3-VL-2B write "Error: no" in most reports despite flagging most
  conflicts in the detection probe.
- Per cell (figure `cells`): image answers fall from S1 to S3 for attributes in every
  model; object conflicts are the hardest to resolve for the Qwen3-VL models; relations
  are near zero for SmolVLM2 and InternVL3-2B.
- Cross-platform check: see `results/platform_agreement.csv` (98.6-99.5% identical
  multiple-choice answers between the Mac float16 and Spark bfloat16 runs).

### Further analyses (2026-11-09, `scripts/extra_analysis.py`)

- Arbitration failure: among conflicts where the model perceives the fact from the image
  alone and flags the conflict, the share it still answers from the text: 26.1
  (Qwen2.5-VL-7B) to 87.9 (SmolVLM2); Qwen3-VL-32B 51.2, Qwen2.5-VL-32B 41.6.
- Report contradicts detection (attribution says "Error: no" on a flagged conflict):
  Qwen3-VL-4B 99.7, InternVL3-8B 85.5, Qwen3-VL-2B 71.7, Qwen2.5-VL-7B 71.0,
  InternVL3-2B 51.9; below 15 for the rest (Qwen3-VL-32B 0.3).
- Paired size comparisons on the same records (image-clustered CI, exact McNemar):
  Qwen3-VL-8B -> 32B -11.2 [-13.4, -8.9]; Qwen2.5-VL-7B -> 32B -16.2 [-18.6, -13.9];
  Qwen3-VL-30B-A3B -> 32B -15.9; all p < 1e-10. The 32B drop is not noise.
- Severity, pooled over the eleven models, S1 minus S3: detection +7.7 [+5.6, +9.8],
  resolution +9.3 [+5.7, +12.7]; attributes strongest (+13.3 / +19.0); object
  resolution shows no severity effect (-0.8 [-6.4, +4.7]). Report as support for the
  grading with the object exception.
- Steerability (with "rely on the image" minus without): +1.8 (SmolVLM2) to +43.2
  (Qwen3-VL-32B).

### Paper assets (regenerate with the commands below)

```
python scripts/analyze.py outputs/spark/*/core.jsonl --out results/spark
python scripts/extra_analysis.py
python scripts/compare_platforms.py
python scripts/make_tables.py
python scripts/make_figures.py
```

- Tables (`results/tables/`, booktabs; dataset needs `\usepackage{array}`): `dataset`,
  `main`, `probes` (asterisk = McNemar p < 0.001), `pairs`, `severity`; supplement:
  `cells` (perceive/detect/resolve per cell), `platform`.
- Figures (`results/figures/`, PDF and PNG): `scaling` (three panels vs size),
  `outcomes` (joint outcome shares), `probes` (image answers under each probe),
  `cells` (heatmap per cell). Still to make: a teaser with one real example (needs the
  photos from the Spark or the annotation page).

### Four small models, full core set, Mac float16 run (cross-platform check)

`scripts/analyze.py` on all 2,001 records of each model (1,800 conflicts, 201 controls).
These are the Mac numbers; the paper's rows will come from the Spark run. 95% CIs.

| Metric | InternVL3-2B | Qwen2.5-VL-3B | Qwen3-VL-4B | SmolVLM2-2.2B |
|---|---|---|---|---|
| Perceives fact from image alone | 73.7 | 77.1 | 83.0 | 68.7 |
| Flags conflict | 71.1 | 94.2 | 94.9 | 83.3 |
| False flags on controls | 19.9 | 46.3 | 36.8 | 60.7 |
| Detection, adjusted | 51.2 [45.2, 56.8] | 48.0 [40.6, 55.0] | 58.1 [51.1, 64.9] | 22.6 [15.6, 29.5] |
| Answers from image (mcq) | 16.4 [14.7, 18.2] | 18.1 [16.3, 20.0] | 41.1 [38.7, 43.5] | 8.1 [6.8, 9.5] |
| Answers from text (mcq) | 82.6 | 81.6 | 57.1 | 91.7 |
| From image when perceived | 21.9 | 23.3 | 47.3 | 11.7 |
| Detected but answered from text (D--) | 56% | 70% | 54% | 76% |
| Open-ended from image | 11.6 | 19.4 | 38.1 | 10.6 |
| "Rely on the image" (gain) | 21.6 (+5.2) | 25.8 (+7.8) | 76.1 (+34.9) | 9.4 (+1.3) |
| Question in a later turn, "OK." reply (delay) | 30.2 | 25.0 | 70.4 | 13.8 |
| Same, after the check prompt (check_silent) | 39.6 | 20.7 | 72.1 | 18.3 |
| After its own error report (verbalize) | 52.7 | 23.9 | 80.1 | 12.7 |
| Verbalize gain = structure + check + report | 36.3 = 13.8 + 9.4 + 13.1 | 5.9 = 6.9 - 4.3 + 3.3 | 39.0 = 29.3 + 1.6 + 8.1 | 4.6 = 5.7 + 4.5 - 5.6 |
| Wrong words located | 72.1 | 86.6 | 87.8 | 69.1 |
| Error type / wrong input / severity | 40.1 / 19.4 / 20.4 | 54.1 / 60.7 / 30.6 | 82.0 / 0.8 / 32.9 | 23.8 / 36.2 / 0.9 |
| All four attribution parts | 0.8 | 10.3 | 0.5 | 0.0 |
| Coupling: resolved if detected minus not | +13.9 | +18.1 | +38.7 | +6.1 |
| Controls answered correctly | 100.0 | 100.0 | 99.0 | 100.0 |
| Unparsed attribution reports | 0.0 | 0.0 | 0.0 | 6.3 |

Readings (to be confirmed on the Spark run):

- The thesis holds for all four: models flag the conflict on 71-95% of conflicts and
  still answer from the text; "detected but answered from text" is the most common
  pattern for every model (54-76% of conflicts).
- Raw detection is inflated by false flags (20-61% of faithful controls are flagged), so
  adjusted detection is the honest number; SmolVLM2 flags almost everything.
- The verbalize effect is mostly turn structure, not the model's own report. Moving the
  question to a later turn behind a neutral "OK." accounts for most of the gain for
  Qwen3-VL-4B (+29.3 of +39.0) and about a third for InternVL3-2B. The report itself adds
  +13.1, +8.1, +3.3 and -5.6. Asking the model to check without its report helps
  InternVL3-2B (+9.4) and hurts Qwen2.5-VL-3B (-4.3). The earlier InternVL-only reading
  ("verbalizing helps by +36") must not be used; this decomposition replaces it.
- Qwen3-VL-4B is the most image-reliant and the most steerable (+34.9 from one sentence).
- Attribution parts come apart: models locate the wrong words (69-88%) but rarely say
  which input is wrong. Qwen3-VL-4B writes "Error: no / Wrong input: none / How obvious:
  none" on 1,795 of 1,800 conflicts while naming the right error type and words, so its
  report contradicts its own detection answer (95% flagged). Report as a finding about
  self-report consistency; do not change the prompt after seeing it (prompts stay p1.1).

By cell (perceive / detect / resolve):

| Model | Type | S1 | S2 | S3 |
|---|---|---|---|---|
| Qwen2.5-VL-3B | attribute | 87 / 94 / 32 | 84 / 94 / 30 | 86 / 78 / 13 |
| | object | 93 / 99 / 16 | 94 / 100 / 10 | 94 / 96 / 12 |
| | relation | 62 / 98 / 20 | 50 / 96 / 16 | 44 / 94 / 12 |
| Qwen3-VL-4B | attribute | 88 / 93 / 64 | 89 / 96 / 57 | 86 / 82 / 39 |
| | object | 94 / 100 / 28 | 94 / 100 / 25 | 91 / 96 / 34 |
| | relation | 76 / 98 / 48 | 68 / 96 / 39 | 62 / 94 / 35 |
| SmolVLM2-2.2B | attribute | 87 / 94 / 20 | 84 / 96 / 8 | 76 / 87 / 8 |
| | object | 87 / 96 / 8 | 92 / 94 / 10 | 86 / 92 / 18 |
| | relation | 37 / 66 / 0 | 34 / 66 / 0 | 34 / 59 / 0 |

Relation detection near 95% with perception near 50% (Qwen models) matches the high
false-flag rate on relation controls (46-52%): relation sentences are flagged whether or
not they are wrong. Attribute detection falls at S3 for every model, supporting the
severity grading; object severity barely changes detection.

## Decisions to report honestly

- Strict attribution (all four parts) is too rare to condition on (14 records for
  InternVL3-2B), so attribution-resolution coupling uses "wrong words located".
- Two parser fixes made after seeing real outputs (scoring only, no rerun): the short
  label "Obvious:" is accepted, and only the named words (quoted part or text before an
  explanation) count for the span. Both are tested on real outputs.
- The verbalize gain was found before its controls existed; the controls were added
  before the remaining models ran, and InternVL3-2B got them in an extra pass. With the
  controls, most of the gain is turn structure (see the table above); say so plainly.
- Third parser fix (2026-11-04, scoring only): a reply of exactly five unlabeled lines is
  read by position. Qwen3-VL-2B answers the attribution format this way (32% of its first
  539 records were unparsed). No existing Mac or Spark result changes: no other model
  produced such a reply.
- Fourth parser fix (2026-11-06, scoring only): a field that copies the option list
  ("Wrong input: text, image, or none") is no answer; before, its first word counted.
  Changes 1,190 Qwen3-VL-2B, 387 InternVL3-2B, 597 + 1,068 SmolVLM2 and 42 Qwen2.5-VL-7B
  fields; every changed value was checked to be a copied list.
- Prompts are frozen at p1.1 for the Spark run, including the attribution prompt that
  Qwen3-VL-4B answers inconsistently, so the Mac and Spark runs stay comparable and no
  prompt is tuned on results.
- Models often answer "Wrong input: image" while quoting the text's words; scored as
  wrong. Possible prompt ambiguity; discuss in limitations.
- Object conflicts are often labelled "relation" by models (the sentence is a relation
  sentence); type accuracy is strict.

## Limitations to state

Largest model 32B (no 72B); one hardware platform for reported rows; template descriptions; scene-graph label
noise (estimated by the human check); 2D image-plane relations only; heuristic parsing
(unparsed rates reported, 0% so far for InternVL3-2B); no closed models.

## Related work: must be read before citing

Found by search, not yet read: CrossCheck-Bench (arXiv 2511.21717; closest; detection
and compositional reasoning, no attribution or severity), CMC-Bench (MAGMaR 2026; RAG
conflicts), DO-Bench (arXiv 2604.22822; "attributable" object hallucination, different
sense), CDH-Bench (arXiv 2603.27982), Mixed Signals (arXiv 2504.08974), HallusionBench.
From the outline, to verify: Deng et al. CVPR 2025, CLASH, MC2, Nguyen et al. 2025.

## Open items

- Mac queue (cross-platform check): phase 1 done for all four models on Nov 1;
  everything about Nov 3, 05:00.
- Spark runs: set up Nov 2; estimated 3.5 to 5.5 days for all eleven models (estimate from
  memory bandwidth, not measured; the 32B models dominate). Freeze all runs by Nov 11.
- Cross-platform agreement analysis (Mac float16 vs Spark bfloat16, four small models).
- Scaling figure: metric vs parameters for Qwen3-VL, with the other families as points.
- Human check of 500 samples by two annotators (A1, A2): page and scoring ready
  (`docs/ANNOTATION.md`); about 3 to 4 hours per annotator. Gives human accuracy per
  cell, the estimated valid rate, agreement (kappa) and the severity correlation.
- OpenReview profiles for all authors before the Nov 10 registration.
