# Paper notes

Working notes for writing the CVPR 2027 paper: what was built, why, what has been found,
and what still needs checking. Numbers are copied from `scripts/analyze.py` output; rerun
it before quoting anything in the paper. Last updated 2026-11-02.

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

### Observations from smoke tests (3 records each)

- SmolVLM2 often does not follow the attribution format (echoes the option list).
- Qwen2.5-VL-3B answered from the text on all three records even after verbalizing.
- Qwen3-VL-4B switched to the image answer with the focus instruction and after
  verbalizing on the shown record. Anecdotes only; wait for the full runs.

## Decisions to report honestly

- Strict attribution (all four parts) is too rare to condition on (14 records for
  InternVL3-2B), so attribution-resolution coupling uses "wrong words located".
- Two parser fixes made after seeing real outputs (scoring only, no rerun): the short
  label "Obvious:" is accepted, and only the named words (quoted part or text before an
  explanation) count for the span. Both are tested on real outputs.
- The verbalize gain was found before its controls existed; the controls were added
  before the remaining models ran, and InternVL3-2B gets them in an extra pass.
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
- Human check of 500 samples by two annotators (agreement, error rate per cell).
- OpenReview profiles for all authors before the Nov 10 registration.
