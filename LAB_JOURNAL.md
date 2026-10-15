# Lab journal

One entry per working day. Each entry: planned / done / problems / next.
Deadlines (AoE): registration Nov 10, paper Nov 16, supplementary Nov 23, 2026.

## 2026-10-14

**Planned**
- Set up repository and project structure.
- Decide data sources; build the conflict generation pipeline.

**Done**
- Repository scaffold, config, README.
- Data sources checked: GQA scene graphs (sceneGraphs.zip, 42.7 MB) and Visual Genome
  image metadata (image_data.json.zip, 17.6 MB); both CC BY 4.0 per their download pages.
- Generation pipeline: candidate extraction, geometric relation check, typed and graded
  edits, MCQ options, controls with foils, quota assignment, schema validation.
  17 unit tests on synthetic scene graphs; end-to-end smoke build on 3,000 synthetic images.
- Construction rules written up in docs/DATASET.md.

**Decisions**
- Images come from Visual Genome, not COCO val2017 directly. GQA bounding boxes are in VG
  pixel coordinates, so VG images keep boxes and images consistent. Many VG images are COCO
  images; the COCO id is kept in each record where available.
- Only the GQA *validation* scene graphs are used, to reduce overlap with GQA training
  data that open VLMs commonly see.
- Relations are restricted to 2D spatial relations (left of, right of, above, below) and are
  re-verified from bounding boxes, so every relation conflict is geometrically checkable.
- People are excluded as targets and as replacement objects (avoids gender/identity edits).
- No closed-model APIs. All evaluation runs on open models on Kaggle T4.

**Problems / risks**
- Scene graphs are incomplete: a replacement object marked absent might still be visible.
  The human verification pass checks this.
- S3 colour edits (e.g. red to orange) may be ambiguous for humans too; the verification
  pass decides whether S3 survives.

**Problems found while building**
- Colour distractors could be near the true colour (beige vs tan), making MCQ ambiguous.
  Distractors now exclude near colours of the true answer.
- Some object pairs were hypernyms (taxi/car, duck/bird, mug/cup, desk/table,
  monitor/television, sign/stop sign); removed from the vocabulary.

**Next**
- Run download + build on real GQA val data; check cell supply against the 400+60 quota.
- Read 50 generated samples with their images before fetching all images.

## 2026-10-15

**Planned**
- Evaluation harness for small open VLMs on the available hardware.

**Done**
- Compute changed: Kaggle and the shared lab server are not used. Runs are on a Windows PC
  with two RTX 2060 (6 GB each, no bfloat16) under WSL2. This replaces the Kaggle plan of
  2026-10-14.
- Harness: seven probes per record plus a verbalize-then-answer follow-up; multiple choice
  scored from option-letter probabilities instead of generation; outputs appended per
  record with automatic resume and retry of failed records.
- Parsing and scoring kept separate from running, so parser fixes never need a rerun.
- Model list (2B to 4B, four families, all fit one 6 GB GPU at 4-bit): Qwen2.5-VL-3B,
  Qwen3-VL-4B, InternVL3-2B, SmolVLM2-2.2B. Ids checked on the Hugging Face model pages.
- Checked against transformers 5.19 source: AutoModelForImageTextToText covers all four;
  processors take text and images as keywords; BitsAndBytesConfig arguments.
  The PyPI torch build (2.14.1, CUDA 13.0) includes sm_75 kernels for the RTX 2060.
- 35 unit tests, including a fake text-following model run end to end and a stub model
  for letter scoring. Real models could not be run here (no GPU, model hub not reachable),
  so the first real check is the 5-record smoke test.
- docs/EVAL.md (probes and metrics) and docs/RUNNING.md (WSL2 setup and long runs).

**Decisions**
- Same precision for every model: NF4 weights, float16 compute.
- Images downscaled to longer side 672 px for every model.
- Added `ResA_mcq_if_perceived`: resolution measured only on records the model gets
  right from the image alone, so perception failures are not counted as trusting the text.
- Added an instruction baseline (`mcq_focus`, "rely on the image") alongside verbalize.
- No 7B+ or 32B models: the paper states the 2B to 4B scope as a limitation.

**Problems / risks**
- Some models may overflow in float16; the harness records non-finite logits as errors.
- Detection and attribution parsing is heuristic; unparsed rates are reported and a sample
  of outputs will be checked by hand.

**Next**
- WSL2 setup, dataset build, 5-record smoke test per model; record seconds per record.
