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

**Next**
- Generation pipeline with unit tests; run on real data on the Mac.
