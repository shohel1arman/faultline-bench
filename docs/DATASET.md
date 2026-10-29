# FaultLine dataset construction

This document specifies exactly how samples are built. Every rule here is implemented in
`faultline/` and checked by `tests/`. Thresholds live in `configs/dataset.toml`.

## Sources

| Source | Used for | License |
|---|---|---|
| GQA scene graphs, validation split (`val_sceneGraphs.json`) | objects, boxes, colour attributes | CC BY 4.0 |
| Visual Genome image metadata (`image_data.json`) | image URL, COCO id | CC BY 4.0 |
| Visual Genome images | the images shown to models | CC BY 4.0 |

GQA boxes are in Visual Genome pixel coordinates, so images are taken from Visual Genome.
Where a VG image is also a COCO image, its COCO id is recorded. Only the validation split
of GQA is used, to limit overlap with data that open VLMs are commonly trained on.

## Vocabulary

- **Objects.** 80 canonical nouns in 9 groups (vehicle, animal, food, furniture, tableware,
  electronics, sports, household, street). Scene-graph spellings are mapped to canonical
  forms through an alias table (e.g. `tv` to `television`, `bike` to `bicycle`).
- **Excluded on purpose.** People (no edits that change gender or identity); words with a
  second common sense (orange, mouse, bat, glass); hypernym pairs (taxi/car, duck/bird,
  mug/cup, desk/table, monitor/television) so that every object swap is unambiguously false.
- **Colours.** 15 colours in 7 perceptual families. Each colour has a list of *near*
  colours (e.g. red: orange, pink; white: silver, beige, gray). *Far* edits draw from seven
  basic colours outside the true colour's family and near list.
- **Relations.** Four 2D spatial relations: to the left of, to the right of, above, below.

## Which objects can be edited

A *target* must be in the vocabulary, be the only instance of its canonical name in the
image, and cover at least 2% of the image. An *anchor* (the reference object in a relation)
must have a unique name in the image, cover at least 1%, and be either in the vocabulary or
one of a few large scene objects (tree, building, window, door, fence, house, plant, flower).

### Spatial relations are recomputed from boxes

The scene-graph relation labels are not used. For a target t and anchor a, "t is to the left
of a" holds only if t ends before a starts by at least 2% of the image width, and the two
boxes overlap vertically by at least 30% of the shorter box. Above/below are defined the
same way on the other axis. A pair that satisfies none or more than one relation is
discarded, so diagonal and overlapping pairs never appear.

## Conflict types

Each sample changes exactly one span of an otherwise faithful description.

| Type | Faithful text | Edit | Question |
|---|---|---|---|
| attribute | The red bus is to the left of the tree. | colour of the target | What color is the bus? |
| object | The bus is to the left of the tree. | target noun replaced by a noun absent from the image | What is to the left of the tree? |
| relation | The red bus is to the left of the tree. | relation replaced by its antonym | Where is the bus relative to the tree? |

- Attribute edits require exactly one colour label on the target, and the target must not
  be an animal. The same rule decides when a colour may appear anywhere in a description
  (relation subject, filler sentence). Animal coats are often patterned, so a single
  scene-graph colour can be false for the image before any edit (a cow labelled only
  "white" that is black and white); found in the first smoke test.
- Object replacements must be absent from the image under all their aliases. The target
  colour is dropped from the object template so that implausible combinations (a "red
  elephant") do not reveal the edit from the text alone. Object samples also require that
  no other in-vocabulary object stands in the same relation to the anchor, so the question
  has a single answer in the image.
- Every description may contain one filler sentence about a different object
  ("The sign is white."), placed before or after the target sentence at random.

## Severity

Severity grades how hard the conflict is to verify from the image.

| Level | attribute / object | relation |
|---|---|---|
| S1 salient | far edit, target covers at least 10% of the image | gap at least 15% and both objects at least 3% |
| S2 moderate | far edit, target covers 2 to 10% | gap between 6% and 15% (or small objects) |
| S3 subtle | near edit (near colour, or object from the same group) | gap between 2% and 6% |

For attribution scoring, S1/S2/S3 map to obvious/moderate/subtle.

## Controls

Controls use the same templates and questions with the faithful text. Each control still
carries the edit it *would* have received (`foil`), which is one of its answer options, so
option sets are distributed the same way as for conflicts and the rate of choosing the foil
on clean input is measurable.

## Sampling

All valid candidate edits are collected per image. Each target object is used at most
once and each image contributes at most two samples. Quotas per cell: 400 main + 60
reserve (reserves replace items rejected during human verification); controls: 134 main
+ 20 reserve per question type.

One object usually qualifies for several cells. An object with a valid anchor yields an
object edit (far or near), a relation edit and often a colour edit, and any of these can
instead be rendered as a control. The cells therefore compete for the same objects, and
object/relation cells are the scarce ones because both need an anchor. Assignment is a
max-min fair allocation: cells grow one sample at a time, least-filled cell first; a
cell takes a free object if one exists, otherwise an augmenting path moves assigned
objects between cells to make room. A cell stops only when it cannot grow without
shrinking another. If all quotas fit, all are filled; if not, the least-filled cell is
as full as possible. Main quotas (controls included) are filled before reserves, and
within a cell the main/reserve split and the core subset are random.

A stratified **core** subset (200 per cell + 67 controls per question type, 2,001 items)
is evaluated for every model; the full main set is evaluated where compute allows.

## Record fields

| Field | Meaning |
|---|---|
| `id` | `fl_NNNNN`, assigned after a seeded shuffle so ids do not reveal the cell |
| `image_id`, `image_file`, `image_url`, `coco_id`, `width`, `height` | image |
| `is_conflict`, `conflict_type`, `question_type`, `severity`, `edit_kind` | design cell |
| `text` | description shown to the model |
| `text_clean` | faithful description (equals `text` for controls) |
| `edited_span`, `edited_span_char` | inserted span and its character range in `text` |
| `original_span`, `original_span_char` | replaced span and its range in `text_clean` |
| `question`, `mcq_options` | resolution question and four options |
| `answer_image`, `answer_text`, `foil` | image-consistent, text-consistent and edit answers |
| `target`, `anchor`, `relation`, `relation_gap` | objects and geometry behind the sample |
| `split`, `core` | main/reserve, and membership of the core subset |

## Invariants checked on every record

`faultline/schema.py::validate` rejects a record unless: `text` and `text_clean` differ
only inside the edited span; the character ranges point at the stated spans; the spans
equal the text- and image-consistent answers; options are four distinct strings containing
both answers and the foil; controls have identical text and answers.

## Known limitations

- Scene graphs are incomplete. A replacement object recorded as absent may still be
  visible; colour labels can be wrong. Human verification of a 500-item subset estimates
  the error rate.
- Descriptions are template-based. They are unambiguous but less varied than free-form
  captions.
- Above/below are image-plane relations, not 3D ones.
