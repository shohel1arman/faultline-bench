# Human check

Two annotators check the same 500 core records, to measure how often FaultLine's labels
and conflicts are valid and whether the severity grades match human judgement.

## Sample

`faultline/annotation.py:sample`: 50 records from each of the nine conflict cells
(object/attribute/relation x S1/S2/S3) and 50 faithful controls (17/17/16 by question
type), all from the first 1,000 records of the stratified order, so every model has them.
Seed 0; the list is in `annotation/sample_ids.json`. Records are shown in a shuffled
order with no cell, edit or answer information.

## Protocol (per record)

1. **Answer from the photo.** The question and the four options the models saw, without
   the description. "Can't tell" if the photo does not decide it, the question could mean
   several objects, or more than one option is true.
2. **Does the description match the photo?** Yes / no / can't tell. The description is the
   record's text: edited for conflicts, faithful for controls; annotators do not know which.
3. **If no: how obvious is the mistake?** Obvious / moderate / subtle.

Left, right, above and below are as seen in the photo. Answers stay in the browser and are
exported to a JSON file; annotators use IDs (A1, A2), not names.

## Measures (`scripts/score_annotation.py`)

- Picks the labelled image answer: human accuracy from the image alone, per cell.
- Description judged correctly: edited text judged wrong, faithful text judged right.
- Records where both annotators agree on a different answer than the label, agree that an
  edited description matches, or agree that a faithful description is wrong: label
  problems; their share gives the estimated valid rate.
- Agreement: percent and Cohen's kappa for the answer, the description and obviousness.
- Severity: Spearman correlation between S1/S2/S3 and rated obviousness on conflicts.

## Commands

```
uv run python scripts/make_annotation.py --images data/images
```

This writes `annotation/faultline_check.html` (one file, photos included, about 28 MB).
Open it in Chrome or Firefox; it works offline.

A hosted version (`scripts/make_annotation_site.py`, built from that file so the photos
are identical) runs from one link in any browser. When the annotator is signed in with
write access it saves every answer to their account as they go (store path
`ann/<viewer id>/items/<record id>`; only the owner reads all annotators' answers), so
they can switch computers; otherwise it keeps answers in the browser and offers Export,
like the single file. When both annotators have finished, put
their exported files in `annotation/results/` and run:

```
uv run python scripts/score_annotation.py annotation/results/*.json
```
