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

## Round 1 (10 Nov 2026): not used for the paper

Both annotators finished (A1: 500 items, A2: 498; two A2 items lack an obviousness rating
and are skipped). Files: `annotation/results/round1/`, scores: `results/human_check_r1.json`.

The step-1 answers are not independent of the description. On relation conflicts, A1
picks the description's (wrong) relation in 78.6% of answered items and A2 in 41.9%,
against 5.7% and 17.8% for the two other wrong options together. Examples: fl_00927 (the
table is clearly right of the plant, boxes 384-482 vs 323-371 px; all eleven models
answer "right" from the photo alone; both annotators answer "left", as the description
says) and fl_00508 (the truck is right of the chair; same pattern). On relation
controls, where the description is true, the annotators mostly pick the label, so this is
not a misread left/right convention. The same pull is weaker but present for attributes
(A1: 16.5% description answer vs 6.5% for both other wrong options).

Cause: the page hid the description until step 1 was answered, but the step-1 buttons
stayed clickable afterwards, so an answer could be changed after reading the description.
The page does not log changes, so this cannot be confirmed per item, but no other
explanation fits answers that match one specific wrong option on clear photos.

Fix (round 2): the step-1 answer is final once given (buttons disabled, the page ignores
further clicks and keys, and records the time of the choice as `chose`), and the intro
says so. Round 2 uses 500 new records from the whole core set, none from round 1, so
annotators have not seen their descriptions:

```
bash scripts/send_annotation_site.sh 2     # on a machine with data/images
uv run python scripts/score_annotation.py annotation/results/round2/*.json --out results/human_check.json
```

The scorer now also prints, per annotator and conflict type, how often a step-1 answer
picks the description's option, which should be close to the rate for the other wrong
options when the protocol works.
