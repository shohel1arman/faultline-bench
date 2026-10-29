# FaultLine

FaultLine is a benchmark for asking a simple question of vision-language models (VLMs):
when an image and its accompanying text disagree, does the model **notice**, does it
**answer from the right modality**, and can it **say what is wrong**?

Every sample pairs a real image with a short description in which exactly one fact has
been altered. The same sample is probed three ways:

| Level | Probe | Measures |
|---|---|---|
| Detection | "Does the text accurately describe the image?" | whether the conflict is noticed |
| Resolution | a question about the altered fact, no hint of a conflict | whether the answer follows the image or the text |
| Attribution | "Which input is wrong, what kind of error, which words, how obvious?" | whether the model can localise and characterise the conflict |

Conflicts are typed (object, attribute, relation) and graded by perceptual difficulty
(S1 salient, S2 moderate, S3 subtle). Clean controls with faithful text measure false
alarms.

## Status

Work in progress. Dataset construction and evaluation harness are under active development.

## Layout

```
faultline/          library code (dataset construction, schema, evaluation)
scripts/            command-line entry points
configs/            dataset and evaluation configs
tests/              unit tests (run on synthetic fixtures, no downloads needed)
docs/DATASET.md     construction rules, in enough detail to reproduce the data
docs/EVAL.md        probes, parsing and metrics
docs/RUNNING.md     environment setup and running long evaluations
data/               generated dataset files (raw downloads and images are not tracked)
LAB_JOURNAL.md      dated record of what was planned, done and found
```

## Quickstart

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). Dataset construction uses only
the standard library; evaluation needs an NVIDIA GPU or an Apple Silicon Mac.

```bash
uv sync --extra eval --group dev
uv run pytest

uv run python scripts/download_data.py --out data/raw
uv run python scripts/build_dataset.py --config configs/dataset.toml
uv run python scripts/fetch_images.py --dataset data/faultline_v0.jsonl --out data/images

uv run python scripts/run_eval.py --model qwen25vl-3b --device cuda:0 --limit 5
uv run python scripts/score_eval.py outputs/qwen25vl-3b/core_limit5.jsonl --show 2
```

See docs/RUNNING.md for setup on macOS (Apple Silicon) or Windows and for long runs, and docs/EVAL.md for
the probes and metrics.

## Data sources and licenses

- Scene graphs: GQA (Hudson and Manning, 2019), CC BY 4.0
- Images and image metadata: Visual Genome (Krishna et al., 2017), CC BY 4.0

FaultLine annotations are derived from these sources and will be released under CC BY 4.0.
Code is released under the MIT license.
