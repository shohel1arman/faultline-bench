# Running FaultLine

Two supported setups: an Apple Silicon Mac (models in full precision on the Apple GPU) and
a Windows PC with NVIDIA GPUs (4-bit models). A model's results must come from one setup
only. Commands below are one per block; run them one at a time.

## macOS (Apple Silicon)

### Setup

If `uv --version` fails, install uv and open a new terminal:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone (or, if already cloned, `cd ~/faultline-bench && git pull`):

```
cd ~ && git clone https://github.com/shohel1arman/faultline-bench.git && cd faultline-bench
```
```
uv sync --extra eval --group dev
```
```
uv run pytest -q
```
```
uv run python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```
```
sysctl -n hw.memsize
```

The torch line should end with `True`. The last command prints memory in bytes
(17179869184 = 16 GB). With 16 GB, run one model at a time and close other large apps.

### Dataset

```
uv run python scripts/download_data.py --out data/raw
```
```
uv run python scripts/build_dataset.py --config configs/dataset.toml
```
```
uv run python scripts/fetch_images.py --dataset data/faultline_v0.jsonl --out data/images
```

### Smoke test

```
uv run python scripts/run_eval.py --model internvl3-2b --device mps --models-config configs/models_mac.toml --limit 5
```
```
uv run python scripts/score_eval.py outputs/internvl3-2b/core_limit5.jsonl --show 2
```

### Full runs

Run in the Terminal app, not inside VS Code, so closing VS Code does not stop the run.
`caffeinate -i` keeps the Mac awake for as long as the command runs:

```
cd ~/faultline-bench && caffeinate -i uv run python scripts/run_eval.py --model internvl3-2b --device mps --models-config configs/models_mac.toml --subset core
```

Run one model at a time on the Apple GPU. Rerunning the same command resumes a stopped run.

## Windows (NVIDIA GPUs)

Target machine: Windows with two RTX 2060 (6 GB each). Everything runs natively in
PowerShell; no WSL.

### 1. One-time setup

Install Git and uv (skip any you already have), then close and reopen the terminal:

```
winget install --id Git.Git -e
```
```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Clone into a short local folder. Avoid Desktop and Documents if they are synced by
OneDrive: the dataset images and model outputs are large.

```
mkdir C:\work; cd C:\work; git clone https://github.com/shohel1arman/faultline-bench.git; cd faultline-bench
```

Install dependencies. uv downloads a matching Python and the CUDA build of torch
(about 3 GB the first time):

```
uv sync --extra eval --group dev
```
```
uv run pytest -q
```

Check the GPUs and 4-bit support:

```
uv run python -c "import torch; print(torch.__version__, torch.cuda.device_count(), torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"
```
```
uv run python -c "import bitsandbytes; print(bitsandbytes.__version__)"
```

The first should print a `+cu130` version, `2`, `NVIDIA GeForce RTX 2060`, and a list
containing `sm_75`. If it prints `+cpu` or `0`, torch came from the wrong index: run
`uv sync --extra eval --group dev --reinstall-package torch`.

Commit the `uv.lock` file that the first sync created, so every run uses the same versions.

### 2. Build the dataset

```
uv run python scripts/download_data.py --out data/raw
```
```
uv run python scripts/build_dataset.py --config configs/dataset.toml
```
```
uv run python scripts/fetch_images.py --dataset data/faultline_v0.jsonl --out data/images
```

### 3. Smoke test (always before a long run)

```
uv run python scripts/run_eval.py --model qwen25vl-3b --device cuda:0 --limit 5
```
```
uv run python scripts/score_eval.py outputs/qwen25vl-3b/core_limit5.jsonl --show 2
```

Check: no errors, outputs are readable English, `letter_mass` is not tiny, and the
reported seconds per record.

### 4. Full runs

Run one model per GPU, each in its own PowerShell window. Open these windows from the
Start menu, not inside VS Code: closing VS Code closes its terminals and stops the run.

Window 1:
```
cd C:\work\faultline-bench; uv run python scripts/run_eval.py --model qwen25vl-3b --device cuda:0 --subset core
```

Window 2:
```
cd C:\work\faultline-bench; uv run python scripts/run_eval.py --model internvl3-2b --device cuda:1 --subset core
```

A stopped run resumes where it left off when the same command is run again. The output
file in `outputs/<model>/` is the record of the run; the window only shows progress.

GPU 0 also drives the display (about 0.7 GB in use), so give the larger models
(`qwen3vl-4b`) GPU 1.

### Keep the PC running

In PowerShell (as administrator), disable sleep and hibernation on mains power:

```
powercfg /change standby-timeout-ac 0
```
```
powercfg /change hibernate-timeout-ac 0
```

Also pause Windows Update for the run period (Settings, Windows Update, Pause updates),
otherwise an automatic restart can stop a run overnight. A restart loses nothing already
written; rerun the same commands to resume.

### Notes

- RTX 20-series GPUs have no bfloat16 support; models run as 4-bit NF4 weights with
  float16 compute (configs/models.toml). Some models can produce NaN in float16; such
  probes are recorded as errors and show up in the smoke test.
- Model weights are cached in `%USERPROFILE%\.cache\huggingface` (about 30 GB for the four
  models). A warning about symlinks on Windows is harmless.
