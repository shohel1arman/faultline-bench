# Running FaultLine on a Windows PC with NVIDIA GPUs (WSL2)

Tested target: Windows with two RTX 2060 (6 GB each). Everything runs inside WSL2
(Ubuntu), which uses the Windows NVIDIA driver; do not install an NVIDIA driver inside Ubuntu.

## 1. One-time setup

In PowerShell (as administrator), if `wsl -l -v` shows no Ubuntu:

```
wsl --install -d Ubuntu-24.04
```

Restart Windows when asked, open "Ubuntu" from the Start menu, create a user, then:

```
nvidia-smi
```

Both GPUs should be listed. Install tmux (keeps long runs alive), then uv, and clone the repo into the Linux home
directory (not under /mnt/c, which is much slower):

```
sudo apt update && sudo apt install -y tmux
```
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
```
source ~/.bashrc && cd ~ && git clone https://github.com/shohel1arman/faultline-bench.git && cd faultline-bench
```
```
uv sync --extra eval --group dev
```
```
uv run pytest -q
```
```
uv run python -c "import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

The last command should print `2 NVIDIA GeForce RTX 2060`.

## 2. Build the dataset

```
uv run python scripts/download_data.py --out data/raw
```
```
uv run python scripts/build_dataset.py --config configs/dataset.toml
```
```
uv run python scripts/fetch_images.py --dataset data/faultline_v0.jsonl --out data/images
```

## 3. Smoke test (always before a long run)

```
uv run python scripts/run_eval.py --model qwen25vl-3b --device cuda:0 --limit 5
```
```
uv run python scripts/score_eval.py outputs/qwen25vl-3b/core_limit5.jsonl --show 2
```

Check: no errors, outputs are readable English, `letter_mass` is not tiny, and the
reported seconds per record.

## 4. Full runs

Run one model per GPU in two terminals. Use `tmux` so runs survive a closed terminal:

```
tmux new -s gpu0
```
```
uv run python scripts/run_eval.py --model qwen25vl-3b --device cuda:0 --subset core
```

Detach with Ctrl-b then d. In a second tmux session use `--device cuda:1` and another model.
Reattach later with `tmux attach -t gpu0`.

A stopped run resumes where it left off when the same command is run again.

Keep Windows awake while runs are going: Settings, System, Power, set sleep to Never.

## Notes

- RTX 20-series GPUs have no bfloat16 support; models run as 4-bit NF4 weights with
  float16 compute (configs/models.toml). Some models can produce NaN in float16; such
  probes are recorded as errors and show up in the smoke test.
- WSL2 uses at most half of the Windows RAM by default. If loading a model is killed,
  raise the limit in `%UserProfile%\.wslconfig` (`[wsl2]` then `memory=24GB`) and run
  `wsl --shutdown` in PowerShell.
- Model weights are cached in `~/.cache/huggingface` (about 30 GB for the four models).
