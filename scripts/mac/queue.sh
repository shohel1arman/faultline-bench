#!/bin/zsh
# Runs the FaultLine model queue; used by hand or by the auto-resume agent after a restart.
# Finished records are skipped, and run_queue.py refuses to start if a queue is running.
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
cd "$(dirname "$0")/../.." || exit 1
mkdir -p outputs
echo "=== queue start $(date '+%Y-%m-%d %H:%M:%S')" >> outputs/queue.log
exec /usr/bin/caffeinate -i uv run python scripts/run_queue.py \
    --device mps --models-config configs/models_mac.toml \
    --models smolvlm2-2b,qwen25vl-3b,qwen3vl-4b \
    --phases 1000,all \
    --extra-models internvl3-2b >> outputs/queue.log 2>&1
