#!/bin/bash
# Runs the FaultLine model queue on the DGX Spark. Start it with
#   nohup bash scripts/spark/queue.sh > /dev/null 2>&1 &
# so closing the terminal does not stop it. Finished records are skipped, so after a
# restart the same command resumes; run_queue.py refuses to start a second queue.
# Phase 1 brings every model to the same first 1,000 records; phase 2 completes all 2,001.
# Models run smallest first; the 32B models are last so they can be dropped if time runs out.
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
cd "$(dirname "$0")/../.." || exit 1
mkdir -p outputs/spark
echo "=== queue start $(date '+%Y-%m-%d %H:%M:%S')" >> outputs/spark/queue.log
MODELS=internvl3-2b,qwen3vl-2b,qwen25vl-3b,qwen3vl-4b,smolvlm2-2b,qwen25vl-7b,internvl3-8b,qwen3vl-8b,qwen3vl-30b-a3b,qwen3vl-32b,qwen25vl-32b
CMD=(uv run python scripts/run_queue.py --device cuda:0 --models-config configs/models_spark.toml
     --out-root outputs/spark --models "$MODELS" --phases 1000,all)
# keep the machine from suspending while the queue runs, where systemd allows it
if command -v systemd-inhibit > /dev/null; then
    CMD=(systemd-inhibit --what=sleep:idle --why="FaultLine evaluation" "${CMD[@]}")
fi
exec "${CMD[@]}" >> outputs/spark/queue.log 2>&1
