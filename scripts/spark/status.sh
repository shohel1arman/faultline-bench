#!/bin/bash
# One look at the Spark run: is the queue alive, what is it doing, how far has it got.
cd "$(dirname "$0")/../.." || exit 1
echo "== processes (STAT D = waiting on the disk, Z = dead)"
ps -o pid,stat,etime,pcpu,rss,args -C python 2>/dev/null | grep -E "PID|run_queue|run_eval" | cut -c1-150
echo; echo "== GPU"
nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw --format=csv 2>/dev/null
nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>/dev/null
echo; echo "== last lines of the log"
tail -c 1500 outputs/spark/queue.log | tr '\r' '\n' | tail -n 8
echo; echo "== log last written: $(date -r outputs/spark/queue.log '+%Y-%m-%d %H:%M')   now: $(date '+%Y-%m-%d %H:%M')"
echo; echo "== records saved"
uv run python scripts/progress.py
