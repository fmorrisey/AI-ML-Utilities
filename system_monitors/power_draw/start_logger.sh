#!/bin/sh
# Start GPU power logger in background
cd "$(dirname "$0")"

run_id="$1"
mkdir -p ./logs

if [ -n "$run_id" ]; then
    out="./logs/gpu_power_logger_${run_id}.out"
    echo "[start_logger] starting with run_id=$run_id -> $out"
    nohup python3 ./gpu_power_logger.py --run-id "$run_id" --outdir ./logs > "$out" 2>&1 &
else
    out="./logs/gpu_power_logger.out"
    echo "[start_logger] starting without run_id -> $out"
    nohup python3 ./gpu_power_logger.py --outdir ./logs > "$out" 2>&1 &
fi
