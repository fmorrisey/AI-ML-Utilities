#!/bin/sh
# Start GPU power logger in background
cd "$(dirname "$0")"
nohup ./gpu_power_logger.py --outdir ./logs > ./gpu_power_logger.out 2>&1 &
