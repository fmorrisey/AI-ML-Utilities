#!/bin/bash
# Analyze GPU power logs
cd "$(dirname "$0")"
python gpu_power_plot.py "./logs/gpu_power_*.csv"
if [ $? -ne 0 ]; then
    echo "Error: Failed to execute gpu_power_plot.py"
    exit 1
fi
# --- IGNORE ---