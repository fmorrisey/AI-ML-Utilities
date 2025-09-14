#!/bin/sh
# Stop GPU power logger
# give me exit code

if pkill -f gpu_power_logger.py; then
    echo "Stopped GPU power logger successfully"
    python3 "$(dirname "$0")/gpu_power_plot.py" "$(dirname "$0")/logs/gpu_power_*.csv"
    exit 0
else
    echo "Failed to stop GPU power logger"
    exit 1
fi