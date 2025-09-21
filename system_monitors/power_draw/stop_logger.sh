#!/bin/sh
# Stop GPU power logger
# give me exit code

cd "$(dirname "$0")"

run_id="$1"
pids=""

if [ -n "$run_id" ]; then
    # find processes that include the run_id
    pids=$(pgrep -f "gpu_power_logger.py.*${run_id}" || true)
else
    pids=$(pgrep -f gpu_power_logger.py || true)
fi

if [ -n "$pids" ]; then
    echo "[stop_logger] killing pids: $pids"
    kill $pids 2>/dev/null || true
    # wait briefly
    sleep 0.2
    # ensure they're gone
    still=$(pgrep -f "gpu_power_logger.py" || true)
    if [ -n "$still" ]; then
        echo "[stop_logger] forcing kill: $still"
        kill -9 $still 2>/dev/null || true
    fi

    echo "Stopped GPU power logger successfully"
    # run plotter; prefer to pass run_id if provided
    if [ -n "$run_id" ]; then
        python3 "$(dirname "$0")/gpu_power_plot.py" "$(dirname "$0")/logs/gpu_power_*.csv" --run-id "$run_id"
    else
        python3 "$(dirname "$0")/gpu_power_plot.py" "$(dirname "$0")/logs/gpu_power_*.csv"
    fi
    exit 0
else
    echo "No gpu_power_logger.py processes found to stop"
    # still attempt to run plotter (maybe there are logs)
    if [ -n "$run_id" ]; then
        python3 "$(dirname "$0")/gpu_power_plot.py" "$(dirname "$0")/logs/gpu_power_*.csv" --run-id "$run_id"
    else
        python3 "$(dirname "$0")/gpu_power_plot.py" "$(dirname "$0")/logs/gpu_power_*.csv"
    fi
    exit 1
fi