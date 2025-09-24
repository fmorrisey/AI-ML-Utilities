#!/bin/bash
### See README.md in this project's repor for usage instructions
### Place this file at project root and pass the path to your python script as the first argument
### e.g. bash run_py.sh pytorch101/04_mnist_cnn.py


currentdir=$(pwd)
pathToUtils="$currentdir/utils/system_monitors/power_draw"
# cd ~/code/ai-ml-utilities/system_monitors/power_draw || exit 1
# generate a run id (timestamp + short random hex) and export for child processes
shorthex=$(head -c4 /dev/urandom | od -An -tx1 | tr -d ' \n')
run_id="$(date +%Y%m%d_%H%M%S)_${shorthex}"
export RUN_ID="$run_id"
echo "[run_py] RUN_ID=$RUN_ID"

logger_pid=""

# Start the logger (prefer start_logger.sh if present)
if [ -f "$pathToUtils/start_logger.sh" ]; then
    echo "[run_py] Starting logger via start_logger.sh with run_id=$RUN_ID"
    bash "$pathToUtils/start_logger.sh" "$RUN_ID" || echo "start_logger.sh returned non-zero"
    # give it a moment and try to detect a logger process that includes the run id
    sleep 0.2
    pids=$(pgrep -f "gpu_power_logger.py.*${RUN_ID}" || true)
    if [ -n "$pids" ]; then
        logger_pid=$(echo "$pids" | awk '{print $1}')
    fi
else
    # Fallback: start gpu_power_logger.py directly in background
    if [ -f "$pathToUtils/gpu_power_logger.py" ]; then
        mkdir -p "$pathToUtils/logs"
        echo "[run_py] Starting gpu_power_logger.py directly (out -> $pathToUtils/logs/logger_${RUN_ID}.out)"
        nohup python3 "$pathToUtils/gpu_power_logger.py" --run-id "$RUN_ID" --outdir "$pathToUtils/logs" > "$pathToUtils/logs/logger_${RUN_ID}.out" 2>&1 &
        logger_pid=$!
        echo "[run_py] logger pid=$logger_pid"
    else
        echo "Logger script not found, continuing..."
    fi
fi

cd "$currentdir" || exit 1
# first CLI arg is the python script to run; defaults to pytorch101/04_mnist_cnn.py
script_path="${1:-pytorch101/04_mnist_cnn.py}"
shift || true

# if not absolute and exists under currentdir, use that
if [[ "$script_path" != /* ]] && [ -f "$currentdir/$script_path" ]; then
    script_path="$currentdir/$script_path"
fi

if [ ! -f "$script_path" ]; then
    echo "Python script not found: $script_path" >&2
    # Attempt to stop logger before exiting
    if [ -f "$pathToUtils/stop_logger.sh" ]; then
        bash "$pathToUtils/stop_logger.sh" "$RUN_ID" || true
    elif [ -n "$logger_pid" ]; then
        kill "$logger_pid" 2>/dev/null || true
    fi
    exit 1
fi

# Run the training script with RUN_ID exported
python3 "$script_path" "$@"

# After training, stop the logger (prefer stop_logger.sh)
if [ -f "$pathToUtils/stop_logger.sh" ]; then
    echo "[run_py] Stopping logger via stop_logger.sh with run_id=$RUN_ID"
    bash "$pathToUtils/stop_logger.sh" "$RUN_ID" || echo "stop_logger.sh returned non-zero"
else
    # If we started a logger ourselves, try to kill it gracefully
    if [ -n "$logger_pid" ]; then
        echo "[run_py] Stopping logger pid $logger_pid"
        kill "$logger_pid" 2>/dev/null || true
        sleep 0.2
        kill -0 "$logger_pid" 2>/dev/null && kill -9 "$logger_pid" 2>/dev/null || true
    else
        # Try to find any gpu_power_logger.py processes for this run_id and kill them
        pids=$(pgrep -f "gpu_power_logger.py.*${RUN_ID}" || true)
        if [ -n "$pids" ]; then
            echo "[run_py] Killing logger pids: $pids"
            kill $pids 2>/dev/null || true
        else
            echo "Logger script not found, continuing..."
        fi
    fi
fi
