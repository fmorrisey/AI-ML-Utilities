#!/usr/bin/env python3
import argparse, csv, datetime as dt, os, signal, subprocess, sys, time
from collections import defaultdict, deque
import uuid

def parse_args():
    p = argparse.ArgumentParser(
        description="Log per-GPU power draw to CSV & compute averages."
    )
    p.add_argument("--interval", type=float, default=1.0,
                   help="Sampling interval in seconds (default: 1.0).")
    p.add_argument("--outdir", default="./logs",
                   help="Output directory (default: ./logs).")
    p.add_argument("--window", type=int, default=60,
                   help="Rolling average window in seconds (default: 60).")
    p.add_argument("--basename", default="gpu_power",
                   help="Log file base name (default: gpu_power).")
    p.add_argument("--nvidia_smi", default="nvidia-smi",
                   help="Path to nvidia-smi if not on PATH.")
    p.add_argument("--run-id", default=None,
                   help="Run identifier to tag this logging run (default: auto-generated).")
    return p.parse_args()

class Rolling:
    def __init__(self, seconds):
        self.seconds = seconds
        self.buf = deque()  # list of (t, value)
        self.sum = 0.0
    def add(self, t, v):
        self.buf.append((t, v))
        self.sum += v
        cutoff = t - self.seconds
        while self.buf and self.buf[0][0] < cutoff:
            _, old = self.buf.popleft()
            self.sum -= old
    def avg(self):
        if not self.buf:
            return 0.0
        return self.sum / len(self.buf)

def now_iso():
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Run id: use provided or generate short uuid
    run_id = args.run_id or dt.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    # include run_id in filename for easier discovery
    out_path = os.path.join(args.outdir, f"{args.basename}_{stamp}_{run_id}.csv")

    # --- New: determine CUDA-visible devices (if any) ---
    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    allowed_idxs = set()
    allowed_uuids = set()
    if cuda_env:
        tokens = [t.strip() for t in cuda_env.split(",") if t.strip() != ""]
        for t in tokens:
            # numeric index
            if t.isdigit():
                try:
                    allowed_idxs.add(int(t))
                except Exception:
                    pass
            else:
                # accept UUID-like tokens (e.g. GPU-....) or any non-numeric token
                allowed_uuids.add(t)
        allowed_any = True
    else:
        allowed_any = False

    # Per-GPU accumulators
    running_sum = defaultdict(float)
    running_cnt = defaultdict(int)
    rolling = defaultdict(lambda: Rolling(args.window))
    vmax = defaultdict(lambda: 0.0)
    # For energy (Wh), we integrate power over time between samples
    energy_Wh = defaultdict(float)  # cumulative per GPU

    # For block-detection so that all GPUs in the same nvidia-smi tick use same dt
    last_block_ts = None
    current_block_dt_s = 0.0
    # Handle clean shutdown
    alive = {"run": True}
    def _stop(signum, frame):
        alive["run"] = False
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, _stop)

    # Start nvidia-smi stream (include uuid so we can filter by CUDA_VISIBLE_DEVICES)
    cmd = [
        args.nvidia_smi,
        "--query-gpu=index,name,uuid,power.draw",
        "--format=csv,noheader,nounits",
        "-l", str(max(1, int(args.interval)))  # nvidia-smi uses integer seconds
    ]
    # If user gave a sub-second interval, we'll still read each line but also sleep.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "gpu_index", "gpu_name", "power_w",
            "running_avg_w", f"rolling_avg_{args.window}s_w", "vmax_w",
            "energy_Wh", "total_energy_Wh", "run_id"
        ])

        if allowed_any:
            print(f"[gpu_power_logger] CUDA_VISIBLE_DEVICES set, filtering to indices={sorted(allowed_idxs)} uuids={sorted(allowed_uuids)}")
        else:
            print(f"[gpu_power_logger] writing -> {out_path}")

        next_sleep = 0.0
        while alive["run"]:
            line = proc.stdout.readline()
            if not line:
                # If the process ended, break; otherwise tiny pause
                if proc.poll() is not None:
                    break
                time.sleep(0.05)
                continue

            # Timestamp for this line
            ts = time.time()
            ts_iso = now_iso()

            try:
                parts = [p.strip() for p in line.strip().split(",")]
                # now expecting: index, name, uuid, power
                if len(parts) < 4:
                    continue
                gpu_idx = int(parts[0])
                gpu_name = parts[1]
                gpu_uuid = parts[2]
                power_w = float(parts[3])
            except Exception:
                continue

            # If CUDA_VISIBLE_DEVICES was set, skip GPUs not in that set.
            if allowed_any:
                # match either numeric index or uuid token
                if (gpu_idx not in allowed_idxs) and (gpu_uuid not in allowed_uuids):
                    # skip logging this GPU as it's not in CUDA_VISIBLE_DEVICES
                    continue

            # Detect start of a new nvidia-smi block (tick)
            threshold = max(0.05, args.interval * 0.5)
            if (last_block_ts is None) or ((ts - last_block_ts) > threshold):
                # new block: compute dt for this block; no integration for first ever block
                if last_block_ts is None:
                    current_block_dt_s = 0.0
                else:
                    current_block_dt_s = ts - last_block_ts
                last_block_ts = ts
            # else: same block, reuse current_block_dt_s

            # Update stats
            running_sum[gpu_idx] += power_w
            running_cnt[gpu_idx] += 1
            vmax[gpu_idx] = max(vmax[gpu_idx], power_w)
            rolling[gpu_idx].add(ts, power_w)

            # Energy integration using the block dt (applied uniformly to all GPUs in the block)
            if current_block_dt_s > 0:
                energy_Wh[gpu_idx] += (power_w * current_block_dt_s) / 3600.0

            # Write row (include per-GPU cumulative energy and total across GPUs)
            avg = running_sum[gpu_idx] / max(1, running_cnt[gpu_idx])
            ravg = rolling[gpu_idx].avg()
            total_e = sum(energy_Wh.values())
            writer.writerow([
                ts_iso, gpu_idx, gpu_name, f"{power_w:.3f}",
                f"{avg:.3f}", f"{ravg:.3f}", f"{vmax[gpu_idx]:.3f}",
                f"{energy_Wh[gpu_idx]:.6f}", f"{total_e:.6f}", run_id
            ])
            f.flush()

            # Optional sub-second sleep (nvidia-smi -l is integer seconds)
            if args.interval < 1.0:
                next_sleep = max(0.0, args.interval)
                time.sleep(next_sleep)

    # Cleanup & final summary
    try:
        proc.terminate()
    except Exception:
        pass

    # Print summary
    print("\n=== GPU Power Summary ===")
    for gpu_idx in sorted(running_cnt.keys()):
        cnt = running_cnt[gpu_idx]
        avg = running_sum[gpu_idx] / cnt if cnt else 0.0
        print(f"GPU {gpu_idx}: samples={cnt}, avg={avg:.2f} W, max={vmax[gpu_idx]:.2f} W, energy≈{energy_Wh[gpu_idx]:.3f} Wh")
    total_run_energy = sum(energy_Wh.values())
    print(f"Total energy for run {run_id}: ≈ {total_run_energy:.3f} Wh")
    print(f"Log saved: {out_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
