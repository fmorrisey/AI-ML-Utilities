#!/usr/bin/env python3
import argparse, csv, datetime as dt, os, signal, subprocess, sys, time
from collections import defaultdict, deque

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
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.outdir, f"{args.basename}_{stamp}.csv")

    # Per-GPU accumulators
    running_sum = defaultdict(float)
    running_cnt = defaultdict(int)
    rolling = defaultdict(lambda: Rolling(args.window))
    vmax = defaultdict(lambda: 0.0)
    # For energy (Wh), we integrate power over time between samples
    last_ts = None
    energy_Wh = defaultdict(float)  # sum(power * dt)/3600 per GPU

    # Handle clean shutdown
    alive = {"run": True}
    def _stop(signum, frame):
        alive["run"] = False
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, _stop)

    # Start nvidia-smi stream (noheader + nounits makes parsing easy)
    cmd = [
        args.nvidia_smi,
        "--query-gpu=index,name,power.draw",
        "--format=csv,noheader,nounits",
        "-l", str(max(1, int(args.interval)))  # nvidia-smi uses integer seconds
    ]
    # If user gave a sub-second interval, we'll still read each line but also sleep.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "gpu_index", "gpu_name", "power_w",
            "running_avg_w", f"rolling_avg_{args.window}s_w", "vmax_w"
        ])

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

            # A single nvidia-smi tick prints one line per GPU.
            # We'll attach the same timestamp to all GPUs in that tick.
            # nvidia-smi prints in blocks; we handle per-line.
            ts = time.time()
            ts_iso = now_iso()

            try:
                parts = [p.strip() for p in line.strip().split(",")]
                # Expect: index, name, power
                if len(parts) < 3:
                    continue
                gpu_idx = int(parts[0])
                gpu_name = parts[1]
                power_w = float(parts[2])
            except Exception:
                continue

            # Update stats
            running_sum[gpu_idx] += power_w
            running_cnt[gpu_idx] += 1
            vmax[gpu_idx] = max(vmax[gpu_idx], power_w)
            rolling[gpu_idx].add(ts, power_w)

            # Energy integration (use dt from last_ts to now for all GPUs uniformly)
            if last_ts is not None:
                dt_s = ts - last_ts
                # Approximate: apply current power to this dt interval (OK for 1s sampling)
                energy_Wh[gpu_idx] += (power_w * dt_s) / 3600.0
            # Write row
            avg = running_sum[gpu_idx] / max(1, running_cnt[gpu_idx])
            ravg = rolling[gpu_idx].avg()
            writer.writerow([ts_iso, gpu_idx, gpu_name, f"{power_w:.3f}", f"{avg:.3f}", f"{ravg:.3f}", f"{vmax[gpu_idx]:.3f}"])
            f.flush()

            last_ts = ts

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
    print(f"Log saved: {out_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
