#!/usr/bin/env python3
import argparse, glob, pandas as pd, matplotlib.pyplot as plt, os

def gpu_power_plot(path, run_id=None):
    files = sorted(glob.glob(path))
    if not files:
        raise SystemExit(f"No CSV files found at {path}")
    df = pd.concat([pd.read_csv(f) for f in files])
    # If run_id provided, filter rows for that run only
    if run_id is not None:
        if "run_id" not in df.columns:
            raise SystemExit(f"No run_id column in CSVs; cannot filter by run_id={run_id}")
        df = df[df["run_id"].astype(str) == str(run_id)]
        if df.empty:
            raise SystemExit(f"No rows found for run_id={run_id}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["timestamp","gpu_index"])
    df["gpu_index"] = df["gpu_index"].astype(str)

    # pivot into wide format: time × gpu for power
    pivot = df.pivot_table(index="timestamp", columns="gpu_index", values="power_w")
    pivot = pivot.interpolate().ffill()
    pivot["total_power"] = pivot.sum(axis=1)

    # Determine energy series: prefer energy_Wh present in CSV
    if "energy_Wh" in df.columns:
        pivot_energy = df.pivot_table(index="timestamp", columns="gpu_index", values="energy_Wh")
        pivot_energy = pivot_energy.interpolate().ffill()
        pivot_energy["total_energy_Wh"] = pivot_energy.sum(axis=1)
        energy_Wh = pivot_energy
    else:
        # trapezoidal integration to get Wh (fallback)
        dt = pivot.index.to_series().diff().dt.total_seconds().fillna(0)
        energy_Wh = (pivot.mul(dt, axis=0) / 3600).cumsum()
        # ensure total_power column not in energy_Wh (we only want per-gpu); add total if missing
        if "total_power" in energy_Wh.columns:
            energy_Wh = energy_Wh.drop(columns=["total_power"])
        energy_Wh["total_energy_Wh"] = energy_Wh.sum(axis=1)

    # Plot
    fig, axs = plt.subplots(2,1,figsize=(10,6), sharex=True)
    pivot.drop(columns="total_power").plot(ax=axs[0])
    pivot["total_power"].plot(ax=axs[0], ls="--", lw=1.5, label="Total")
    axs[0].set_ylabel("Power (W)"); axs[0].legend(); axs[0].grid(True)

    energy_Wh.plot(ax=axs[1])
    axs[1].set_ylabel("Energy (Wh)"); axs[1].set_xlabel("Time"); axs[1].grid(True)

    plt.tight_layout()
    run_tag = f"_{run_id}" if run_id else ""
    out_png = f"gpu_power_plot{run_tag}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
    out_dir = "./power_plots"
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, out_png), dpi=150)
    print(f"Saved plot: {out_png}")

    print("\n=== Energy summary ===")
    final_energy = energy_Wh.iloc[-1]
    for col, val in final_energy.items():
        print(f"{col}: {val:.2f} Wh  ({val/1000:.3f} kWh)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="CSV glob, e.g. './logs/gpu_power_*.csv'")
    ap.add_argument("--run-id", default=None, help="If set, only use rows tagged with this run_id")
    args = ap.parse_args()
    gpu_power_plot(args.path, run_id=args.run_id)
    print("=== Done ===")