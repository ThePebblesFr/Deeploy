#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def prepare_series(df, value_col, scale):
    tmp = df.copy()
    tmp["N_c"] = pd.to_numeric(tmp["N_c"], errors="coerce")
    tmp["s_L1 (kB)"] = pd.to_numeric(tmp["s_L1 (kB)"], errors="coerce")
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")

    tmp = tmp.dropna(subset=["N_c", "s_L1 (kB)", value_col])

    # Keep only valid L1 region
    tmp = tmp[tmp["s_L1 (kB)"] > 4]

    series = (
        tmp.groupby("N_c", as_index=False)[value_col]
        .mean()
        .sort_values("N_c")
    )

    series[value_col] *= scale
    return series


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file")
    parser.add_argument("--title", default=None)
    parser.add_argument("--only-correct", action="store_true")
    args = parser.parse_args()

    csv_path = Path(args.csv_file)
    df = pd.read_csv(csv_path)

    # Optional: remove failing runs
    if args.only_correct and "result correctness" in df.columns:
        df = df[df["result correctness"] == 1].copy()

    # Model name
    if "Model" in df.columns and not df["Model"].empty:
        model_name = str(df["Model"].iloc[0])
    else:
        model_name = csv_path.stem

    output_dir = Path(f"gvsoc_log_{model_name}")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{model_name}_vs_nc.png"

    # Prepare data
    delta = prepare_series(df, "delta (cycles)", 1e-6)          # Mcycles
    energy = prepare_series(df, "Energy Consumption (J)", 1e6)  # µJ

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # ---- Latency ----
    # axes[0].plot(delta["N_c"], delta["delta (cycles)"], color="#243b61", marker="o")
    axes[0].bar(delta["N_c"], delta["delta (cycles)"],
            color="#243b61", alpha=0.8)
    axes[0].set_title("Latency vs $N_c$")
    axes[0].set_xlabel("$N_c$")
    axes[0].set_ylabel("Latency $\delta_1$ (Mcycles)")
    axes[0].set_xticks(delta["N_c"])
    axes[0].grid(True, alpha=0.3)

    # ---- Energy ----
    # axes[1].plot(energy["N_c"], energy["Energy Consumption (J)"], color="#243b61", marker="o")
    axes[1].bar(energy["N_c"], energy["Energy Consumption (J)"],
            color="#243b61", alpha=0.8)
    axes[1].set_title("Energy vs $N_c$")
    axes[1].set_xlabel("$N_c$")
    axes[1].set_ylabel("Energy $\mathcal{E}_1$ (µJ)")
    axes[1].set_xticks(energy["N_c"])
    axes[1].grid(True, alpha=0.3)

    # fig.suptitle(args.title if args.title else model_name)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close(fig)

    print(f"Saved {output_file}")


if __name__ == "__main__":
    main()