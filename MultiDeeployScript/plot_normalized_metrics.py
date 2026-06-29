#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def prepare_model(df):
    df = df.copy()
    df["N_c"] = pd.to_numeric(df["N_c"], errors="coerce")
    df["s_L1 (kB)"] = pd.to_numeric(df["s_L1 (kB)"], errors="coerce")
    df["delta (cycles)"] = pd.to_numeric(df["delta (cycles)"], errors="coerce")
    df["Energy Consumption (J)"] = pd.to_numeric(df["Energy Consumption (J)"], errors="coerce")

    df = df.dropna(subset=["N_c", "s_L1 (kB)", "delta (cycles)", "Energy Consumption (J)"])
    df = df[df["s_L1 (kB)"] > 4]
    # df = df[df["result correctness"] == 1]

    grouped = (
        df.groupby("N_c", as_index=False)[["delta (cycles)", "Energy Consumption (J)"]]
        .mean()
        .sort_values("N_c")
    )

    delta_1 = grouped.loc[grouped["N_c"] == 1, "delta (cycles)"].iloc[0]
    energy_1 = grouped.loc[grouped["N_c"] == 1, "Energy Consumption (J)"].iloc[0]

    grouped["speedup"] = delta_1 / grouped["delta (cycles)"]
    grouped["parallel_efficiency"] = grouped["speedup"] / grouped["N_c"]
    grouped["energy_ratio"] = grouped["Energy Consumption (J)"] / energy_1

    return grouped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_files", nargs="+")
    parser.add_argument("-o", "--output", default="normalized_metrics.png")
    parser.add_argument("--title", default="Normalized scaling metrics")
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    summary_rows = []
    colors = ["#64cdf6", "#b85450", "#145a32"]

    for i, csv_file in enumerate(args.csv_files):
        path = Path(csv_file)
        df = pd.read_csv(path)

        color = colors[i % len(colors)]

        if "Model" in df.columns and not df["Model"].empty:
            model_name = str(df["Model"].iloc[0])
        else:
            model_name = path.stem

        grouped = prepare_model(df)

        # axes[0].bar(grouped["N_c"], grouped["speedup"], color=color, alpha=0.8)
        axes[0].scatter(grouped["N_c"], grouped["speedup"], color=color, marker="x", label=model_name)
        # axes[1].bar(grouped["N_c"], grouped["energy_ratio"], color=color, alpha=0.8)
        axes[1].scatter(grouped["N_c"], grouped["energy_ratio"], color=color, marker="x", label=model_name)

        row8 = grouped[grouped["N_c"] == 8]
        row4 = grouped[grouped["N_c"] == 4]

        summary_rows.append({
            "Model": model_name,
            "Max speedup": grouped["speedup"].max(),
            "Efficiency @4": row4["parallel_efficiency"].iloc[0] if not row4.empty else None,
            "Efficiency @8": row8["parallel_efficiency"].iloc[0] if not row8.empty else None,
            "Energy ratio @8": row8["energy_ratio"].iloc[0] if not row8.empty else None,
        })

    axes[0].set_title("Speedup vs $N_c$")
    axes[0].set_xlabel("$N_c$")
    axes[0].set_ylabel("Speedup")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Energy ratio vs $N_c$")
    axes[1].set_xlabel("$N_c$")
    axes[1].set_ylabel("Energy ratio $\mathcal{E}(N_c) / \mathcal{E}(1)$")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # fig.suptitle(args.title)
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    plt.close(fig)

    print(f"Saved {args.output}")

    summary_df = pd.DataFrame(summary_rows)
    print("\nSummary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()