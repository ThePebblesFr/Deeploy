#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def prepare_data(df, value_col, scale):
    tmp = df.copy()
    tmp["N_c"] = pd.to_numeric(tmp["N_c"], errors="coerce")
    tmp["s_L1 (kB)"] = pd.to_numeric(tmp["s_L1 (kB)"], errors="coerce")
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp.dropna(subset=["N_c", "s_L1 (kB)", value_col])

    x = tmp["N_c"]
    y = tmp["s_L1 (kB)"]
    z = tmp[value_col] * scale
    return x, y, z


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", help="Input CSV file")
    parser.add_argument(
        "-o",
        "--output",
        default="energy_scatter.png",
        help="Output image filename",
    )
    parser.add_argument(
        "--title",
        default="Energy consumption scatter plot",
        help="Figure title",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv_file)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Energy in microjoules
    x, y, z = prepare_data(df, "Energy Consumption (J)", scale=1e6)

    sc = ax.scatter(x, y, z)
    ax.set_title(args.title)
    ax.set_xlabel("N_c")
    ax.set_ylabel("s_L1 (kB)")
    ax.set_zlabel("Energy Consumption (µJ)")

    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    plt.close(fig)

    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()