#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


TOTAL_CORES = 8
TOTAL_L1_KB = 64


def infer_model_name(csv_path: Path, df: pd.DataFrame) -> str:
    if "Model" in df.columns and not df["Model"].empty:
        return str(df["Model"].iloc[0])
    return csv_path.stem.replace("results_", "")


def load_csv(csv_path: Path, only_correct: bool = True) -> tuple[str, pd.DataFrame]:
    df = pd.read_csv(csv_path)

    numeric_cols = [
        "N_c",
        "s_L1 (kB)",
        "delta (cycles)",
        "avg phi (W)",
        "Energy Consumption (J)",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if only_correct and "result correctness" in df.columns:
        df = df[df["result correctness"] == 1].copy()

    df = df.dropna(subset=["N_c", "s_L1 (kB)", "delta (cycles)", "avg phi (W)", "Energy Consumption (J)"])
    df["N_c"] = df["N_c"].astype(int)
    df["s_L1 (kB)"] = df["s_L1 (kB)"].astype(int)

    model_name = infer_model_name(csv_path, df)
    return model_name, df


def build_joint_configs(
    model_a: str,
    df_a: pd.DataFrame,
    model_b: str,
    df_b: pd.DataFrame,
    total_cores: int,
    total_l1_kb: int,
) -> pd.DataFrame:
    rows = []

    for _, row_a in df_a.iterrows():
        for _, row_b in df_b.iterrows():
            n_a = int(row_a["N_c"])
            s_a = int(row_a["s_L1 (kB)"])
            n_b = int(row_b["N_c"])
            s_b = int(row_b["s_L1 (kB)"])

            if n_a + n_b <= total_cores and s_a + s_b <= total_l1_kb:
                rows.append({
                    "N_c_a": n_a,
                    "s_L1_a (kB)": s_a,
                    "delta_a (cycles)": float(row_a["delta (cycles)"]),
                    "phi_a (W)": float(row_a["avg phi (W)"]),
                    "E_a (J)": float(row_a["Energy Consumption (J)"]),

                    "N_c_b": n_b,
                    "s_L1_b (kB)": s_b,
                    "delta_b (cycles)": float(row_b["delta (cycles)"]),
                    "phi_b (W)": float(row_b["avg phi (W)"]),
                    "E_b (J)": float(row_b["Energy Consumption (J)"]),

                    "N_c_total": n_a + n_b,
                    "s_L1_total (kB)": s_a + s_b,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["phi_total (W)"] = df["phi_a (W)"] + df["phi_b (W)"]
    df["E_total (J)"] = df["E_a (J)"] + df["E_b (J)"]
    df["delta_max (cycles)"] = df[["delta_a (cycles)", "delta_b (cycles)"]].max(axis=1)
    return df


def add_gamma(df: pd.DataFrame, ops_a: int, ops_b: int) -> pd.DataFrame:
    df = df.copy()
    total_ops = ops_a + ops_b
    df["gamma (OP/J)"] = total_ops / df["E_total (J)"]
    df["gamma (GOP/J)"] = df["gamma (OP/J)"] / 1e9
    return df


def filter_by_constraint(df: pd.DataFrame, args) -> tuple[str, pd.DataFrame]:
    if args.constraint_type is None:
        return "P", df

    if args.constraint_type == "latency":
        out = df.copy()
        if args.delta_a is not None:
            out = out[out["delta_a (cycles)"] <= args.delta_a]
        if args.delta_b is not None:
            out = out[out["delta_b (cycles)"] <= args.delta_b]
        return "Delta", out

    if args.constraint_type == "power":
        out = df.copy()
        if args.phi_a is not None:
            out = out[out["phi_a (W)"] <= args.phi_a]
        if args.phi_b is not None:
            out = out[out["phi_b (W)"] <= args.phi_b]
        if args.phi_total is not None:
            out = out[out["phi_total (W)"] <= args.phi_total]
        return "Phi", out

    if args.constraint_type == "gamma":
        if args.ops_a is None or args.ops_b is None:
            raise ValueError("--constraint-type gamma requires --ops-a and --ops-b")
        if args.gamma_min is None:
            raise ValueError("--constraint-type gamma requires --gamma-min")

        out = add_gamma(df, args.ops_a, args.ops_b)
        out = out[out["gamma (OP/J)"] >= args.gamma_min]
        return "Gamma", out

    raise ValueError(f"Unknown constraint type: {args.constraint_type}")


def main():
    parser = argparse.ArgumentParser(description="Plot Pareto-like scatter for a two-model scenario.")
    parser.add_argument("csv_a")
    parser.add_argument("csv_b")
    parser.add_argument("--only-correct", action="store_true")

    parser.add_argument("--total-cores", type=int, default=TOTAL_CORES)
    parser.add_argument("--total-l1", type=int, default=TOTAL_L1_KB)

    parser.add_argument("--constraint-type", choices=["latency", "power", "gamma"], default=None)

    parser.add_argument("--delta-a", type=float, default=None)
    parser.add_argument("--delta-b", type=float, default=None)

    parser.add_argument("--phi-a", type=float, default=None)
    parser.add_argument("--phi-b", type=float, default=None)
    parser.add_argument("--phi-total", type=float, default=None)

    parser.add_argument("--ops-a", type=int, default=None)
    parser.add_argument("--ops-b", type=int, default=None)
    parser.add_argument("--gamma-min", type=float, default=None)

    parser.add_argument("-o", "--output", default="pareto_scatter.png")
    parser.add_argument("--title", default=None)

    args = parser.parse_args()

    model_a, df_a = load_csv(Path(args.csv_a), only_correct=args.only_correct)
    model_b, df_b = load_csv(Path(args.csv_b), only_correct=args.only_correct)

    P = build_joint_configs(
        model_a, df_a,
        model_b, df_b,
        total_cores=args.total_cores,
        total_l1_kb=args.total_l1,
    )

    if P.empty:
        raise ValueError("No feasible configuration in P under global architectural constraints.")

    set_name, feasible = filter_by_constraint(P, args)

    # Best feasible config according to the objective: minimize E_total
    best = None
    if not feasible.empty:
        best = feasible.sort_values("E_total (J)", ascending=True).iloc[0]

    # Convert for display
    x_all = P["delta_max (cycles)"] / 1e3         # kcycles
    y_all = P["E_total (J)"] * 1e6                # µJ

    x_feas = feasible["delta_max (cycles)"] / 1e3 if not feasible.empty else []
    y_feas = feasible["E_total (J)"] * 1e6 if not feasible.empty else []

    plt.figure(figsize=(8, 6))

    plt.scatter(x_all, y_all, color="#64cdf6", marker="x", alpha=0.4, label="All configurations ($\mathcal{P}$)")
    if not feasible.empty:
        plt.scatter(x_feas, y_feas, color="#145a32", marker="x", alpha=0.8, label="Feasible configurations ($\mathcal{P}_{\\" + set_name + "}$)")

    if best is not None:
        plt.scatter(
            [best["delta_max (cycles)"] / 1e3],
            [best["E_total (J)"] * 1e6],
            marker="*",
            s=200,
            color="#b85450",
            label=r"Best configuration",
        )

    plt.xlabel(r"System latency $\delta_{total}$ (kcycles)")
    plt.ylabel(r"Total energy $\mathcal{E}_{total}$ ($\mu$J)")

    # if args.title:
    #     plt.title(args.title)
    # else:
    #     plt.title(f"{model_a} + {model_b}: {set_name}")

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    plt.close()

    print(f"Saved {args.output}")

    print(f"\ncard(P) = {len(P)}")
    print(f"card({set_name}) = {len(feasible)}")

    if best is not None:
        print("\nSelected best feasible configuration p*:")
        print(f"  {model_a}: N_c={best['N_c_a']}, s_L1={best['s_L1_a (kB)']} kB")
        print(f"  {model_b}: N_c={best['N_c_b']}, s_L1={best['s_L1_b (kB)']} kB")
        print(f"  delta_max = {best['delta_max (cycles)']:.0f} cycles")
        print(f"  E_total   = {best['E_total (J)']:.12g} J")
        print(f"  phi_total = {best['phi_total (W)']:.12g} W")
        if "gamma (GOP/J)" in feasible.columns:
            print(f"  gamma     = {best['gamma (GOP/J)']:.6g} GOP/J")
    else:
        print("\nNo feasible configuration satisfies the selected constraint.")


if __name__ == "__main__":
    main()