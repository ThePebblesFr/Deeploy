#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from MultiDeeploy.ModelProfile import ModelProfile


TOTAL_CORES = 8
TOTAL_L1_KB = 64


def load_total_ops(profiling_json_path: Path) -> int:
    """
    theta for one model, read from a Deeploy `*_profiling.json` (as written by
    DeeployTest/testUtils/testRunner.py with --profileTiling --profileToJSON) --
    an alternative to hand-computing and passing --ops-a/--ops-b.
    """
    with open(profiling_json_path) as f:
        profile = ModelProfile.from_dict(json.load(f))
    total_ops = profile.total_ops()
    if total_ops is None:
        raise ValueError(
            f"{profiling_json_path} has no per-layer 'ops' field -- it was generated before "
            f"testRunner.py started capturing it. Regenerate it, or pass --ops-a/--ops-b directly."
        )
    return total_ops


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
        "Frequency (Hz)",
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
                    "model_a": model_a,
                    "N_c_a": n_a,
                    "s_L1_a (kB)": s_a,
                    "delta_a (cycles)": float(row_a["delta (cycles)"]),
                    "phi_a (W)": float(row_a["avg phi (W)"]),
                    "E_a (J)": float(row_a["Energy Consumption (J)"]),

                    "model_b": model_b,
                    "N_c_b": n_b,
                    "s_L1_b (kB)": s_b,
                    "delta_b (cycles)": float(row_b["delta (cycles)"]),
                    "phi_b (W)": float(row_b["avg phi (W)"]),
                    "E_b (J)": float(row_b["Energy Consumption (J)"]),

                    "N_c_total": n_a + n_b,
                    "s_L1_total (kB)": s_a + s_b,
                    "phi_total (W)": float(row_a["avg phi (W)"]) + float(row_b["avg phi (W)"]),
                    "E_total (J)": float(row_a["Energy Consumption (J)"]) + float(row_b["Energy Consumption (J)"]),
                })

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    out["joint_config"] = out.apply(
        lambda r: (
            f"({r['model_a']}: N_c={r['N_c_a']}, s_L1={r['s_L1_a (kB)']} kB; "
            f"{r['model_b']}: N_c={r['N_c_b']}, s_L1={r['s_L1_b (kB)']} kB)"
        ),
        axis=1,
    )

    return out


def add_gamma(df: pd.DataFrame, ops_a: int, ops_b: int) -> pd.DataFrame:
    df = df.copy()
    total_ops = ops_a + ops_b
    df["gamma (OP/J)"] = total_ops / df["E_total (J)"]
    df["gamma (GOP/J)"] = df["gamma (OP/J)"] / 1e9
    df["gamma (TOP/J)"] = df["gamma (OP/J)"] / 1e12
    return df


def filter_by_constraint(
    df: pd.DataFrame,
    constraint_type: str | None,
    args,
) -> tuple[str, pd.DataFrame]:
    if constraint_type is None:
        return "P", df

    if constraint_type == "latency":
        out = df.copy()
        if args.delta_a is not None:
            out = out[out["delta_a (cycles)"] <= args.delta_a]
        if args.delta_b is not None:
            out = out[out["delta_b (cycles)"] <= args.delta_b]
        return "P_Delta", out

    if constraint_type == "power":
        out = df.copy()
        if args.phi_a is not None:
            out = out[out["phi_a (W)"] <= args.phi_a]
        if args.phi_b is not None:
            out = out[out["phi_b (W)"] <= args.phi_b]
        if args.phi_total is not None:
            out = out[out["phi_total (W)"] <= args.phi_total]
        return "P_Phi", out

    if constraint_type == "gamma":
        if args.ops_a is None or args.ops_b is None:
            raise ValueError("--constraint-type gamma requires --ops-a and --ops-b")
        if args.gamma_min is None:
            raise ValueError("--constraint-type gamma requires --gamma-min")

        out = add_gamma(df, args.ops_a, args.ops_b)
        out = out[out["gamma (OP/J)"] >= args.gamma_min]
        return "P_Gamma", out

    raise ValueError(f"Unknown constraint type: {constraint_type}")


def sort_configs(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "E_total (J)",
        "phi_total (W)",
        "N_c_total",
        "s_L1_total (kB)",
    ]
    present = [c for c in cols if c in df.columns]
    return df.sort_values(present, ascending=[True] * len(present))

def print_best_config(name: str, df: pd.DataFrame):
    if df.empty:
        print(f"\n{name}: no feasible configuration, so no optimal config.")
        return

    best = df.sort_values("E_total (J)", ascending=True).iloc[0]

    print(f"\n  card({name})    = {len(df)}")
    print(f"Best configuration in {name} according to the objective function:")
    print("  Objective: minimize total energy E_total (equivalently maximize gamma when total ops are fixed)")
    print(f"  joint_config   = {best['joint_config']}")
    print(f"  E_total (J)    = {best['E_total (J)']:.12g}")
    print(f"  phi_total (W)  = {best['phi_total (W)']:.12g}")
    print(f"  delta_a        = {best['delta_a (cycles)']:.0f} cycles")
    print(f"  delta_b        = {best['delta_b (cycles)']:.0f} cycles")

    if 'gamma (GOP/J)' in df.columns:
        print(f"  gamma (GOP/J)  = {best['gamma (GOP/J)']:.6g}")
    if 'gamma (TOP/J)' in df.columns:
        print(f"  gamma (TOP/J)  = {best['gamma (TOP/J)']:.6g}")


def print_summary(name: str, df: pd.DataFrame, max_rows: int):
    print(f"\n{name}: {len(df)} feasible joint configuration(s)")

    if df.empty:
        return

    cols = [
        "joint_config",
        "E_total (J)",
        "phi_total (W)",
        "delta_a (cycles)",
        "delta_b (cycles)",
    ]
    for col in ["gamma (GOP/J)", "gamma (TOP/J)"]:
        if col in df.columns:
            cols.append(col)

    show = df[cols].head(max_rows).copy()
    print(show.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Compute P, P_Delta, P_Phi or P_Gamma for two models from their CSVs."
    )
    parser.add_argument("csv_a", help="CSV for model A")
    parser.add_argument("csv_b", help="CSV for model B")

    parser.add_argument("--total-cores", type=int, default=TOTAL_CORES)
    parser.add_argument("--total-l1", type=int, default=TOTAL_L1_KB)
    parser.add_argument("--only-correct", action="store_true")

    parser.add_argument(
        "--constraint-type",
        choices=["latency", "power", "gamma"],
        default=None,
        help="Constraint family to apply. If omitted, computes only P.",
    )

    # Latency constraints
    parser.add_argument("--delta-a", type=float, default=None, help="Max latency for model A in cycles")
    parser.add_argument("--delta-b", type=float, default=None, help="Max latency for model B in cycles")

    # Power constraints
    parser.add_argument("--phi-a", type=float, default=None, help="Max avg power for model A in W")
    parser.add_argument("--phi-b", type=float, default=None, help="Max avg power for model B in W")
    parser.add_argument("--phi-total", type=float, default=None, help="Max total avg power in W")

    # Gamma constraint -- theta (ops) per model, either given directly or read from
    # a Deeploy profiling JSON (see load_total_ops()); the JSON form is preferred
    # since it comes straight from Deeploy's own NetworkDeployer.numberOfOps()
    # rather than a hand-computed number that can drift from the actual model.
    parser.add_argument("--ops-a", type=int, default=None, help="Operation count for model A")
    parser.add_argument("--ops-b", type=int, default=None, help="Operation count for model B")
    parser.add_argument("--ops-a-json", type=str, default=None,
                         help="Path to model A's *_profiling.json; derives --ops-a from it")
    parser.add_argument("--ops-b-json", type=str, default=None,
                         help="Path to model B's *_profiling.json; derives --ops-b from it")
    parser.add_argument("--gamma-min", type=float, default=None, help="Min global energy efficiency in OP/J")

    parser.add_argument("--max-rows", type=int, default=20, help="Max rows to print")
    parser.add_argument("--save-csv", type=str, default=None, help="Optional output CSV path")

    args = parser.parse_args()

    if args.ops_a_json:
        if args.ops_a is not None:
            raise ValueError("Pass only one of --ops-a and --ops-a-json")
        args.ops_a = load_total_ops(Path(args.ops_a_json))
    if args.ops_b_json:
        if args.ops_b is not None:
            raise ValueError("Pass only one of --ops-b and --ops-b-json")
        args.ops_b = load_total_ops(Path(args.ops_b_json))

    model_a, df_a = load_csv(Path(args.csv_a), only_correct=args.only_correct)
    model_b, df_b = load_csv(Path(args.csv_b), only_correct=args.only_correct)

    P = build_joint_configs(
        model_a, df_a,
        model_b, df_b,
        total_cores=args.total_cores,
        total_l1_kb=args.total_l1,
    )

    if P.empty:
        print("No feasible joint configuration found under global architectural constraints.")
        return

    P = sort_configs(P)
    print_summary("P", P, args.max_rows)
    print_best_config("P", P)

    if args.constraint_type is not None:
        set_name, constrained = filter_by_constraint(P, args.constraint_type, args)
        if not constrained.empty:
            constrained = sort_configs(constrained)
        print_summary(set_name, constrained, args.max_rows)
        print_best_config(set_name, constrained)

        if args.save_csv:
            constrained.to_csv(args.save_csv, index=False)
            print(f"\nSaved {set_name} to {args.save_csv}")
    else:
        if args.save_csv:
            P.to_csv(args.save_csv, index=False)
            print(f"\nSaved P to {args.save_csv}")


if __name__ == "__main__":
    main()