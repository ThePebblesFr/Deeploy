#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Phase 6: computes and plots gamma_eff(alpha_LE) = ops-per-joule under a workload
mix that spends alpha_LE of its time in the LOW_ENERGY (low-criticality) regime and
the remaining (1 - alpha_LE) split evenly across the other 3 modes (CRITICAL_TASK1,
CRITICAL_TASK2, BOTH_CRITICAL), across the 5 compared methods.

Deviates from the original plan's "extend MultiDeployScript/": that directory's
feasible_configs.py/plot_pareto.py are built around a completely different CSV
schema (the static core/L1 DSE sweep -- N_c, s_L1, delta_a, phi_a, ...), which none
of this experiment's data is shaped like. Lives alongside this experiment's other
analysis scripts (parse_scenario_log.py, parse_roundrobin_log.py,
parse_priority_sequential_log.py) instead.

Only DynReconf's gamma_eff actually varies with alpha_LE: it's the only method here
whose per-inference theta/energy differ by mode (it genuinely adapts model/cores
per mode). Every other method -- Static-Small, Static-Large, Round-Robin-Large,
Priority-Sequential -- runs the identical workload regardless of which mode the
environment claims to be in, so each has one fixed, mode-invariant theta/energy per
inference and therefore a FLAT gamma_eff(alpha_LE) line. Static-Small/Static-Large
are literally "always run in mode X" -- their own performance number IS that mode's
LOW_ENERGY/BOTH_CRITICAL calibration point, so this script derives them directly
from the same --mode-low-energy/--mode-both-critical CSVs rather than taking
separate (redundant) inputs for them.

Per-mode DynReconf calibration should come from 4 SEPARATE, PURE single-mode runs
(scenarioUtils/create_scenario.py --force-mode <MODE>, one run per mode, same
mechanism Static-Small/Large already use) rather than mining a single mixed-mode
scenario log: a real scenario's natural mode distribution is rarely balanced (e.g.
one validation run visited BOTH_CRITICAL zero times and CRITICAL_TASK2 only 4 times
out of 100 ticks), giving too few/no samples to average some modes reliably.

Usage:
    python3 scenarioUtils/alpha_sweep.py \
        --mode-low-energy /tmp/static_small.csv \
        --mode-critical-task1 /tmp/force_c1.csv \
        --mode-critical-task2 /tmp/force_c2.csv \
        --mode-both-critical /tmp/static_large.csv \
        --round-robin-large /tmp/rr_report_final.csv \
        --priority-sequential /tmp/ps_report.csv \
        --output /tmp/alpha_sweep.png
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODES = ["LOW_ENERGY", "CRITICAL_TASK1", "CRITICAL_TASK2", "BOTH_CRITICAL"]


def load_mode_point(csv_path: Path, expected_mode: str) -> Tuple[float, float]:
    """
    Returns (theta_per_tick, mean_energy_j_per_tick) from a parse_scenario_log.py
    CSV that stays in a single mode for its entire run (as produced by
    create_scenario.py --force-mode). Errors out if the run isn't actually pure,
    since a mixed-mode CSV here would silently average across modes.
    """
    thetas, energies, modes_seen = [], [], set()
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            modes_seen.add(row["mode_name"])
            if row["theta"] and row["energy_j"]:
                thetas.append(float(row["theta"]))
                energies.append(float(row["energy_j"]))

    if modes_seen != {expected_mode}:
        raise ValueError(
            f"{csv_path}: expected a pure {expected_mode} run (--force-mode {expected_mode}), "
            f"but found mode(s) {sorted(modes_seen)}. Use a dedicated single-mode calibration run."
        )
    if not thetas:
        raise ValueError(f"{csv_path}: no rows with both theta and energy_j populated -- "
                          f"was this generated with --test-dir?")
    return thetas[0], sum(energies) / len(energies)


def load_fixed_policy_point(csv_path: Path, theta_col: str, energy_col: str) -> Tuple[float, float]:
    """(mean theta per tick, mean energy per tick) for a non-adaptive policy."""
    thetas, energies = [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get(theta_col) and row.get(energy_col):
                thetas.append(float(row[theta_col]))
                energies.append(float(row[energy_col]))
    if not thetas:
        raise ValueError(f"{csv_path}: no rows with both {theta_col!r} and {energy_col!r} populated.")
    return sum(thetas) / len(thetas), sum(energies) / len(energies)


def dynreconf_gamma_eff(per_mode: Dict[str, Tuple[float, float]], alpha_le: float) -> float:
    other_mass = (1.0 - alpha_le) / 3.0
    weights = {"LOW_ENERGY": alpha_le, "CRITICAL_TASK1": other_mass,
               "CRITICAL_TASK2": other_mass, "BOTH_CRITICAL": other_mass}
    total_theta = sum(weights[m] * per_mode[m][0] for m in MODES)
    total_energy = sum(weights[m] * per_mode[m][1] for m in MODES)
    return total_theta / total_energy


def find_crossings(alphas: List[float], curve: List[float], baseline: float) -> List[float]:
    crossings = []
    for i in range(1, len(alphas)):
        if (curve[i - 1] - baseline) * (curve[i] - baseline) < 0:
            # Linear-interpolate the exact alpha where the curve crosses the flat baseline.
            a0, a1 = alphas[i - 1], alphas[i]
            c0, c1 = curve[i - 1], curve[i]
            crossings.append(a0 + (baseline - c0) * (a1 - a0) / (c1 - c0))
        elif curve[i] == baseline:
            crossings.append(alphas[i])
    return crossings


def main():
    parser = argparse.ArgumentParser(description="Plot the alpha_LE gamma_eff crossover across the 5 compared methods.")
    parser.add_argument("--mode-low-energy", type=str, required=True,
                         help="parse_scenario_log.py CSV, pure LOW_ENERGY (--force-mode LOW_ENERGY). "
                              "Also used directly as the Static-Small baseline.")
    parser.add_argument("--mode-critical-task1", type=str, required=True,
                         help="Pure CRITICAL_TASK1 (--force-mode CRITICAL_TASK1).")
    parser.add_argument("--mode-critical-task2", type=str, required=True,
                         help="Pure CRITICAL_TASK2 (--force-mode CRITICAL_TASK2).")
    parser.add_argument("--mode-both-critical", type=str, required=True,
                         help="Pure BOTH_CRITICAL (--force-mode BOTH_CRITICAL). "
                              "Also used directly as the Static-Large baseline.")
    parser.add_argument("--round-robin-large", type=str, required=True, help="parse_roundrobin_log.py CSV.")
    parser.add_argument("--priority-sequential", type=str, required=True, help="parse_priority_sequential_log.py CSV.")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num-points", type=int, default=101)
    args = parser.parse_args()

    per_mode = {
        "LOW_ENERGY": load_mode_point(Path(args.mode_low_energy), "LOW_ENERGY"),
        "CRITICAL_TASK1": load_mode_point(Path(args.mode_critical_task1), "CRITICAL_TASK1"),
        "CRITICAL_TASK2": load_mode_point(Path(args.mode_critical_task2), "CRITICAL_TASK2"),
        "BOTH_CRITICAL": load_mode_point(Path(args.mode_both_critical), "BOTH_CRITICAL"),
    }
    static_small = per_mode["LOW_ENERGY"]
    static_large = per_mode["BOTH_CRITICAL"]
    round_robin = load_fixed_policy_point(Path(args.round_robin_large), "theta", "total_energy_j")
    priority_seq = load_fixed_policy_point(Path(args.priority_sequential), "theta", "total_energy_j")

    alphas = [i / (args.num_points - 1) for i in range(args.num_points)]
    dynreconf_curve = [dynreconf_gamma_eff(per_mode, a) / 1e9 for a in alphas]  # GOP/J

    baselines = [
        ("Static-Small", static_small, "tab:green"),
        ("Static-Large", static_large, "tab:red"),
        ("Round-Robin-Large", round_robin, "tab:orange"),
        ("Priority-Sequential", priority_seq, "tab:purple"),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, dynreconf_curve, label="DynReconf", linewidth=2, color="tab:blue")
    for label, (theta, energy), color in baselines:
        gamma = (theta / energy) / 1e9
        ax.axhline(gamma, label=label, linestyle="--", color=color)

    ax.set_xlabel(r"$\alpha_{LE}$ (fraction of time in LOW_ENERGY)")
    ax.set_ylabel(r"$\gamma_{eff}$ (GOP/J)")
    ax.set_title("Energy efficiency vs. workload criticality mix")
    ax.set_xlim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"[alpha_sweep] Wrote {args.output}")

    print(f"[alpha_sweep] DynReconf gamma_eff: alpha_LE=0 -> {dynreconf_curve[0]:.3f} GOP/J, "
          f"alpha_LE=1 -> {dynreconf_curve[-1]:.3f} GOP/J")
    for label, (theta, energy), _ in baselines:
        baseline_gop_j = (theta / energy) / 1e9
        crossings = find_crossings(alphas, dynreconf_curve, baseline_gop_j)
        if crossings:
            print(f"[alpha_sweep] DynReconf ({dynreconf_curve[0]:.1f}->{dynreconf_curve[-1]:.1f} GOP/J) "
                  f"crosses {label} ({baseline_gop_j:.1f} GOP/J) at alpha_LE ~= "
                  f"{', '.join(f'{c:.3f}' for c in crossings)}")
        else:
            above = dynreconf_curve[0] > baseline_gop_j
            print(f"[alpha_sweep] DynReconf stays {'above' if above else 'below'} "
                  f"{label} ({baseline_gop_j:.1f} GOP/J) for all alpha_LE in [0,1]")


if __name__ == "__main__":
    main()
