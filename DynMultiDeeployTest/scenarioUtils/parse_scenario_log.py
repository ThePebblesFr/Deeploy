#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Parses a saved GVSoC scenario run log (dynmultideeploytest.c's DynReconf FSM, built
with -D GVSOC_EXTRA_FLAGS=--power) into a clean per-tick CSV: mode, hardware
repartition, delta/phi/E, theta, and the context state that drove/resulted from each
tick.

Usage:
    python3 test_dynamic_2_models.py -t Tests/<pair> --models <m1> <m2> \
        --scenario Scenarios/demo --gen-step 2>&1 | tee /tmp/run.log
    python3 scenarioUtils/parse_scenario_log.py --log /tmp/run.log --output /tmp/report.csv \
        --test-dir Tests/<pair> [--configs Configs/<file>.json]

Scope note: delta/phi/E are for the WHOLE co-resident RunNetworks_* call (both tasks
together), not split per task. theta is per-tick total ops for whichever mode ran
that tick (task1's total_ops + task2's total_ops for that mode's specific
variant/cores pair), read from the profiling JSONs testRunner_2_models.py already
writes unconditionally during --gen-step -- see ThetaLookup.py. Pass --test-dir to
enable it (needs the fused test directory to locate those profiling JSONs);
--configs defaults to the same file test_dynamic_2_models.py defaults to. Without
--test-dir, theta/cumulative_theta are left blank, same as before this was added.
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

# This script lives in scenarioUtils/ and is invoked as
# `python3 scenarioUtils/parse_scenario_log.py`, which puts scenarioUtils/ itself
# (not DynMultiDeeployTest/) on sys.path[0] -- add the parent explicitly so the
# testUtils/scenarioUtils sibling-package imports below resolve regardless of how
# this script is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from testUtils.ConfigLoader import load_configs, DEFAULT_CONFIGS_PATH
from scenarioUtils.ThetaLookup import build_theta_by_mode

FREQUENCY_HZ = 50_000_000.0

# Mode -> (name, cores_task1, cores_task2). Intrinsic to the paper's 4-mode design
# (LE (4,32)/(4,32), C1 (6,48)/(2,16), C2 (2,16)/(6,48), CC (4,32)/(4,32)), so this
# stays correct regardless of which models/variants are plugged into each mode.
MODE_INFO = {
    1: ("LOW_ENERGY", 4, 4),
    2: ("CRITICAL_TASK1", 6, 2),
    3: ("CRITICAL_TASK2", 2, 6),
    4: ("BOTH_CRITICAL", 4, 4),
}

TICK_DECISION_RE = re.compile(
    r"\[HOST CORE\] Tick (\d+): operation mode (\d+) \("
    r"task1_critical=(\d+), task2_critical=(\d+), operator_cmd=(\d+), power_over_budget=(\d+)\)"
)
INIT_CYCLES_RE = re.compile(r"Runtime InitNetwork: (\d+) cycles")
RUN_CYCLES_RE = re.compile(r"Runtime RunNetwork: (\d+) cycles")
TICK_CONTEXT_RE = re.compile(
    r"\[HOST CORE\] Tick (\d+) context: task1_margin=(-?\d+) task2_margin=(-?\d+) total_energy_est_nj=(\d+)"
)
POWER_RE = re.compile(r"@power\.measure_(\d+)@([\d.]+)@")


def parse_log(log_path: Path, theta_by_mode: Optional[Dict[str, int]] = None) -> List[Dict]:
    raw_text = log_path.read_text()

    # Power lines can appear out of stream order relative to the rest of the log
    # (UART output buffering vs GVSoC's own unbuffered power print), but are always
    # numbered in program order -- collect them by index rather than by position.
    phi_by_index: Dict[int, float] = {}
    for m in POWER_RE.finditer(raw_text):
        phi_by_index[int(m.group(1))] = float(m.group(2))

    # At scale (confirmed at DELTA_T=100, not visible in earlier 12-tick validation
    # runs), a burst of buffered @power.measure_N@ lines can flush mid-printf, right
    # in between the "Tick %d:" and " operation mode %d (...)" halves of a single
    # `printf("[HOST CORE] Tick %d: operation mode %d (...)")` call -- splitting one
    # decision line's text across several lines, e.g.:
    #   [HOST CORE] Tick 35:@power.measure_35@0.000289@
    #   @power.measure_36@0.000283@
    #   ...
    #    operation mode 2 (task1_critical=1, ...)
    # No data is actually lost: the two halves are still the exact original text,
    # just separated by whole power-measurement lines (each ending in its own
    # newline, coincidentally at exactly the point where the interleaving landed).
    # Stripping each power token together with its own trailing newline removes
    # those lines cleanly and re-joins the two halves back into one line, restoring
    # the text a plain TICK_DECISION_RE match expects -- without needing to change
    # phi_by_index above, which already reads from the un-stripped raw text.
    cleaned_text = re.sub(r"@power\.measure_\d+@[\d.]+@\n?", "", raw_text)

    ticks: List[Dict] = []
    current: Optional[Dict] = None
    run_call_index = 0  # counts RunNetworks_* calls in program order == power measure index

    for line in cleaned_text.splitlines():
        m = TICK_DECISION_RE.search(line)
        if m:
            if current is not None:
                ticks.append(current)
            tick, mode, t1_crit, t2_crit, op_cmd, pwr_budget = (int(g) for g in m.groups())
            mode_name, cores_t1, cores_t2 = MODE_INFO.get(mode, (f"UNKNOWN_{mode}", None, None))
            current = {
                "tick": tick,
                "mode": mode,
                "mode_name": mode_name,
                "cores_task1": cores_t1,
                "cores_task2": cores_t2,
                "task1_critical_in": t1_crit,
                "task2_critical_in": t2_crit,
                "operator_cmd": op_cmd,
                "power_over_budget": pwr_budget,
                "init_cycles": 0,
                "mode_changed": 0,
                "delta_cycles": None,
                "phi_watts": None,
                "task1_margin": None,
                "task2_margin": None,
                "total_energy_est_nj": None,
                "theta": theta_by_mode.get(mode_name) if theta_by_mode else None,
            }
            continue

        if current is None:
            continue

        m = INIT_CYCLES_RE.search(line)
        if m:
            current["init_cycles"] = int(m.group(1))
            current["mode_changed"] = 1
            continue

        m = RUN_CYCLES_RE.search(line)
        if m:
            current["delta_cycles"] = int(m.group(1))
            current["phi_watts"] = phi_by_index.get(run_call_index)
            run_call_index += 1
            continue

        m = TICK_CONTEXT_RE.search(line)
        if m:
            _, task1_margin, task2_margin, total_energy_est_nj = (int(g) for g in m.groups())
            current["task1_margin"] = task1_margin
            current["task2_margin"] = task2_margin
            current["total_energy_est_nj"] = total_energy_est_nj

    if current is not None:
        ticks.append(current)

    cumulative_theta = 0
    for row in ticks:
        delta_cycles = row["delta_cycles"]
        phi = row["phi_watts"]
        row["delta_seconds"] = (delta_cycles / FREQUENCY_HZ) if delta_cycles is not None else None
        row["energy_j"] = (row["delta_seconds"] * phi) if (row["delta_seconds"] is not None and phi is not None) else None
        if row["theta"] is not None:
            cumulative_theta += row["theta"]
            row["cumulative_theta"] = cumulative_theta
        else:
            row["cumulative_theta"] = None

    return ticks


def write_csv(ticks: List[Dict], output_path: Path) -> None:
    fieldnames = [
        "tick", "mode", "mode_name", "cores_task1", "cores_task2", "mode_changed",
        "operator_cmd", "power_over_budget",
        "task1_critical_in", "task2_critical_in",
        "delta_cycles", "delta_seconds", "init_cycles", "phi_watts", "energy_j",
        "theta", "cumulative_theta",
        "task1_margin", "task2_margin", "total_energy_est_nj",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in ticks:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Parse a scenario GVSoC log into a per-tick CSV report.")
    parser.add_argument("--log", type=str, required=True, help="Path to the saved run log (stdout of test_dynamic_2_models.py).")
    parser.add_argument("--output", type=str, required=True, help="Path to write the CSV report to.")
    parser.add_argument("--test-dir", type=str, default=None,
                         help="Same value as test_dynamic_2_models.py's -t/--test-folder (e.g. "
                              "Tests/TinyClassifier_x_AnomalyDetection, WITHOUT a TEST_SIRACUSA/ prefix -- "
                              "ThetaLookup.py adds that itself), used to locate each mode's profiling JSONs and "
                              "compute theta. Omit to leave theta/cumulative_theta blank (matches this script's "
                              "behavior before theta support was added).")
    parser.add_argument("--configs", type=str, default=None,
                         help="Path to the --configs JSON used for the run (defaults to the same file "
                              "test_dynamic_2_models.py defaults to). Only used when --test-dir is given.")
    args = parser.parse_args()

    theta_by_mode = None
    if args.test_dir:
        configs_path = Path(args.configs) if args.configs else DEFAULT_CONFIGS_PATH
        configs = load_configs(configs_path)
        theta_by_mode = build_theta_by_mode(configs, Path(args.test_dir))
        print(f"[parse_scenario_log] theta by mode (from {configs_path}): {theta_by_mode}")

    ticks = parse_log(Path(args.log), theta_by_mode)
    if not ticks:
        raise RuntimeError(f"No 'Tick N: operation mode ...' lines found in {args.log} -- "
                            "is this a scenario run log (built with --scenario)?")

    write_csv(ticks, Path(args.output))

    total_energy_j = sum(t["energy_j"] for t in ticks if t["energy_j"] is not None)
    print(f"[parse_scenario_log] Wrote {args.output} ({len(ticks)} ticks)")
    print(f"[parse_scenario_log] Total measured energy: {total_energy_j:.6e} J "
          f"(vs on-device estimate at last tick: {ticks[-1]['total_energy_est_nj']} nJ)")

    if theta_by_mode:
        total_theta = ticks[-1]["cumulative_theta"]
        if total_theta is not None and total_energy_j > 0:
            gamma_eff = total_theta / total_energy_j
            print(f"[parse_scenario_log] Total theta: {total_theta} ops -- "
                  f"gamma_eff = {gamma_eff:.6e} OP/J ({gamma_eff / 1e9:.3f} GOP/J)")


if __name__ == "__main__":
    main()
