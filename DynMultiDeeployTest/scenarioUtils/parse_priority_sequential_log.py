#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Parses a saved GVSoC Priority-Sequential run log (priority_sequential.c, built with
-D GVSOC_EXTRA_FLAGS=--power) into a per-tick CSV.

Unlike every other method in this experiment, only ONE task actually runs each
tick here (true preemption -- see priority_sequential_template.c's module
comment): each row reports which task ran, its own Init/Run cycles and power
(Init is 0/blank when skipped -- same task as the previous tick, no reload), and
theta attributed ONLY to that task's own op count (not both tasks' combined, since
the other task did no work this tick).

Usage:
    python3 test_priority_sequential.py --models <m1> <m2> --scenario Scenarios/static_large \
        2>&1 | tee /tmp/ps.log
    python3 scenarioUtils/parse_priority_sequential_log.py --log /tmp/ps.log --output /tmp/ps_report.csv \
        --model1-tag TinyClassifier_heavy --model2-tag AnomalyDetection_heavy
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from MultiDeeploy.ModelProfile import ModelProfile

FREQUENCY_HZ = 50_000_000.0

# running: 1 = TASK2_RUNNING, 2 = TASK1_RUNNING (RunningTask enum order in the C source).
TICK_RE = re.compile(
    r"\[HOST CORE\] Tick (\d+): running (\d+) \(task1_critical=(\d+), task2_critical=(\d+)\)"
)
INIT_CYCLES_RE = re.compile(r"\[(TASK1|TASK2)\] Runtime InitNetwork: (\d+) cycles")
RUN_CYCLES_RE = re.compile(r"\[(TASK1|TASK2)\] Runtime RunNetwork: (\d+) cycles")
POWER_RE = re.compile(r"@power\.measure_(\d+)@([\d.]+)@")

RUNNING_TASK_NAME = {1: "TASK2", 2: "TASK1"}


def _load_total_ops(profiling_json_path: Path) -> int:
    with open(profiling_json_path) as f:
        profile = ModelProfile.from_dict(json.load(f))
    total_ops = profile.total_ops()
    if total_ops is None:
        raise ValueError(f"{profiling_json_path} has no per-layer 'ops' field -- rerun --gen-step.")
    return total_ops


def parse_log(log_path: Path, theta_task2: Optional[int] = None, theta_task1: Optional[int] = None) -> List[Dict]:
    raw_text = log_path.read_text()

    phi_by_index: Dict[int, float] = {}
    for m in POWER_RE.finditer(raw_text):
        phi_by_index[int(m.group(1))] = float(m.group(2))

    # Same UART-buffering artifact fixed in parse_scenario_log.py.
    cleaned_text = re.sub(r"@power\.measure_\d+@[\d.]+@\n?", "", raw_text)

    ticks: List[Dict] = []
    current: Optional[Dict] = None
    call_index = 0  # counts bracketed InitNetwork/RunNetwork calls in program order

    for line in cleaned_text.splitlines():
        m = TICK_RE.search(line)
        if m:
            if current is not None:
                ticks.append(current)
            tick, running, t1_crit, t2_crit = (int(g) for g in m.groups())
            current = {
                "tick": tick,
                "running_task": RUNNING_TASK_NAME.get(running, f"UNKNOWN_{running}"),
                "task1_critical_in": t1_crit,
                "task2_critical_in": t2_crit,
                "init_cycles": 0,
                "init_phi_watts": None,
                "run_cycles": None,
                "run_phi_watts": None,
                "task1_margin": None,
                "task2_margin": None,
            }
            continue

        if current is None:
            continue

        m = INIT_CYCLES_RE.search(line)
        if m:
            current["init_cycles"] = int(m.group(2))
            current["init_phi_watts"] = phi_by_index.get(call_index)
            call_index += 1
            continue

        m = RUN_CYCLES_RE.search(line)
        if m:
            current["run_cycles"] = int(m.group(2))
            current["run_phi_watts"] = phi_by_index.get(call_index)
            call_index += 1
            continue

        m = re.search(r"Tick \d+ context: task1_margin=(-?\d+) task2_margin=(-?\d+)", line)
        if m:
            current["task1_margin"] = int(m.group(1))
            current["task2_margin"] = int(m.group(2))
            continue

    if current is not None:
        ticks.append(current)

    cumulative_theta = 0
    for row in ticks:
        init_e = ((row["init_cycles"] / FREQUENCY_HZ) * row["init_phi_watts"]
                  if (row["init_cycles"] and row["init_phi_watts"] is not None) else 0)
        run_e = ((row["run_cycles"] / FREQUENCY_HZ) * row["run_phi_watts"]
                 if (row["run_cycles"] is not None and row["run_phi_watts"] is not None) else None)
        row["init_energy_j"] = init_e
        row["run_energy_j"] = run_e
        row["total_energy_j"] = (init_e + run_e) if run_e is not None else None

        theta = None
        if theta_task2 is not None and theta_task1 is not None:
            theta = theta_task2 if row["running_task"] == "TASK2" else theta_task1
        row["theta"] = theta
        if theta is not None:
            cumulative_theta += theta
            row["cumulative_theta"] = cumulative_theta
        else:
            row["cumulative_theta"] = None

    return ticks


def write_csv(ticks: List[Dict], output_path: Path) -> None:
    fieldnames = [
        "tick", "running_task", "task1_critical_in", "task2_critical_in",
        "init_cycles", "init_phi_watts", "init_energy_j",
        "run_cycles", "run_phi_watts", "run_energy_j",
        "total_energy_j", "theta", "cumulative_theta",
        "task1_margin", "task2_margin",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in ticks:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Parse a Priority-Sequential GVSoC log into a per-tick CSV report.")
    parser.add_argument("--log", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--test-dir", type=str, default="Tests/RoundRobin_Large",
                         help="Directory holding <model1-tag>/deeployStates/<model1-tag>_profiling.json etc. "
                              "(TEST_SIRACUSA/ prefix added automatically) -- Priority-Sequential reuses "
                              "Round-Robin-Large's profiling JSONs since it's the same two models.")
    parser.add_argument("--model1-tag", type=str, default=None, help="e.g. TinyClassifier_heavy (Task 2).")
    parser.add_argument("--model2-tag", type=str, default=None, help="e.g. AnomalyDetection_heavy (Task 1).")
    args = parser.parse_args()

    theta_task2 = theta_task1 = None
    if args.model1_tag and args.model2_tag:
        base = Path("TEST_SIRACUSA") / args.test_dir
        theta_task2 = _load_total_ops(base / args.model1_tag / "deeployStates" / f"{args.model1_tag}_profiling.json")
        theta_task1 = _load_total_ops(base / args.model2_tag / "deeployStates" / f"{args.model2_tag}_profiling.json")
        print(f"[parse_priority_sequential_log] theta_task2({args.model1_tag})={theta_task2}, "
              f"theta_task1({args.model2_tag})={theta_task1}")

    ticks = parse_log(Path(args.log), theta_task2, theta_task1)
    if not ticks:
        raise RuntimeError(f"No '[HOST CORE] Tick N: running ...' lines found in {args.log}.")

    write_csv(ticks, Path(args.output))

    total_energy_j = sum(t["total_energy_j"] for t in ticks if t["total_energy_j"] is not None)
    task2_ticks = sum(1 for t in ticks if t["running_task"] == "TASK2")
    task1_ticks = sum(1 for t in ticks if t["running_task"] == "TASK1")
    skipped_inits = sum(1 for t in ticks if t["init_cycles"] == 0)
    print(f"[parse_priority_sequential_log] Wrote {args.output} ({len(ticks)} ticks: "
          f"{task2_ticks} Task2, {task1_ticks} Task1, {skipped_inits} with InitNetwork skipped)")
    print(f"[parse_priority_sequential_log] Total measured energy: {total_energy_j:.6e} J")

    if theta_task2 is not None:
        total_theta = ticks[-1]["cumulative_theta"]
        if total_theta is not None and total_energy_j > 0:
            gamma_eff = total_theta / total_energy_j
            print(f"[parse_priority_sequential_log] Total theta: {total_theta} ops -- "
                  f"gamma_eff = {gamma_eff:.6e} OP/J ({gamma_eff / 1e9:.3f} GOP/J)")


if __name__ == "__main__":
    main()
